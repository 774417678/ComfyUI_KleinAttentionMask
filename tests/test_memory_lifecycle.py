"""CPU-only regression suite; stubs sampling, uses real torch + LoadedModel source.

Covers two things:
  * model-clone / LoadedModel lifecycle (VRAM unload regression)
  * the sampling-record -> influence-map split: parameter changes must never
    need another diffusion pass, and recorded data must reproduce the maps.
"""
import ast
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parent
PLUGIN=ROOT.parent
# For a standalone checkout, point COMFYUI_ROOT at a compatible ComfyUI tree.
# An installation under ComfyUI/custom_nodes is detected automatically.
CORE=Path(os.environ.get('COMFYUI_ROOT', str(PLUGIN.parent.parent)))
SOURCE=PLUGIN/'sampler_variants.py'
INTERNAL=PLUGIN/'internal_sampler.py'
if not (CORE/'comfy/model_management.py').is_file():
    raise RuntimeError('Set COMFYUI_ROOT to a ComfyUI checkout before running this test.')

# This class is AST-extracted unchanged from core, without importing GPU services.
loaded_ast=next(n for n in ast.parse((CORE/'comfy/model_management.py').read_text(encoding='utf8')).body if isinstance(n,ast.ClassDef) and n.name=='LoadedModel')
ns={'weakref':weakref,'ModelPatcher':object}
exec(compile(ast.Module(body=[loaded_ast],type_ignores=[]),'real_loaded_model','exec'),ns)
LoadedModel=ns['LoadedModel']

Flux=type('Flux',(),{'__module__':'comfy.ldm.flux.model'})
class FakeRaw:
    def __init__(self):
        self.diffusion_model=Flux();self.diffusion_model.params=types.SimpleNamespace(context_in_dim=12288)
    def process_latent_in(self,x):return x+0.25

class FakePatcher:
    def __init__(self,raw=None,parent=None):
        self.model=raw or FakeRaw();self.parent=parent;self.model_options={};self.load_device='cuda:0'
        self.clones=[]
    def clone(self):
        out=FakePatcher(self.model,self);out.model_options=copy.copy(self.model_options)
        self.clones.append(weakref.ref(out));return out
    def get_model_object(self,name):return getattr(self.model,name)
    def set_model_unet_function_wrapper(self,fn):self.model_options['model_function_wrapper']=fn

class FakeCFG:
    def __init__(self,model):
        self.model_patcher=model;self.model_options=model.model_options;self.original_conds={'positive':[{'id':1}]}
class Guider_Basic(FakeCFG):pass

class FakeClip:
    patcher=types.SimpleNamespace(forced_hooks=None)
    apply_hooks_to_conds=None
    use_clip_schedule=False
    def tokenize(self,text):return text
    def encode_from_tokens_scheduled(self,tokens):return [[torch.ones(1,7,12288),{}]]

class FakeNodeSampler:
    @classmethod
    def INPUT_TYPES(cls):return {'required':{}}
    def sample(self,*a,**kw):return (a[-1],)

nodes=types.ModuleType('nodes');nodes.KSampler=FakeNodeSampler;nodes.KSamplerAdvanced=FakeNodeSampler
samplers=types.ModuleType('comfy.samplers');samplers.CFGGuider=FakeCFG
comfy=types.ModuleType('comfy');comfy.samplers=samplers
patchers=types.ModuleType('comfy.model_patcher');patchers.create_model_options_clone=copy.deepcopy
custom=types.ModuleType('comfy_extras.nodes_custom_sampler');custom.Guider_Basic=Guider_Basic
custom.SamplerCustomAdvanced=types.SimpleNamespace(execute=lambda *a:None)
custom.SamplerCustom=types.SimpleNamespace(execute=lambda *a:None)
extras=types.ModuleType('comfy_extras');extras.nodes_custom_sampler=custom
package=types.ModuleType('klein_under_test');package.__path__=[str(PLUGIN)]
sys.modules.update({'nodes':nodes,'comfy':comfy,'comfy.samplers':samplers,'comfy.model_patcher':patchers,'comfy_extras':extras,'comfy_extras.nodes_custom_sampler':custom,'klein_under_test':package})

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
internal=load('klein_under_test.internal_sampler',INTERNAL)
record_mod=load('klein_under_test.sampling_record',PLUGIN/'sampling_record.py')
influence_mod=load('klein_under_test.influence_node',PLUGIN/'influence_node.py')
mod=load('klein_under_test.sampler_variants',SOURCE)


def _call(rec,x,ref=None,flags=(0,),apply=None):
    c={'transformer_options':{},'c_crossattn':torch.zeros(x.shape[0],512,12288)}
    if ref is not None:c['ref_latents']=[ref]
    fn=apply or (lambda x,t,**c:x)
    return rec.wrap(fn,{'input':x,'timestep':torch.ones(x.shape[0]),'c':c,'cond_or_uncond':list(flags)})


class MemoryLifecycle(unittest.TestCase):
    def setUp(self):
        self.model=FakePatcher();self.latent={'samples':torch.randn(2,4,3,5)}
    def recorder(self,**kw):return mod.Recorder(self.model,self.latent,4,**kw)
    def test_close_idempotent_and_only_one_clone(self):
        rec=self.recorder();loaded=LoadedModel(rec.patched);loaded.real_model=weakref.ref(self.model.model)
        self.assertIs(rec.patched.parent,self.model);self.assertEqual(len(self.model.clones),1)
        rec.close();rec.close();gc.collect()
        self.assertIs(loaded.model,self.model);self.assertFalse(loaded.is_dead());self.assertIsNone(rec.neutral)
        self.assertNotIn('model_function_wrapper',self.model.model_options)
    def test_validation_does_not_clone(self):
        for opts in ({'model_function_wrapper':lambda *a:None},{'sampler_calc_cond_batch_function':object()}):
            with self.assertRaises(ValueError):self.recorder(model_options=opts)
        self.assertEqual(self.model.clones,[])
    def test_record_neutral_is_cpu_and_left_padded(self):
        with self.recorder(record_neutral=True,clip=FakeClip()) as rec:
            self.assertEqual(rec.neutral.shape,(1,512,12288));self.assertTrue(torch.equal(rec.neutral[:,:505],torch.zeros(1,505,12288)))
            calls=[]
            def apply(x,t,**c):calls.append(1);return x+c['c_crossattn'].mean()
            _call(rec,torch.zeros(2,4,3,5),apply=apply)
            self.assertEqual(len(calls),2)                       # one edit + one neutral forward
            self.assertIsNotNone(rec.record.steps[0]['neutral'])
            self.assertEqual(rec.record.steps[0]['denoised'].device.type,'cpu')
            self.assertNotIn('x_in',rec.record.steps[0])
    def test_prepared_neutral_skips_clip_and_owns_no_copy(self):
        conditioning=[[torch.ones(1,7,12288),{}]]
        class ExplodingClip:
            def tokenize(self,text):raise AssertionError('Prepared input must not touch CLIP')
        with self.recorder(record_neutral=True,neutral_conditioning=conditioning,clip=ExplodingClip()) as rec:
            self.assertEqual(rec.neutral_source,'conditioning')
            rec.neutral.zero_()
            self.assertTrue(torch.equal(conditioning[0][0],torch.ones(1,7,12288)))
        self.assertIsNone(rec.neutral)
    def test_no_neutral_forward_without_flag(self):
        calls=[]
        with self.recorder(neutral_conditioning=[[torch.ones(1,7,12288),{}]]) as rec:
            _call(rec,torch.zeros(2,4,3,5),apply=lambda x,t,**c:(calls.append(1),x)[1])
            self.assertEqual(len(calls),1)
            self.assertFalse(rec.record.neutral_enabled)
    def test_optional_contract(self):
        expected=['record_neutral','clip','source_latent','neutral_conditioning']
        self.assertEqual(list(mod.options()),expected)
        for cls in (mod.KleinMaskSamplerCustomAdvanced,mod.KleinMaskSamplerCustom,mod.KleinMaskKSampler,mod.KleinMaskKSamplerAdvanced):
            self.assertEqual(cls.INPUT_TYPES()['optional']['record_neutral'][0],'BOOLEAN')
            self.assertEqual(len(cls.RETURN_TYPES),len(cls.RETURN_NAMES))
            self.assertIn('sampling_record',cls.RETURN_NAMES)
            for gone in ('mode','threshold','grow','feather','probe_steps','capture_start','capture_end','common_influence','source_reference'):
                self.assertNotIn(gone,cls.INPUT_TYPES()['optional'])
        opt=influence_mod.KleinInfluenceMap.INPUT_TYPES()['optional']
        for name in ('mode','capture_start','capture_end','probe_steps','common_influence','source_reference'):
            self.assertIn(name,opt)
    def _record_two(self,**kw):
        with self.recorder(**kw) as rec:
            for _ in range(2):_call(rec,torch.zeros(2,4,3,5),torch.ones(2,4,3,5))
            record,info=rec.finish()
        return record,json.loads(info)
    def test_reference_latent_is_recorded_and_selected(self):
        record,_=self._record_two()
        ref,origin=record.source_for('auto');self.assertEqual(origin,'reference_latent');self.assertTrue(torch.equal(ref,torch.ones(2,4,3,5)))
        # The sampler input stays available under the explicit choice.
        given,origin=record.source_for('source_latent_input');self.assertEqual(origin,'source_latent_input')
        self.assertFalse(torch.equal(given,ref));self.assertEqual(record.src_origin,'latent_image')
    def test_reference_choice_fails_closed(self):
        latent={'samples':torch.zeros(2,4,3,5)}
        with mod.Recorder(self.model,latent,4) as rec:     # no ReferenceLatent in the call
            _call(rec,torch.zeros(2,4,3,5))
            record,info=rec.finish()
        with self.assertRaises(ValueError):record.source_for('reference_latent')
        with self.assertRaises(ValueError):record.source_for('bogus')
        fallback,origin=record.source_for('auto');self.assertEqual(origin,'latent_image')
    def test_reference_is_sliced_from_cfg_batch(self):
        latent={'samples':torch.zeros(2,4,3,5)}
        with mod.Recorder(self.model,latent,4) as rec:
            _call(rec,torch.zeros(4,4,3,5),torch.cat([torch.full((2,4,3,5),7.),torch.full((2,4,3,5),2.)]),flags=(1,0))
            record,info=rec.finish()
        self.assertTrue(torch.equal(record.reference,torch.full((2,4,3,5),2.)))
    def test_maps_match_manual_math(self):
        record,_=self._record_two()
        maps,info=record_mod.build_maps(record,'source_latent',0,-1,2,False,'auto')
        src,origin=record.source_for('auto')
        expected_caps=[record_mod._rms(s['denoised'],src) for s in record.steps]
        expected=internal.influence_map(expected_caps,48,80,'mean')
        self.assertTrue(torch.equal(maps['source_latent'],expected))
        self.assertEqual(info['source_origin'],origin)
        self.assertEqual(info['captured_calls']['source_latent'],[0,1])
    def test_common_influence_elementwise_min(self):
        a=torch.zeros(1,4,4);b=torch.zeros(1,4,4)
        a[0,:2]=1;b[0,1:3]=1;a[0,3,3]=0.5;b[0,3,3]=0.5
        mean=internal.influence_map([a,b],4,4,'mean');common=internal.influence_map([a,b],4,4,'min')
        self.assertGreater(float(common[0,1].mean()),0.9)
        self.assertLess(float(common[0,0].max()),1e-6)
        self.assertLess(float(common[0,2].max()),1e-6)
        self.assertGreater(float(mean[0,0].mean()),float(common[0,0].mean()))
        with self.assertRaises(ValueError):internal.influence_map([a],4,4,'max')
    def test_adjacent_and_capture_window(self):
        record,_=self._record_two()
        maps,info=record_mod.build_maps(record,'adjacent_prediction',1,-1,2,False,'auto')
        self.assertEqual(info['captured_calls']['adjacent_prediction'],[1])
        self.assertIn('adjacent_prediction',maps)
        maps,info=record_mod.build_maps(record,'source_latent',0,0,2,False,'auto')
        self.assertEqual(info['captured_calls']['source_latent'],[0])
    def test_neutral_modes_need_record_neutral(self):
        record,_=self._record_two()
        with self.assertRaises(ValueError):record_mod.build_maps(record,'sparse_neutral',0,-1,2,False,'auto')
    def test_neutral_modes_reuse_recorded_neutral(self):
        record,_=self._record_two(record_neutral=True,neutral_conditioning=[[torch.ones(1,7,12288),{}]])
        self.assertTrue(record.neutral_enabled)
        first,_=record_mod.build_maps(record,'sparse_neutral',0,-1,2,False,'auto')
        # Recomputing with different parameters must not need (or touch) new data.
        before=len(record.steps);neutral_before=dict(record._neutral_by_index)
        second,info=record_mod.build_maps(record,'dense_neutral',0,-1,5,True,'auto')
        self.assertEqual(len(record.steps),before)
        self.assertEqual(set(record._neutral_by_index),set(neutral_before))
        self.assertEqual(info['combine'],'common_min')
        self.assertEqual(tuple(first['sparse_neutral'].shape),tuple(second['dense_neutral'].shape))
    def test_compare_all_orders_maps(self):
        record,_=self._record_two(record_neutral=True,neutral_conditioning=[[torch.ones(1,7,12288),{}]])
        with patch.object(influence_mod.KleinInfluenceMap,'compute',influence_mod.KleinInfluenceMap.compute):
            heat,text,heats=influence_mod.KleinInfluenceMap().compute(record,mode='compare_all',capture_start=0,capture_end=-1,probe_steps=2,common_influence=False,source_reference='auto')
        info=json.loads(text)
        self.assertEqual(info['comparison_order'],internal.METRICS)
        self.assertEqual(heats.shape[0],len(internal.METRICS)*heat.shape[0])
        self.assertGreaterEqual(float(heat.min()),0.);self.assertLessEqual(float(heat.max()),1.)
    def test_noise_mask_is_applied(self):
        latent={'samples':torch.randn(2,4,3,5),'noise_mask':torch.ones(2,3,5)}
        with mod.Recorder(self.model,latent,4) as rec:
            for _ in range(2):_call(rec,torch.zeros(2,4,3,5),torch.ones(2,4,3,5))
            record,info=rec.finish()
        maps,meta=record_mod.build_maps(record,'source_latent',0,-1,2,False,'auto')
        self.assertTrue(meta['noise_mask_applied'])
        self.assertEqual(tuple(maps['source_latent'].shape),(2,48,80))
    def test_positive_replay_keeps_samples_and_reference_blocks_aligned(self):
        x=torch.arange(4*4*3*5).reshape(4,4,3,5).float();t=torch.tensor([9.,9.,8.,8.]);ref=x+10
        args={'input':x,'timestep':t,'cond_or_uncond':[1,0],'c':{'c_crossattn':torch.zeros(4,512,12),'guidance':torch.arange(4.),'ref_latents':[ref],'ref_latents_method':'index_timestep_zero','transformer_options':{'cond_or_uncond':[1,0],'uuids':['neg','pos'],'sigmas':torch.ones(2)}}}
        nx,nt,c,chunks,pos=mod.positive_replay_args(args,torch.ones(1,512,12))
        self.assertTrue(torch.equal(nx,x[2:]));self.assertTrue(torch.equal(nt,t[2:]));self.assertTrue(torch.equal(c['ref_latents'][0],ref[2:]))
        self.assertTrue(torch.equal(c['guidance'],torch.arange(4.)[2:]));self.assertEqual(c['transformer_options']['uuids'],['pos'])
        self.assertEqual(c['transformer_options']['cond_or_uncond'],[0]);self.assertEqual((chunks,pos),(1,0))
        self.assertEqual(args['c']['transformer_options']['cond_or_uncond'],[1,0]);self.assertEqual(c['c_crossattn'].shape,(2,512,12))
    def test_patch_specific_batches_keep_full_replay(self):
        args={'input':torch.zeros(4,4,3,5),'timestep':torch.ones(4),'cond_or_uncond':[1,0],'c':{'c_crossattn':torch.zeros(4,512,12),'transformer_options':{'patches':{'special':[object()]}}}}
        x,t,c,chunks,pos=mod.positive_replay_args(args,torch.ones(1,512,12));self.assertEqual((chunks,pos),(2,1));self.assertEqual(x.shape[0],4)
    def test_unknown_shapes_fail_closed(self):
        args={'input':torch.zeros(4,4,3,5),'timestep':torch.ones(4),'cond_or_uncond':[1,0],'c':{'c_crossattn':torch.zeros(4,512,12),'bad':torch.zeros(3,6),'transformer_options':{}}}
        with self.assertRaises(ValueError):mod.positive_replay_args(args,torch.ones(1,512,12))
    def check_adapter(self,failure=None):
        guider=FakeCFG(self.model);opts=guider.model_options;conds=guider.original_conds;seen=[];private=[]
        def execute(noise,g,sampler,sigmas,latent):
            lm=LoadedModel(g.model_patcher);lm.real_model=weakref.ref(self.model.model);seen.append(lm);private.append(g)
            if failure:raise failure
            g.model_patcher.model_options['model_function_wrapper'](lambda x,t,**c:x,{'input':latent['samples'],'timestep':torch.ones(2),'c':{},'cond_or_uncond':[0]})
            return latent,latent
        with patch.object(mod.SamplerCustomAdvanced,'execute',execute):
            if failure:
                with self.assertRaises(type(failure)):mod.KleinMaskSamplerCustomAdvanced().sample(None,guider,None,torch.tensor([1.,.5,0.]),self.latent)
            else:
                result=mod.KleinMaskSamplerCustomAdvanced().sample(None,guider,None,torch.tensor([1.,.5,0.]),self.latent)
                self.assertIs(result[0],self.latent);self.assertIn('steps',result[2].describe())
        gc.collect();self.assertIs(seen[0].model,self.model);self.assertFalse(seen[0].is_dead())
        self.assertIs(guider.model_options,opts);self.assertIs(guider.original_conds,conds);self.assertNotIn('model_function_wrapper',opts)
        self.assertEqual(len(self.model.clones),1);self.assertEqual(private[0].__dict__,{})
    def test_adapter_success_returns_tracking_to_upstream(self):self.check_adapter()
    def test_adapter_error_returns_tracking_to_upstream(self):self.check_adapter(RuntimeError('simulated OOM'))
    def test_adapter_cancellation_returns_tracking_to_upstream(self):self.check_adapter(KeyboardInterrupt('simulated cancel'))
    def test_other_adapter_failures_close_recorder(self):
        latent=self.latent;model=self.model;made=[]
        real_rec=mod.Recorder
        class Capture(real_rec):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);made.append(self)
        def fail(*a,**kw):raise RuntimeError('simulate sampler failure')
        with patch.object(mod,'Recorder',Capture),patch.object(nodes.KSampler,'sample',fail),patch.object(nodes.KSamplerAdvanced,'sample',fail),patch.object(mod.SamplerCustom,'execute',fail):
            calls=[lambda:mod.KleinMaskKSampler().sample(model,1,4,1.,'euler','simple',[],[],latent),lambda:mod.KleinMaskKSamplerAdvanced().sample(model,'enable',1,4,1.,'euler','simple',[],[],latent,0,4,'disable'),lambda:mod.KleinMaskSamplerCustom().sample(model,True,1,1.,[],[],None,torch.ones(4),latent)]
            for call in calls:
                with self.assertRaises(RuntimeError):call()
        self.assertEqual(len(made),3)
        for rec in made:self.assertIsNone(rec.patched);self.assertIsNone(rec.neutral);self.assertEqual(rec.kw,{})
    def test_rms_matches_math(self):
        a=torch.randn(2,4,3,5);b=torch.randn_like(a)
        self.assertTrue(torch.equal(internal.rms(a,b),((a-b).square().mean(1).sqrt())))

if __name__=='__main__':unittest.main(verbosity=2)
