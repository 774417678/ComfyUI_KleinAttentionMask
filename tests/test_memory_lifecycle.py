"""CPU-only regression suite; stubs sampling, uses real torch + LoadedModel source."""
import ast
import copy
import gc
import importlib.util
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
    def tokenize(self,text):return text
    def encode_from_tokens_scheduled(self,tokens):return [[torch.ones(1,7,12),{}]]

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
mod=load('klein_under_test.sampler_variants',SOURCE)

class MemoryLifecycle(unittest.TestCase):
    def setUp(self):
        self.model=FakePatcher();self.latent={'samples':torch.randn(2,4,3,5)}
    def recorder(self,**kw):return mod.Recorder(self.model,self.latent,4,capture_start=0,capture_end=1,**kw)
    def test_close_idempotent_and_only_one_clone(self):
        rec=self.recorder();loaded=LoadedModel(rec.patched);loaded.real_model=weakref.ref(self.model.model)
        self.assertIs(rec.patched.parent,self.model);self.assertEqual(len(self.model.clones),1)
        rec.close();rec.close();gc.collect()
        self.assertIs(loaded.model,self.model);self.assertFalse(loaded.is_dead());self.assertIsNone(rec.src)
        self.assertNotIn('model_function_wrapper',self.model.model_options)
    def test_validation_does_not_clone(self):
        for opts in ({'model_function_wrapper':lambda *a:None},{'sampler_calc_cond_batch_function':object()}):
            with self.assertRaises(ValueError):self.recorder(model_options=opts)
        self.assertEqual(self.model.clones,[])
    def test_neutral_and_captures_are_cpu(self):
        with self.recorder(mode='sparse_neutral',clip=FakeClip()) as rec:
            self.assertEqual(rec.neutral.shape,(1,512,12));self.assertTrue(torch.equal(rec.neutral[:,:505],torch.zeros(1,505,12)))
            x=torch.randn(2,4,3,5)
            calls=[]
            def apply(x,t,**c):calls.append(c);return x+c['c_crossattn'].mean()
            out=rec.wrap(apply,{'input':x,'timestep':torch.ones(2),'c':{'c_crossattn':torch.zeros(2,512,12),'transformer_options':{}},'cond_or_uncond':[0]})
            self.assertTrue(torch.equal(out,x));self.assertEqual(len(calls),2)
            self.assertEqual(rec.caps['sparse_neutral'][0].device.type,'cpu')
            result=rec.finish();self.assertEqual(result[0].shape,(2,48,80))
        self.assertEqual(rec.caps,{})
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
                with self.assertRaises(type(failure)):mod.KleinMaskSamplerCustomAdvanced().sample(None,guider,None,torch.tensor([1.,.5,0.]),self.latent,mode='source_latent',capture_start=0)
            else:
                result=mod.KleinMaskSamplerCustomAdvanced().sample(None,guider,None,torch.tensor([1.,.5,0.]),self.latent,mode='source_latent',capture_start=0)
                self.assertIs(result[0],self.latent)
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
        for rec in made:self.assertIsNone(rec.patched);self.assertIsNone(rec.src);self.assertEqual(rec.caps,{})
    def test_rms_matches_math(self):
        a=torch.randn(2,4,3,5);b=torch.randn_like(a)
        self.assertTrue(torch.equal(internal.rms(a,b),((a-b).square().mean(1).sqrt())))

if __name__=='__main__':unittest.main(verbosity=2)
