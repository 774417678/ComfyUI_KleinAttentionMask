"""Native sampler adapters. Delegate sampling, noise, callbacks and metadata to core."""
import copy
import json
import time
import torch
import torch.nn.functional as F
import nodes
import comfy.samplers
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced, SamplerCustom, Guider_Basic
from comfy.model_patcher import create_model_options_clone
from .internal_sampler import MODES, METRICS, rms, make_heat, pick_steps

MASK_TYPES=('MASK','MASK','STRING','MASK','MASK')
MASK_NAMES=('edit_mask','influence_heatmap','diagnostics','comparison_masks','comparison_heatmaps')
def options():
    return {
        'mode':(MODES,{'default':'source_latent'}),
        'threshold':('FLOAT',{'default':0.20,'min':0.01,'max':0.95,'step':0.01}),
        'grow':('INT',{'default':2,'min':0,'max':64}),
        'feather':('INT',{'default':4,'min':0,'max':64}),
        'probe_steps':('INT',{'default':2,'min':1,'max':100}),
        'capture_start':('INT',{'default':1,'min':0,'max':10000}),
        'capture_end':('INT',{'default':-1,'min':-1,'max':10000}),
        'clip':('CLIP',),
        'source_latent':('LATENT',),
    }

def positive_replay_args(args,neutral):
    """Replay only the positive CFG block for native, unpatched conditioning.

    Unknown batch-shaped values fail closed. Transformer patches can define their
    own batch semantics, so retain the original full-batch path for those.
    """
    flags=args['cond_or_uncond']
    x=args['input'];t=args['timestep'];c=dict(args['c'])
    if flags.count(0)!=1 or x.shape[0]%len(flags):
        raise ValueError('Unsupported conditioning batching for positive-only neutral replay.')
    pos=flags.index(0)
    if c.get('control') is not None:
        raise ValueError('Neutral replay with ControlNet is unsupported.')
    cross=c.get('c_crossattn')
    if cross is None:raise ValueError('No c_crossattn in guider prediction; incompatible guider.')
    to=dict(c.get('transformer_options',{}))
    replay_chunks=len(flags)
    if replay_chunks>1 and not any(to.get(k) for k in ('patches','patches_replace','wrappers')):
        rows=x.shape[0];batch=rows//len(flags);sl=slice(pos*batch,(pos+1)*batch)
        def select(value):
            if torch.is_tensor(value):
                if value.ndim==0:return value
                if value.shape[0]==rows:return value[sl]
                if value.shape[0]==1:return value
                raise ValueError('Unsupported conditioning batch shape for neutral replay: '+str(tuple(value.shape)))
            if isinstance(value,list):return [select(v) for v in value]
            if isinstance(value,tuple):return tuple(select(v) for v in value)
            if value is None or isinstance(value,(str,int,float,bool)):return value
            raise ValueError('Unsupported conditioning type for positive-only neutral replay: '+type(value).__name__)
        c={k:(v if k=='transformer_options' else select(v)) for k,v in c.items()}
        x=x[sl];t=select(t)
        to['cond_or_uncond']=[0]
        if 'uuids' in to:to['uuids']=[to['uuids'][pos]]
        replay_chunks=1;pos=0
    c['c_crossattn']=neutral.to(cross).expand(x.shape[0],-1,-1)
    c['transformer_options']=to
    return x,t,c,replay_chunks,pos

class Recorder:
    def __init__(self,model,latent,total,model_options=None,**kw):
        opts=model.model_options if model_options is None else model_options
        if opts.get('model_function_wrapper') or model.model_options.get('model_function_wrapper'):
            raise ValueError('Existing model_function_wrapper is unsupported; use an unwrapped model.')
        if opts.get('sampler_calc_cond_batch_function'):
            raise ValueError('Custom conditioning executor is unsupported.')
        # Keep scalar settings, not CLIP/LATENT objects, in the recorder's metadata.
        self.kw={k:v for k,v in kw.items() if k not in ('clip','source_latent')}
        self.mode=kw.get('mode','source_latent')
        if self.mode not in MODES:raise ValueError('Unknown mask mode: '+self.mode)
        self.active=METRICS if self.mode=='compare_all' else [self.mode]
        dm=model.get_model_object('diffusion_model')
        if type(dm).__module__!='comfy.ldm.flux.model' or getattr(dm.params,'context_in_dim',0)!=12288:
            raise ValueError('Native Flux2 Klein9B required; third-party/KV engines not validated.')
        src=kw.get('source_latent',latent)['samples']
        if src.is_nested or src.ndim!=4: raise ValueError('Only regular image latents [B,C,H,W] are supported, not nested/video latents.')
        self.src=model.model.process_latent_in(src.detach().cpu().clone()).float()
        self.start=kw.get('capture_start',1)
        requested_end=kw.get('capture_end',-1)
        self.end=1000000000 if requested_end<0 else requested_end
        if self.end<self.start: raise ValueError('capture_end < capture_start')
        sparse_end=max(total-1,self.start) if requested_end<0 else self.end
        self.sparse=pick_steps(self.start,sparse_end,kw.get('probe_steps',2))
        self.neutral=None
        if set(self.active)&{'fixed_neutral','sparse_neutral','dense_neutral'}:
            clip=kw.get('clip')
            if clip is None: raise ValueError('Connect optional CLIP for neutral/compare_all modes; fast modes do not require CLIP.')
            self.neutral=clip.encode_from_tokens_scheduled(clip.tokenize('Keep the image unchanged.'))[0][0].detach().cpu()
            # Match native Flux2.extra_conds: left-pad the text context to 512.
            # Calling apply_model directly bypasses this normal preparation step.
            if self.neutral.shape[1]<512:
                self.neutral=F.pad(self.neutral,(0,0,512-self.neutral.shape[1],0))
        self.caps={m:[] for m in self.active};self.at={m:[] for m in self.active}
        self.sigmas=[];self.base_at=[];self.previous=None;self.fixed=None;self.t=time.perf_counter()
        nm=latent.get('noise_mask')
        self.noise_mask=None if nm is None else nm.detach().cpu()
        # Clone exactly once from the cached upstream patcher. A temporary parent
        # plus a bound-method cycle can be GC'd together; LoadedModel then loses
        # both weakrefs even while the raw model stays cached on the GPU.
        self.patched=model.clone()
        if model_options is not None:
            self.patched.model_options=create_model_options_clone(opts)
        self.patched.set_model_unet_function_wrapper(self.wrap)
    def close(self):
        """Break wrapper ownership on success, errors and cancellation (idempotent)."""
        patched=self.patched
        if patched is not None:
            wrapper=patched.model_options.get('model_function_wrapper')
            if getattr(wrapper,'__self__',None) is self:
                patched.model_options.pop('model_function_wrapper',None)
        self.patched=None
        self.neutral=None;self.fixed=None;self.previous=None;self.src=None
        self.noise_mask=None;self.caps.clear();self.kw.clear()
    def __enter__(self):
        return self
    def __exit__(self,exc_type,exc_value,traceback):
        self.close()
        return False
    def wrap(self,apply,args):
        out=apply(args['input'],args['timestep'],**args['c'])
        flags=args.get('cond_or_uncond',[])
        if 0 not in flags: return out
        if flags.count(0)!=1 or out.shape[0]%len(flags): raise ValueError('Unsupported conditioning batching for internal mask collection.')
        idx=len(self.sigmas);self.sigmas.append(float(args['timestep'].flatten()[0]))
        edited=out.chunk(len(flags))[flags.index(0)]
        if edited.shape!=self.src.shape: raise ValueError('Source latent and positive prediction shapes differ. Connect matching source_latent; regional conditioning is not supported.')
        base=None
        within=self.start<=idx<=self.end
        need=('fixed_neutral' in self.active and idx==0) or (within and ('dense_neutral' in self.active or ('sparse_neutral' in self.active and idx in self.sparse)))
        if need:
            x,t,c,chunks,pos=positive_replay_args(args,self.neutral)
            base=apply(x,t,**c).chunk(chunks)[pos].detach().cpu()
            del x,t,c
            self.base_at.append(idx)
            if idx==0 and 'fixed_neutral' in self.active:self.fixed=base.float()
        # Reduce on CPU: do not allocate full-latent FP32 differences on the GPU
        # or upload source/fixed/previous predictions at every capture.
        edited_cpu=None
        if within:
            for m in self.active:
                ref=None
                if m=='source_latent':ref=self.src
                elif m=='fixed_neutral':ref=self.fixed
                elif m=='adjacent_prediction' and self.previous is not None:ref=self.previous
                elif m=='sparse_neutral' and idx in self.sparse:ref=base
                elif m=='dense_neutral':ref=base
                if ref is not None:
                    if edited_cpu is None:edited_cpu=edited.detach().cpu().float()
                    self.caps[m].append(rms(edited_cpu,ref));self.at[m].append(idx)
        if 'adjacent_prediction' in self.active:
            self.previous=edited_cpu if edited_cpu is not None else edited.detach().cpu().float()
        return out
    def finish(self):
        results={}
        for m in self.active:
            if not self.caps[m]: raise ValueError('No collected predictions for '+m+'. For short/partial sampling set capture_start=0 (adjacent requires at least two calls).')
            mask,heat=make_heat(self.caps[m],self.src.shape[-2]*16,self.src.shape[-1]*16,self.kw.get('threshold',.2),self.kw.get('grow',2),self.kw.get('feather',4))
            if self.noise_mask is not None:
                nm=self.noise_mask.detach().cpu().reshape(-1,1,*self.noise_mask.shape[-2:]).float()
                nm=F.interpolate(nm,size=mask.shape[-2:],mode='bilinear',align_corners=False)[:,0]
                if nm.shape[0]==1:nm=nm.expand(mask.shape[0],-1,-1)
                if nm.shape[0]!=mask.shape[0]:raise ValueError('noise_mask batch differs from source batch')
                mask=mask*nm.clamp(0,1);heat=heat*nm.clamp(0,1)
            results[m]=(mask,heat)
        selected='source_latent' if self.mode=='compare_all' else self.mode
        info=json.dumps({'mode':self.mode,'comparison_order':self.active,'index_semantics':'positive model invocations, not scheduler steps; CFG raw positive branch, not guided result','nominal_capture_window':[self.start,self.end],'captured_calls':self.at,'positive_calls':len(self.sigmas),'extra_batched_neutral_forwards':len(self.base_at),'neutral_calls':self.base_at,'sigmas':self.sigmas,'elapsed_seconds':round(time.perf_counter()-self.t,3),'noise_mask_applied':self.noise_mask is not None,'experimental':True},ensure_ascii=False)
        print('[KleinMaskAdapter] '+info)
        return (*results[selected],info,torch.cat([results[m][0] for m in self.active]),torch.cat([results[m][1] for m in self.active]))

class KleinMaskSamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'noise':('NOISE',),'guider':('GUIDER',),'sampler':('SAMPLER',),'sigmas':('SIGMAS',),'latent_image':('LATENT',)},'optional':options()}
    RETURN_TYPES=('LATENT','LATENT')+MASK_TYPES
    RETURN_NAMES=('output','denoised_output')+MASK_NAMES
    FUNCTION='sample';CATEGORY='KleinMask/sampler variants'
    def sample(self,noise,guider,sampler,sigmas,latent_image,**kw):
        import inspect
        from pathlib import Path
        native_file=Path(inspect.getfile(Guider_Basic)).resolve()
        native_basic=type(guider).__name__=='Guider_Basic' and Path(inspect.getfile(type(guider))).resolve()==native_file
        if type(guider) is not comfy.samplers.CFGGuider and not native_basic:raise ValueError('Only native BasicGuider and CFGGuider supported; DualCFG/custom guiders have different branch semantics.')
        if sigmas.numel()<2:raise ValueError('At least two sigma values required for mask discovery.')
        with Recorder(guider.model_patcher,latent_image,sigmas.numel()-1,model_options=guider.model_options,**kw) as rec:
            # Preserve guider-specific options, but do not create another patcher.
            g=copy.copy(guider)
            g.model_patcher=rec.patched;g.model_options=rec.patched.model_options
            g.original_conds={k:[dict(c) for c in conds] for k,conds in guider.original_conds.items()}
            try:
                result=SamplerCustomAdvanced.execute(noise,g,sampler,sigmas,latent_image)
                return (result[0],result[1],*rec.finish())
            finally:
                # A traceback may keep this private guider alive after an error.
                # Never retain the temporary patcher/conditioning through it.
                g.__dict__.clear()

class KleinMaskKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        d=copy.deepcopy(nodes.KSampler.INPUT_TYPES());d['optional']=options();return d
    RETURN_TYPES=('LATENT',)+MASK_TYPES
    RETURN_NAMES=('samples',)+MASK_NAMES
    FUNCTION='sample';CATEGORY='KleinMask/sampler variants'
    def sample(self,model,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,denoise=1.,**kw):
        with Recorder(model,latent_image,steps,**kw) as rec:
            result=nodes.KSampler().sample(rec.patched,seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,denoise)
            return (*result,*rec.finish())

class KleinMaskKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        d=copy.deepcopy(nodes.KSamplerAdvanced.INPUT_TYPES());d['optional']=options();return d
    RETURN_TYPES=('LATENT',)+MASK_TYPES
    RETURN_NAMES=('samples',)+MASK_NAMES
    FUNCTION='sample';CATEGORY='KleinMask/sampler variants'
    def sample(self,model,add_noise,noise_seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_step,end_at_step,return_with_leftover_noise,**kw):
        total=max(1,min(steps,end_at_step)-start_at_step)
        with Recorder(model,latent_image,total,**kw) as rec:
            result=nodes.KSamplerAdvanced().sample(rec.patched,add_noise,noise_seed,steps,cfg,sampler_name,scheduler,positive,negative,latent_image,start_at_step,end_at_step,return_with_leftover_noise)
            return (*result,*rec.finish())

class KleinMaskSamplerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'model':('MODEL',),'add_noise':('BOOLEAN',{'default':True}),'noise_seed':('INT',{'default':0,'min':0,'max':0xffffffffffffffff,'control_after_generate':True}),'cfg':('FLOAT',{'default':1.,'min':0.,'max':100.,'step':.1}),'positive':('CONDITIONING',),'negative':('CONDITIONING',),'sampler':('SAMPLER',),'sigmas':('SIGMAS',),'latent_image':('LATENT',)},'optional':options()}
    RETURN_TYPES=('LATENT','LATENT')+MASK_TYPES
    RETURN_NAMES=('output','denoised_output')+MASK_NAMES
    FUNCTION='sample';CATEGORY='KleinMask/sampler variants'
    def sample(self,model,add_noise,noise_seed,cfg,positive,negative,sampler,sigmas,latent_image,**kw):
        with Recorder(model,latent_image,max(1,sigmas.numel()-1),**kw) as rec:
            result=SamplerCustom.execute(rec.patched,add_noise,noise_seed,cfg,positive,negative,sampler,sigmas,latent_image)
            return (result[0],result[1],*rec.finish())
