"""Internal prediction comparisons for native Klein9B; experimental, not TAL."""
import json
import time
import torch
import torch.nn.functional as F
import nodes

MODES = ['source_latent', 'fixed_neutral', 'adjacent_prediction', 'sparse_neutral', 'dense_neutral', 'compare_all']
METRICS = MODES[:-1]

def pick_steps(start, end, count):
    return set(torch.linspace(start, end, min(count, end-start+1)).round().long().tolist())

def rms(a, b):
    return (a.float()-b.float()).square().mean(1).sqrt().detach().cpu()

def influence_map(captures, height, width, combine='mean'):
    """Continuous 0..1 influence map; no binarization.

    combine='mean': average of per-capture normalized maps, then 9px box blur (original).
    combine='min': common influence, i.e. keep only positions strong in EVERY capture;
    bilinear upsample without extra blur, matching the validated offline maps.
    """
    if combine not in ('mean','min'): raise ValueError('combine must be mean or min')
    normalized=[]
    for x in captures:
        flat=x.flatten(1)
        lo=flat.median(1).values[:,None,None]
        hi=torch.quantile(flat,0.995,dim=1)[:,None,None]
        normalized.append(((x-lo)/(hi-lo).clamp_min(1e-7)).clamp(0,1))
    stack=torch.stack(normalized)
    heat=stack.amin(0) if combine=='min' else stack.mean(0)
    heat=F.interpolate(heat[:,None],size=(height,width),mode='bilinear',align_corners=False)[:,0]
    if combine=='mean': heat=F.avg_pool2d(heat[:,None],9,1,4)[:,0]
    return heat

def make_heat(captures, height, width, threshold, grow, feather, combine='mean'):
    """Legacy KleinInternalEditMaskSampler only: influence map plus built-in mask."""
    heat=influence_map(captures,height,width,combine)
    mask=(heat>=threshold).float()
    if grow: mask=F.max_pool2d(mask[:,None],2*grow+1,1,grow)[:,0]
    if feather: mask=F.avg_pool2d(mask[:,None],2*feather+1,1,feather)[:,0]
    return mask,heat

class KleinInternalEditMaskSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{
            'model':('MODEL',),'clip':('CLIP',),
            'positive':('CONDITIONING',),'negative':('CONDITIONING',),'latent_image':('LATENT',),
            'seed':('INT',{'default':20260101,'min':0,'max':0xffffffffffffffff}),
            'steps':('INT',{'default':8,'min':2,'max':100}),
            'threshold':('FLOAT',{'default':0.20,'min':0.01,'max':0.95,'step':0.01}),
            'grow':('INT',{'default':2,'min':0,'max':64}),
            'feather':('INT',{'default':4,'min':0,'max':64}),
        },'optional':{
            'mode':(MODES,{'default':'source_latent'}),
            'probe_steps':('INT',{'default':2,'min':1,'max':100,'tooltip':'Sparse neutral evaluations, evenly spaced inside capture window.'}),
            'start_step':('INT',{'default':1,'min':0,'max':99,'tooltip':'Zero-based first captured prediction; clamped to steps.'}),
            'end_step':('INT',{'default':-1,'min':-1,'max':99,'tooltip':'-1 means final prediction.'}),
        }}
    RETURN_TYPES=('LATENT','MASK','MASK','STRING','MASK','MASK')
    RETURN_NAMES=('samples','edit_mask','influence_heatmap','diagnostics','comparison_masks','comparison_heatmaps')
    FUNCTION='sample'
    CATEGORY='KleinMask/experimental'

    def sample(self,model,clip,positive,negative,latent_image,seed,steps,threshold,grow,feather,
               mode='source_latent',probe_steps=2,start_step=1,end_step=-1):
        if mode not in MODES: raise ValueError(mode)
        if len(positive)!=1: raise ValueError('Use one unscheduled positive conditioning.')
        dm=model.get_model_object('diffusion_model')
        if type(dm).__module__!='comfy.ldm.flux.model' or getattr(dm.params,'context_in_dim',0)!=12288:
            raise ValueError('Native ComfyUI Flux2 Klein9B required.')
        if model.model_options.get('model_function_wrapper'): raise ValueError('An existing model wrapper is present.')
        refs=positive[0][1].get('reference_latents')
        if not refs: raise ValueError('Reference image conditioning required; connect PainterFluxImageEdit without a mask.')
        if latent_image.get('noise_mask') is not None: raise ValueError('First-pass mask discovery requires an unmasked latent.')
        # Never compare raw VAE latents with internal denoised predictions: their coordinate systems differ.
        src=refs[0].detach().cpu()
        if src.shape!=latent_image['samples'].shape:
            raise ValueError('First reference and target latent must have identical batch/channel/spatial shape.')
        source=model.model.process_latent_in(src.clone()).float().cpu()
        start=min(start_step,steps-1)
        end=steps-1 if end_step<0 else min(end_step,steps-1)
        if end<start: raise ValueError('end_step must not precede start_step.')
        sparse=pick_steps(start,end,probe_steps)
        active=METRICS if mode=='compare_all' else [mode]
        use_neutral=bool(set(active)&{'fixed_neutral','sparse_neutral','dense_neutral'})
        neutral=None
        if use_neutral:
            neutral=clip.encode_from_tokens_scheduled(clip.tokenize('Keep the image unchanged.'))[0][0]
        captures={m:[] for m in active}; captured_at={m:[] for m in active}
        calls=[]; baseline_steps=[]; fixed=None; previous=None
        t0=time.perf_counter()
        def wrapped(apply,args):
            nonlocal fixed,previous
            if args.get('cond_or_uncond',[])!=[0]: raise ValueError('Requires CFG=1, one positive pass.')
            c=args['c']; idx=len(calls);calls.append(float(args['timestep'].flatten()[0]))
            edited=apply(args['input'],args['timestep'],**c)
            if edited.shape!=source.shape: raise ValueError('Internal prediction/source latent shape mismatch.')
            in_window=start<=idx<=end
            want_base=('fixed_neutral' in active and idx==0) or (in_window and ('dense_neutral' in active or ('sparse_neutral' in active and idx in sparse)))
            base=None
            if want_base:
                nc=dict(c)
                nc['c_crossattn']=neutral.to(c['c_crossattn']).expand(c['c_crossattn'].shape[0],-1,-1)
                nc['transformer_options']=dict(c.get('transformer_options',{}))
                base=apply(args['input'],args['timestep'],**nc)
                baseline_steps.append(idx)
                if idx==0 and 'fixed_neutral' in active: fixed=base.detach().float().cpu()
            if in_window:
                for m in active:
                    ref=None
                    if m=='source_latent': ref=source.to(edited.device)
                    elif m=='fixed_neutral': ref=fixed.to(edited.device)
                    elif m=='adjacent_prediction' and previous is not None: ref=previous.to(edited.device)
                    elif m=='sparse_neutral' and idx in sparse: ref=base
                    elif m=='dense_neutral': ref=base
                    if ref is not None:
                        captures[m].append(rms(edited,ref));captured_at[m].append(idx)
            if 'adjacent_prediction' in active: previous=edited.detach().float().cpu()
            return edited
        patched=model.clone();patched.set_model_unet_function_wrapper(wrapped)
        samples=nodes.common_ksampler(patched,seed,steps,1.0,'euler','simple',positive,negative,latent_image,denoise=1.0)[0]
        results={}
        for m in active:
            if not captures[m]: raise RuntimeError('No usable comparisons for '+m+'; select a later capture window.')
            results[m]=make_heat(captures[m],src.shape[-2]*16,src.shape[-1]*16,threshold,grow,feather)
        selected='source_latent' if mode=='compare_all' else mode
        mask,heat=results[selected]
        # Comparison batch order is mode-major, then original image batch.
        comparison_masks=torch.cat([results[m][0] for m in active],0)
        comparison_heats=torch.cat([results[m][1] for m in active],0)
        info=json.dumps({'mode':mode,'selected_output':selected,'comparison_order':active,'batch':int(src.shape[0]),
             'capture_window':[start,end],'captured_steps':captured_at,'sigmas':calls,
             'normal_forwards':len(calls),'extra_neutral_forwards':len(baseline_steps),'neutral_steps':baseline_steps,
             'elapsed_seconds':round(time.perf_counter()-t0,3),
             'coverage':{m:results[m][0].mean((1,2)).tolist() for m in active},
             'experimental':True},ensure_ascii=False)
        print('[KleinInternalMask] '+info)
        return samples,mask,heat,info,comparison_masks,comparison_heats
