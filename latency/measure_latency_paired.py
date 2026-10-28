"""Paired on/off latency and memory measurement for the Qwen2.5-VL coupler.

Environment variables:
  PROJECT_ROOT    directory containing code/ (with our VLMEvalKit fork in code/VLMEvalKit)
  COUPLER_CKPT    path to the Qwen coupler training_state.pt
  LATENCY_IMAGES  two comma-separated image paths, alternated to build N-image inputs
"""
import os, sys, time, json, torch, numpy as np
REL=os.environ["PROJECT_ROOT"]+"/code"
sys.path.insert(0, REL+"/VLMEvalKit"); sys.path.insert(0, REL)
MP="Qwen/Qwen2.5-VL-3B-Instruct"
CK=os.environ["COUPLER_CKPT"]
IMGS=os.environ["LATENCY_IMAGES"].split(",")
from vlmeval.vlm.qwen2_vl.model import PoseGCAQwen2_5_VL
m=PoseGCAQwen2_5_VL(model_path=MP, coupler_path=CK, zgca_code_path=REL, use_custom_prompt=False)
m.max_new_tokens=5
if hasattr(m,'generate_kwargs'): m.generate_kwargs['max_new_tokens']=5
if hasattr(m,'kwargs'): m.kwargs['max_new_tokens']=5

vis=None
for name,mod in m.model.named_modules():
    if name.endswith("model.visual") or name=="visual": vis=mod; break
vis._forward_hooks.clear(); vis._forward_hooks_with_kwargs.clear()
coupler=m.coupler; FLAG={"on":True}
def hook(module,args_in,kwargs_in,output):
    if not FLAG["on"]: return output
    g=kwargs_in.get("grid_thw")
    if g is None and len(args_in)>1: g=args_in[1]
    if g is None: return output
    mg=module.spatial_merge_size
    ss=(g.prod(-1)//mg**2).tolist()
    if len(ss)<2: return output
    ie=list(torch.split(output.pooler_output, ss)); coupler.reset_pools()
    for i in range(len(ie)-1):
        a=ie[i].unsqueeze(0).to(torch.bfloat16); b=ie[i+1].unsqueeze(0).to(torch.bfloat16)
        ie[i+1]=coupler(a,b).squeeze(0).to(output.pooler_output.dtype)
    output.pooler_output=torch.cat(ie,dim=0); return output
vis.register_forward_hook(hook, with_kwargs=True)

REPS=25; out={}
for n in [2,4,16,32]:
    msg=[{"type":"image","value":IMGS[i%2]} for i in range(n)]
    msg.append({"type":"text","value":"Is the camera moving left or right?"})
    for st in (True,False):
        FLAG["on"]=st
        for _ in range(3): m.generate_inner(msg)
    on,off=[],[]
    for r in range(REPS):
        for st,acc in ((True,on),(False,off)) if r%2==0 else ((False,off),(True,on)):
            FLAG["on"]=st; torch.cuda.synchronize(); t=time.perf_counter()
            m.generate_inner(msg); torch.cuda.synchronize(); acc.append((time.perf_counter()-t)*1000)
    on=np.array(on); off=np.array(off); d=on-off   # paired, same iteration
    out[n]={"off_ms":float(off.mean()),"on_ms":float(on.mean()),
            "paired_delta_ms":float(d.mean()),"paired_sd_ms":float(d.std(ddof=1)),
            "se_ms":float(d.std(ddof=1)/np.sqrt(len(d))),"pct":float(d.mean()/off.mean()*100),"reps":len(d)}
    o=out[n]
    print(f"N={n}: off {off.mean():7.1f}  on {on.mean():7.1f}  paired D {d.mean():+6.1f} +/- {o['se_ms']:.1f} ms (SE)  {o['pct']:+.2f}%", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_paired_ab_latency.json"), "w"), indent=1)
