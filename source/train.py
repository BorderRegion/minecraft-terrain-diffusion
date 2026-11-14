"""Unconditional 64x64 height-field DDPM; each .npz represents one independent world seed."""
import argparse, copy, glob, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
 import torch_musa
except ImportError:
 torch_musa=None

class Residual(nn.Module):
 def __init__(self, a,b,t=256):
  super().__init__(); self.norm1=nn.GroupNorm(8,a); self.conv1=nn.Conv2d(a,b,3,padding=1); self.time=nn.Linear(t,b); self.norm2=nn.GroupNorm(8,b); self.conv2=nn.Conv2d(b,b,3,padding=1); self.skip=nn.Conv2d(a,b,1) if a!=b else nn.Identity()
 def forward(self,x,t):
  h=self.conv1(F.silu(self.norm1(x))); h=h+self.time(F.silu(t))[:,:,None,None]; return self.skip(x)+self.conv2(F.silu(self.norm2(h)))
class SpatialAttention(nn.Module):
 def __init__(self,channels,heads=8):
  super().__init__();self.heads=heads;self.norm=nn.GroupNorm(8,channels);self.qkv=nn.Conv2d(channels,channels*3,1);self.proj=nn.Conv2d(channels,channels,1)
 def forward(self,x):
  b,c,h,w=x.shape;v=self.qkv(self.norm(x)).reshape(b,3,self.heads,c//self.heads,h*w);q=v[:,0].permute(0,1,3,2).reshape(b*self.heads,h*w,c//self.heads);k=v[:,1].reshape(b*self.heads,c//self.heads,h*w);value=v[:,2].permute(0,1,3,2).reshape(b*self.heads,h*w,c//self.heads)
  a=torch.bmm(q,k)*(c//self.heads)**-.5;a=a.softmax(dim=-1);y=torch.bmm(a,value).reshape(b,self.heads,h*w,c//self.heads).permute(0,1,3,2).reshape(b,c,h,w);return x+self.proj(y)
class UNet(nn.Module):
 def __init__(self,base=64,attention=False):
  super().__init__(); self.time=nn.Sequential(nn.Linear(base,256),nn.SiLU(),nn.Linear(256,256)); self.base=base
  self.input=nn.Conv2d(9,base,3,padding=1); self.r1=Residual(base,base); self.d1=nn.Conv2d(base,base,3,stride=2,padding=1); self.r2=Residual(base,base*2); self.d2=nn.Conv2d(base*2,base*2,3,stride=2,padding=1); self.r3=Residual(base*2,base*4); self.mid=Residual(base*4,base*4); self.attn=SpatialAttention(base*4) if attention else nn.Identity(); self.u2=Residual(base*6,base*2); self.u1=Residual(base*3,base); self.out=nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv2d(base,9,3,padding=1))
 def forward(self,x,t):
  half=self.base//2; freq=torch.exp(-math.log(10000)*torch.arange(half,device=x.device)/(half-1)); v=t.float()[:,None]*freq[None,:]; emb=self.time(torch.cat([v.sin(),v.cos()],dim=1))
  a=self.r1(self.input(x),emb); b=self.r2(self.d1(a),emb); c=self.attn(self.mid(self.r3(self.d2(b),emb),emb)); c=F.interpolate(c,scale_factor=2,mode='nearest'); c=self.u2(torch.cat((c,b),1),emb); c=F.interpolate(c,scale_factor=2,mode='nearest'); return self.out(self.u1(torch.cat((c,a),1),emb))

def load_data(paths):
 samples=[]
 for path in paths:
  with np.load(path,allow_pickle=False) as d:
   h=d['heights'].astype(np.float32)
   if h.ndim!=4 or h.shape[1:]!=(2,64,64): raise ValueError(path)
   if 'materials' in d:
    m=d['materials']; probs=np.stack([(m==k).astype(np.float32) for k in range(2,9)],axis=1)
   elif 'material_probs' in d: probs=d['material_probs'].astype(np.float32)
   else: raise ValueError(f'missing material data: {path}')
   if probs.shape!=(len(h),7,64,64): raise ValueError('invalid material shape '+path)
   if not np.isfinite(h).all() or not np.isfinite(probs).all() or h.min() < -64 or h.max() > 319 or probs.min() < 0 or probs.max() > 1 or np.max(np.abs(probs.sum(axis=1)-1))>1e-5: raise ValueError('invalid terrain or material probabilities: '+path)
   x=np.concatenate(((h-64)/128,probs*2-1),axis=1)
   samples.append(x)
 return torch.from_numpy(np.concatenate(samples))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='natural_dataset/*.npz');p.add_argument('--out',default='runs/terrain');p.add_argument('--steps',type=int,default=100000);p.add_argument('--batch',type=int,default=64);p.add_argument('--base',type=int,default=320);p.add_argument('--save-every',type=int,default=1000);p.add_argument('--device',default='musa');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.data.startswith(('natural_','coarse_','warmup_')):
  a.batch=max(a.batch,64)
  if not a.data.startswith('warmup_'): a.steps=max(a.steps,100000)
  print(json.dumps({'effective_batch':a.batch,'model_base':a.base,'target_steps':a.steps}),flush=True)
 if a.data.startswith('natural_') or a.data.startswith('coarse_'):
  while not Path('semantic_expanded_ready').exists():
   print('waiting for verified material-aware datasets',flush=True); time.sleep(60)
 if a.data.startswith('natural_') or a.data.startswith('coarse_'):
  from audit_dataset import audit
  report=audit('.')
  if report['seed_count']!=256 or report['global_contexts']<255: raise RuntimeError('dataset audit incomplete')
  Path('dataset-audit.json').write_text(json.dumps(report,indent=2)); print('dataset audit passed',report['seed_count'],flush=True)
  if a.data.startswith('coarse_'):
   from audit_dense import audit as audit_global
   dense=audit_global(); Path('dense-audit.json').write_text(json.dumps(dense,indent=2));print('dense dataset audit passed',dense,flush=True)
 paths=sorted(glob.glob(a.data));
 if len(paths)<3: raise RuntimeError('Need at least 3 separate seed files for train and validation')
 val_paths=[p for p in paths if int.from_bytes(hashlib.sha256(Path(p).stem.encode()).digest()[:8],'big')%10==0]; train_paths=[p for p in paths if p not in val_paths]
 if not val_paths or not train_paths: raise RuntimeError('Seed-hash split has an empty partition')
 train=load_data(train_paths); val=load_data(val_paths)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True); device=torch.device(a.device); random.seed(1024); torch.manual_seed(1024); net=UNet(base=a.base,attention=True).to(device); warm_start=None
 if a.data.startswith('natural_') and not (a.resume and (out/'latest.pt').exists()):
  while (Path('warmup_started').exists() and not Path('warmup_done').exists()) or (Path('vprobe_extended_started').exists() and not Path('vprobe_extended_done').exists()):
   print('waiting for warmup process to finish',flush=True);time.sleep(60)
  warm=Path('runs/vprobe/best.pt')
  if warm.exists():
   w=torch.load(warm,map_location=device,weights_only=False);
   if w.get('prediction_type')!='v_prediction':raise ValueError('warm start uses incompatible diffusion target')
   net.load_state_dict(w['ema']);warm_start={'checkpoint':str(warm),'step':w['step']};print('initialized from warmup',warm_start,flush=True)
 if a.data.startswith('coarse_') and not (a.resume and (out/'latest.pt').exists()):
  while Path('global_warmup_started').exists() and not Path('global_warmup_done').exists():
   print('waiting for global warmup process to finish',flush=True);time.sleep(60)
  warm=Path('runs/global_warmup/best.pt')
  while Path('global_extended_started').exists() and not Path('global_extended_done').exists():
   if Path('global_extended_failed').exists():raise RuntimeError('expanded global pretraining failed; inspect global-extended.log')
   print('waiting for expanded global pretraining to finish',flush=True);time.sleep(60)
  if Path('global_extended_done').exists():
   warm=Path('runs/global_extended/best.pt')
   if not warm.exists():raise RuntimeError('expanded global pretraining finished without a best checkpoint')
  if warm.exists():
   w=torch.load(warm,map_location=device,weights_only=False)
   if w.get('prediction_type')!='v_prediction':raise ValueError('global warm start uses incompatible diffusion target')
   net.load_state_dict(w['ema']);warm_start={'checkpoint':str(warm),'step':w['step']};print('initialized from global warmup',warm_start,flush=True)
 ema=copy.deepcopy(net).eval().requires_grad_(False); opt=torch.optim.AdamW(net.parameters(),lr=0.0001); beta=torch.linspace(0.0001,0.02,1000,device=device); abar=torch.cumprod(1-beta,dim=0); step=0; best_val=float('inf')
 print(json.dumps({'train_seeds':train_paths,'validation_seeds':val_paths,'train_tiles':len(train),'val_tiles':len(val),'device':str(device)}),flush=True)
 ck=out/'latest.pt'
 if a.resume and ck.exists():
  state=torch.load(ck,map_location=device,weights_only=False); net.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);step=state['step'];best_val=state.get('best_val',float('inf'))
  if 'python_rng' in state: random.setstate(state['python_rng']);torch.set_rng_state(state['cpu_rng'].cpu());torch.musa.set_rng_state(state['musa_rng'].cpu()) if state['musa_rng'] is not None else None
 def loss(batch):
  batch=batch.to(device); t=torch.randint(0,1000,(len(batch),),device=device); noise=torch.randn_like(batch); alpha=abar[t,None,None,None]; noisy=alpha.sqrt()*batch+(1-alpha).sqrt()*noise; return F.mse_loss(net(noisy,t),alpha.sqrt()*noise-(1-alpha).sqrt()*batch)
 start=time.time()
 while step<a.steps:
  idx=torch.randint(len(train),(a.batch,)); batch=train[idx]; k=random.randrange(4);batch=torch.rot90(batch,k,(-2,-1));
  if random.random()<0.5: batch=batch.flip(-1)
  net.train(); opt.zero_grad(set_to_none=True); l=loss(batch);
  if not torch.isfinite(l): raise FloatingPointError(f'non-finite loss at step {step}')
  l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1,error_if_nonfinite=True);opt.step();step+=1
  with torch.no_grad():
   for e,q in zip(ema.parameters(),net.parameters()):e.lerp_(q,0.001)
  if step%100==0: print(json.dumps({'step':step,'loss':l.item(),'seconds':round(time.time()-start,1)}),flush=True)
  if step%a.save_every==0 or step==a.steps:
   net.eval()
   gen=torch.Generator(device='cpu').manual_seed(40917); scores=[]
   with torch.no_grad():
    for _ in range(8):
     size=min(a.batch,len(val)); idx=torch.randint(len(val),(size,),generator=gen); t=torch.randint(0,1000,(size,),generator=gen).to(device); batch=val[idx].to(device); noise=torch.randn(batch.shape,generator=gen).to(device); alpha=abar[t,None,None,None]; noisy=alpha.sqrt()*batch+(1-alpha).sqrt()*noise; scores.append(F.mse_loss(ema(noisy,t),alpha.sqrt()*noise-(1-alpha).sqrt()*batch).item())
   v=float(np.mean(scores)); is_best=v<best_val; best_val=min(best_val,v)
   print(json.dumps({'step':step,'validation_velocity_mse_ema':v,'best_validation_velocity_mse_ema':best_val}),flush=True)
   state={'step':step,'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'train_files':train_paths,'val_files':val_paths,'normalization':{'offset':64,'scale':128},'beta_start':0.0001,'beta_end':0.02,'timesteps':1000,'best_val':best_val,'batch_size':a.batch,'model_base':a.base,'attention':True,'warm_start':warm_start,'python_rng':random.getstate(),'cpu_rng':torch.get_rng_state(),'musa_rng':torch.musa.get_rng_state() if device.type=='musa' else None,'channels':9,'prediction_type':'v_prediction','material_classes':[2,3,4,5,6,7,8]}
   tmp=out/'latest.tmp';torch.save(state,tmp);os.replace(tmp,ck)
   if is_best:
    best_tmp=out/'best.tmp';best_tmp.unlink(missing_ok=True);os.link(ck,best_tmp);os.replace(best_tmp,out/'best.pt')
if __name__=='__main__':main()
