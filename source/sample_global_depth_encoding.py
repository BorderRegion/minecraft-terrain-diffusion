"""Generate a 512x512 block region with a global DDIM prior and overlapping guided local DDIM patches."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch_musa
except ImportError:
    torch_musa = None
from train import UNet
from sample import export_schematic

def load_model(path,device):
 s=torch.load(path,map_location='cpu',weights_only=False); m=UNet(base=s.get('model_base',64),attention=s.get('attention',False)).to(device);m.load_state_dict(s['ema']);m.eval()
 return m,torch.cumprod(1-torch.linspace(s['beta_start'],s['beta_end'],s['timesteps'],device=device),0),s

def sample(model,alpha,n,steps,device,guide=None,prediction_type="epsilon"):
 x=torch.randn(n,9,64,64,device=device);schedule=torch.linspace(len(alpha)-1,0,steps,device=device).long().tolist()
 with torch.no_grad():
  for i,t in enumerate(schedule):
   a=alpha[t];b=alpha[schedule[i+1]] if i+1<len(schedule) else torch.ones((),device=device)
   pred=model(x,torch.full((n,),t,device=device,dtype=torch.long))
   if prediction_type=='v_prediction':x0=a.sqrt()*x-(1-a).sqrt()*pred;eps=(1-a).sqrt()*x+a.sqrt()*pred
   elif prediction_type=='epsilon':eps=pred;x0=(x-(1-a).sqrt()*eps)/a.sqrt()
   else:raise ValueError('unsupported diffusion prediction type: '+prediction_type)
   x0[:,:2]=x0[:,:2].clamp(-1.,255./128.);x0[:,2:]=x0[:,2:].clamp(-1.5,1.5)
   if guide is not None:
    delta=guide-F.avg_pool2d(x0,8);x0=x0+F.interpolate(delta,scale_factor=8,mode='bicubic',align_corners=False)
    eps=(x-a.sqrt()*x0)/(1-a).sqrt()
   x=b.sqrt()*x0+(1-b).sqrt()*eps
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--global-checkpoint',default='runs/global/best.pt');p.add_argument('--local-checkpoint',default='runs/terrain/best.pt');p.add_argument('--out',default='samples/global');p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=8);p.add_argument('--seed',type=int,default=42026);p.add_argument('--device',default='musa');p.add_argument('--no-decorate',action='store_true');a=p.parse_args()
 if not 2<=a.steps<=1000: p.error('steps must be 2..1000')
 device=torch.device(a.device);torch.manual_seed(a.seed);global_model,ga,gs=load_model(a.global_checkpoint,device);local_model,la,ls=load_model(a.local_checkpoint,device)
 if gs['normalization']!=ls['normalization']:raise ValueError('incompatible normalization')
 if ls.get('height_representation')!='log_depth_ground_v1':raise ValueError('local checkpoint is not log-depth encoded')
 with torch.no_grad():
  coarse=sample(global_model,ga,1,a.steps,device,prediction_type=gs.get('prediction_type','epsilon')); encoded=coarse.clone(); depth=((coarse[:,0]-coarse[:,1])*gs['normalization']['scale']).clamp(0,383); encoded[:,0]=2*torch.log1p(depth)/np.log(384.0)-1; guide=F.interpolate(encoded,size=(512,512),mode='bilinear',align_corners=False)
  locs=[(z,x) for z in range(0,449,32) for x in range(0,449,32)]; total=torch.zeros((1,9,512,512),device=device);weight=torch.zeros((1,1,512,512),device=device)
  ramp=torch.sin((torch.arange(64,device=device)+.5)*torch.pi/64).square().clamp_min(1e-3); window=(ramp[:,None]*ramp[None,:])[None,None]
  for n in range(0,len(locs),a.batch):
   positions=locs[n:n+a.batch]; patches=torch.cat([guide[:,:,z:z+64,x:x+64] for z,x in positions]); target=F.avg_pool2d(patches,8);fine=sample(local_model,la,len(positions),a.steps,device,target,ls.get('prediction_type','epsilon'))
   for i,(z,x) in enumerate(positions):total[:,:,z:z+64,x:x+64]+=fine[i:i+1]*window;weight[:,:,z:z+64,x:x+64]+=window
   print(json.dumps({'patches':min(n+len(positions),len(locs)),'total':len(locs)}),flush=True)
  field=(total/weight).cpu().numpy()[0];depth=np.expm1(np.clip((field[0]+1)*.5,0,1)*np.log(384.0));field[1]=field[1]*ls['normalization']['scale']+ls['normalization']['offset'];field[0]=field[1]+depth;coarse=coarse.cpu().numpy()[0];coarse[:2]=coarse[:2]*gs['normalization']['scale']+gs['normalization']['offset']
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);np.save(out/'global_field.npy',field);np.save(out/'global_coarse.npy',coarse)
 # One continuous schematic makes all chunk boundaries part of one shared block volume.
 whole=export_schematic(field,out/'terrain_full.schem',decorate=not a.no_decorate,seed=a.seed)
 # Optional smaller tiles retain individual origins for editors with size limits.
 meta=[]
 for z in range(0,512,64):
  for x in range(0,512,64):
   info=export_schematic(field[:,z:z+64,x:x+64],out/f'terrain_x{x}_z{z}.schem');meta.append({'x':x,'z':z,**info})
 (out/'manifest.json').write_text(json.dumps({'seed':a.seed,'global_step':gs['step'],'local_step':ls['step'],'tile_size':64,'full':whole,'tiles':meta,'note':'The full schematic may include deterministic vegetation; the separate tiles are undecorated.'},indent=2));print('saved',out,flush=True)
if __name__=='__main__':main()
