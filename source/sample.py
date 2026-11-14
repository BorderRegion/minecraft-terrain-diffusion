"""DDIM sample and export individual 64x64 tiles as Sponge v2 schematics."""
import argparse, gzip, json, math, sys
from pathlib import Path
import numpy as np
import torch
try:
    import torch_musa
except ImportError:
    torch_musa = None
from train import UNet
sys.path.append(str(Path(__file__).resolve().parent / "vendor"))
import nbtlib
from nbtlib import Compound, Int, Short, ByteArray, IntArray, List

def export_schematic(field,path,decorate=False,seed=0):
 floor=np.clip(np.rint(field[1]),-64,319).astype(int)
 surface=np.clip(np.rint(field[0]),-64,319).astype(int)
 # Preserve the model's wet mask through height quantization, including one-block shallows.
 water=field[0]>field[1]+1
 if np.any(water & (surface<=floor)):raise ValueError('water cannot fit within rounded Minecraft height bounds')
 material=np.argmax(field[2:9],axis=0)+2 if field.shape[0]>=9 else np.full_like(floor,2)
 lo=max(-64,int(floor.min())-12); hi=max(int(floor.max()),int(surface[water].max()) if water.any() else -64); hi=min(319,hi+(9 if decorate else 0));height=hi-lo+1
 length,width=floor.shape; blocks=np.zeros((height,length,width),dtype=np.uint8)
 palette=['minecraft:air','minecraft:stone','minecraft:dirt','minecraft:grass_block[snowy=false]','minecraft:water[level=0]','minecraft:sand','minecraft:snow_block','minecraft:gravel','minecraft:terracotta','minecraft:oak_log[axis=y]','minecraft:oak_leaves[distance=7,persistent=true,waterlogged=false]','minecraft:spruce_log[axis=y]','minecraft:spruce_leaves[distance=7,persistent=true,waterlogged=false]','minecraft:grass','minecraft:poppy']
 top_lookup=np.array([1,1,3,2,5,1,6,7,8],dtype=np.uint8);top=top_lookup[material.clip(0,8)]
 for k in range(height):
  y=lo+k;depth=floor-y;layer=blocks[k];layer[depth>=0]=1
  near=(depth>=0)&(depth<=3);layer[near&np.isin(material,[2,3])]=2;layer[near&(material==4)]=5
  layer[depth==0]=top[depth==0];layer[(depth<0)&water&(surface>=y)]=4
 trees=0; plants=0
 if decorate:
  rng=np.random.default_rng(seed)
  # Plant in world-space cells; each selected crown is wholly inside this schematic.
  for z0 in range(4,length-4,8):
   for x0 in range(4,width-4,8):
    if rng.random()>=.40: continue
    z=min(length-4,z0+int(rng.integers(0,4)));x=min(width-4,x0+int(rng.integers(0,4)));y=int(floor[z,x]);k=y-lo
    if water[z,x] or material[z,x]!=2 or y<63 or y>155 or k+8>=height:continue
    area=floor[z-2:z+3,x-2:x+3]
    if area.max()-area.min()>4:continue
    spruce=y>=98; trunk=7 if spruce else 5;log=11 if spruce else 9;leaf=12 if spruce else 10
    if np.any(blocks[k+1:k+trunk+2,z,x]!=0):continue
    blocks[k+1:k+trunk+1,z,x]=log
    for dy in range(2,trunk+3):
     yy=k+dy
     if yy>=height:continue
     radius=(1 if dy>=trunk else 2) if spruce else (2 if dy<trunk+1 else 1)
     for dz in range(-radius,radius+1):
      for dx in range(-radius,radius+1):
       if abs(dx)==radius and abs(dz)==radius and rng.random()<.4:continue
       if blocks[yy,z+dz,x+dx]==0:blocks[yy,z+dz,x+dx]=leaf
    trees+=1
  # Sparse ground cover adds detail while retaining open grasslands.
  for z0 in range(1,length-1,3):
   for x0 in range(1,width-1,3):
    if rng.random()>=.08 or water[z0,x0] or material[z0,x0]!=2:continue
    k=int(floor[z0,x0])-lo+1
    if 0<=k<height and blocks[k,z0,x0]==0:blocks[k,z0,x0]=14 if rng.random()<.08 else 13;plants+=1
 root=nbtlib.File({'Version':Int(2),'DataVersion':Int(3465),'Width':Short(width),'Height':Short(height),'Length':Short(length),'Offset':IntArray([0,0,0]),'PaletteMax':Int(len(palette)),'Palette':Compound({s:Int(i) for i,s in enumerate(palette)}),'BlockData':ByteArray(blocks.reshape(-1).view(np.int8)),'BlockEntities':List[Compound]([]),'Entities':List[Compound]([])},root_name='Schematic')
 with gzip.open(str(path),'wb',compresslevel=1) as f: root.write(f)
 return {'min_y':lo,'max_y':hi,'water_columns':int(water.sum()),'water_policy':'float_depth_gt_1','blocks':int(blocks.size),'decorated':bool(decorate),'trees':trees,'plants':plants}

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/terrain/best.pt');p.add_argument('--output',default='samples');p.add_argument('--count',type=int,default=4);p.add_argument('--steps',type=int,default=100);p.add_argument('--seed',type=int,default=271828);p.add_argument('--device',default='musa');p.add_argument('--weights',choices=['ema','model'],default='ema');a=p.parse_args()
 if not 2<=a.steps<=1000: p.error('steps must be between 2 and 1000')
 dev=torch.device(a.device);s=torch.load(a.checkpoint,map_location='cpu',weights_only=False);net=UNet(base=s.get('model_base',64),attention=s.get('attention',False)).to(dev);net.load_state_dict(s[a.weights]);net.eval();torch.manual_seed(a.seed)
 beta=torch.linspace(s['beta_start'],s['beta_end'],s['timesteps'],device=dev);alpha=torch.cumprod(1-beta,0);steps=torch.linspace(s['timesteps']-1,0,a.steps,device=dev).long().tolist();x=torch.randn(a.count,s.get('channels',9),64,64,device=dev)
 with torch.no_grad():
  for i,t in enumerate(steps):
   at=alpha[t];prev=alpha[steps[i+1]] if i+1<len(steps) else torch.tensor(1.,device=dev);pred=net(x,torch.full((a.count,),t,device=dev,dtype=torch.long));
   if s.get('prediction_type','epsilon')=='v_prediction': x0=at.sqrt()*x-(1-at).sqrt()*pred;pred=(1-at).sqrt()*x+at.sqrt()*pred
   else: x0=(x-(1-at).sqrt()*pred)/at.sqrt()
   x0[:,:2]=x0[:,:2].clamp(-1.0,255.0/128.0);x0[:,2:]=x0[:,2:].clamp(-1.5,1.5);x=prev.sqrt()*x0+(1-prev).sqrt()*pred
 h=x.cpu().numpy();h[:,:2]=h[:,:2]*s['normalization']['scale']+s['normalization']['offset'];out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for i,field in enumerate(h):
  np.save(out/f'terrain_{i:03}.npy',field);info=export_schematic(field,out/f'terrain_{i:03}.schem');print(json.dumps({'sample':i,'range':[float(field.min()),float(field.max())],**info}),flush=True)
if __name__=='__main__':main()
