"""Portable entrypoint for the retained30k inference bundle; uses unchanged model math."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source'))

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2)

def verify():
 manifest=json.loads((BASE/'MANIFEST.json').read_text())
 for row in manifest['files']:
  p=(BASE/row['path']).resolve();assert p.is_relative_to(BASE) and p.is_file(),row['path']
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
 print(json.dumps(dict(all_packaged_files_verified=len(manifest['files']),quality_accepted=False)),flush=True)
 return manifest

def imports():
 import torch
 from surface_guide_unet_v3 import SurfaceGuideUNet
 from conditional_voxel_diffusion_v3 import ConditionalUNet3D
 from rich_codec_factory_v3 import create_codec
 return torch,SurfaceGuideUNet,ConditionalUNet3D,create_codec

def load(name,device):
 torch,surface,conditional,codec=imports()
 path=BASE/'weights'/(name+'.pt');state=torch.load(path,map_location='cpu',weights_only=True)
 model=(codec(state['config']) if name=='codec20k' else (surface if name=='surface20k' else conditional)(**state['config']))
 model.load_state_dict(state['model'],strict=True)
 assert all(torch.isfinite(t).all() for t in state['model'].values())
 model=model.to(device).eval().requires_grad_(False)
 return model

def check(smoke=False):
 torch,*_=imports();torch.set_num_threads(1);rows=[]
 for name in ['surface20k','coarse50k','fine30k','codec20k']:
  model=load(name,'cpu')
  row=dict(name=name,strict_state_load=True,parameters=sum(p.numel() for p in model.parameters()))
  if smoke:
   with torch.no_grad():
    if name=='surface20k':out=model(torch.zeros(1,9,64,64),torch.tensor([500]),torch.tensor([0]))
    elif name=='codec20k':out=model.decode(torch.zeros(1,32,16,16,16))
    else:
     c,k=(13,12) if name=='coarse50k' else (32,17)
     out=model(torch.zeros(1,c,16,16,16),torch.tensor([500]),torch.zeros(1,k,16,16,16),torch.tensor([0]))
    values=out.values() if isinstance(out,dict) else [out]
    assert all(torch.isfinite(x).all() for x in values)
    row['actual_CPU_forward_finite']=True
    row['output_shapes']={k:list(v.shape) for k,v in out.items()} if isinstance(out,dict) else list(out.shape)
   del out,values
  del model;gc.collect();rows.append(row);print(json.dumps(row),flush=True)
 return rows

def generate(args):
 torch,*_=imports()
 if args.device=='musa':import torch_musa
 import numpy as np
 from global_voxel_diffusion_training_v3 import sample_ddim
 from project_coarse_density_cpu_v3 import project_density_cpu
 from sample_fine_voxel_tiled_v3 import sample_tiled_ddim
 from decode_fine_voxel_tiled_v3 import iter_decoded_blocks
 from regional_voxel_sampling_v3 import stage_seed,sample_surface,project_guide,coarse_condition,fine_condition
 from export_rich_voxel_world_v3_playback import export_world
 import nbtlib
 device=args.device
 if device in ['cuda','musa']:
  backend=getattr(torch,device);assert backend.is_available(),device+' is unavailable'
  free,total=backend.mem_get_info();assert free>=18*1024**3,'Full world requires at least18GiB free GPU memory.'
  backend.set_per_process_memory_fraction(min(1.,16*1024**3/total))
 if device=='cpu' and not args.allow_slow_cpu:raise ValueError('Use --allow-slow-cpu explicitly for extremely slow full-world CPU generation.')
 assert shutil.disk_usage(Path(args.out).resolve().parent).free>=16*1024**3,'Keep at least16GiB free output space.'
 out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(args.cpu_threads)
 def progress(stage,**extra):
  row=dict(at_unix=time.time(),stage=stage,**extra)
  p=out/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(p)
  print(json.dumps(row),flush=True)
 def release(model):
  del model;gc.collect()
  if device in ['cuda','musa']:getattr(torch,device).empty_cache()
 def noise(shape,stage):return torch.randn(shape,generator=torch.Generator().manual_seed(stage_seed(args.seed,stage))).to(device)
 started=time.time();vocab=json.loads((BASE/'metadata/vocabulary.json').read_text())
 norm=json.loads((BASE/'metadata/normalization.json').read_text())
 save(out/'run-plan.json',dict(seed=args.seed,profile_id=args.profile,physical_shape_yzx=[384,512,512],
  steps=args.steps,batch=args.batch,guidance=args.guidance,device=device,decoder_cache_tiles=1,
  shared_whole_region_noise=True,release_manifest_sha256=sha(BASE/'MANIFEST.json'),
  cross_backend_bitwise_equality_claimed=False,quality_accepted=False))
 try:
  model=load('surface20k',device);progress('sampling_surface')
  raw=sample_surface(model,noise((1,9,64,64),'surface'),args.profile,args.steps,args.guidance,
   lambda row:progress('sampling_surface',**row))[0].cpu().numpy()
  del model;gc.collect()
  if device in ['cuda','musa']:getattr(torch,device).empty_cache()
  np.save(out/'surface-guide-raw-encoded.npy',raw);guide,report=project_guide(raw,'bounded')
  np.save(out/'surface-guide-projected-physical.npy',guide);save(out/'surface-guide-projection.json',report)
  model=load('coarse50k',device);progress('sampling_coarse')
  condition=torch.from_numpy(coarse_condition(guide)[None]).to(device)
  raw,prob=sample_ddim(model,condition,torch.tensor([args.profile],device=device),noise((1,13,48,64,64),'coarse'),args.steps,args.guidance)
  raw=raw[0].cpu().numpy();prob=prob[0].cpu().numpy();np.save(out/'coarse3d-device-probabilities.npy',prob)
  prob,report=project_density_cpu(raw,prob);np.save(out/'coarse3d-raw-signed.npy',raw);np.save(out/'coarse3d-probabilities.npy',prob)
  save(out/'coarse3d-projection.json',report);del model,condition;gc.collect()
  if device in ['cuda','musa']:getattr(torch,device).empty_cache()
  condition,surfaces=fine_condition(prob,guide);np.save(out/'surface-guide-interpolated512.npy',surfaces)
  model=load('fine30k',device);condition=torch.from_numpy(condition[None]).to(device)
  latent=sample_tiled_ddim(model,condition,args.profile,noise((1,32,96,128,128),'fine'),args.steps,args.guidance,
   batch=args.batch,progress=lambda row:progress('sampling_shared_fine_field',**row))[0].cpu().numpy()
  np.save(out/'fine3d-normalized-latent.npy',latent)
  latent=latent*np.asarray(norm['train_channel_std'],np.float32)[:,None,None,None]+np.asarray(norm['train_channel_mean'],np.float32)[:,None,None,None]
  assert np.isfinite(latent).all();np.save(out/'fine3d-physical-latent.npy',latent)
  del model,condition;gc.collect()
  if device in ['cuda','musa']:getattr(torch,device).empty_cache()
  model=load('codec20k',device)
  arrays={key:np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=dtype,shape=shape)
   for key,dtype,shape in [('semantic',np.uint8,(384,512,512)),('block_states',np.uint16,(384,512,512)),('biomes',np.uint16,(96,128,128))]}
  seen=set();projections=[]
  for i,(origin,decoded) in enumerate(iter_decoded_blocks(model,latent,vocab,output_block=(32,32,64),device=device,cache_tiles=1)):
   assert origin not in seen;seen.add(origin);extent=decoded['semantic'].shape[1:]
   for key,array in arrays.items():
    scale=4 if key=='biomes' else 1
    array[tuple(slice(lo//scale,(lo+n)//scale)for lo,n in zip(origin,extent))]=decoded[key][0]
   projections.append(dict(origin_yzx=list(origin),**decoded['projection']))
   progress('decoding_full_region',completed_output_bricks=i+1,target_output_bricks=1536)
  assert len(seen)==1536
  for a in arrays.values():a.flush()
  save(out/'rich-decode-projection.json',dict(bricks=projections,voxels=384*512*512,output_bricks=1536))
  del model,latent;gc.collect()
  if device in ['cuda','musa']:getattr(torch,device).empty_cache()
  with (BASE/'template/chunk.nbt').open('rb') as f:chunk=nbtlib.File.parse(f)
  result=export_world(out/'world',arrays['block_states'],arrays['biomes'],vocab,[-256,-64,-256],
   BASE/'template/level.dat',chunk,args.seed,'Retained30k learned region;native validation and aesthetics still required.')
  assert result['chunks']==1024
  save(out/'generation-manifest.json',dict(files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
   for p in out.rglob('*')if p.is_file()and p.name!='status.json'},elapsed_seconds=time.time()-started,
   accepted_for_final_delivery=False,model_goal_complete=False))
  progress('generated_requires_native_and_visual_review',world=str(out/'world'))
 except BaseException:
  import traceback
  progress('failed_preserve_partial_output',traceback=traceback.format_exc());raise

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--verify-pack',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--smoke',action='store_true')
 p.add_argument('--device',choices=['musa','cuda','cpu'],default='musa');p.add_argument('--allow-slow-cpu',action='store_true')
 p.add_argument('--seed',type=int,default=202611050102);p.add_argument('--profile',type=int,choices=[0,1],default=0)
 p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=8);p.add_argument('--guidance',type=float,default=1.)
 p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--out')
 a=p.parse_args();verify()
 if a.smoke or a.check:check(a.smoke)
 if a.out:
  assert 2<=a.steps<=1000 and a.batch>0 and a.guidance>0 and a.cpu_threads>0
  import torch
  with torch.inference_mode():generate(a)
 elif not(a.verify_pack or a.check or a.smoke):p.error('Choose --verify-pack, --check, --smoke or --out DIRECTORY.')
if __name__=='__main__':main()
