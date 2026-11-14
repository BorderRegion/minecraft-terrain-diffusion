"""Recover solid ground and surface materials from actual block palettes, excluding vegetation."""
import argparse, collections, gzip, io, json, struct, zlib
from pathlib import Path
import numpy as np
import nbtlib

def classify(name):
 s=name.split(':')[-1]
 if s in ('water','bubble_column'):return 1
 if s in ('grass_block','moss_block','mycelium','podzol'):return 2
 if s in ('dirt','coarse_dirt','rooted_dirt','mud','packed_mud','clay','farmland'):return 3
 if s in ('sand','red_sand','sandstone','red_sandstone'):return 4
 if s in ('snow_block','snow','ice','packed_ice','blue_ice','powder_snow'):return 6
 if s in ('gravel','suspicious_gravel'):return 7
 if 'terracotta' in s:return 8
 if s.endswith('_ore') or any(t in s for t in ('stone','slate','basalt','granite','diorite','andesite','tuff','calcite','bedrock','obsidian','magma_block','amethyst_block','dripstone_block','netherrack')):return 5
 return 0

def sections(path):
 with open(path,'rb') as f:
  hdr=f.read(4096)
  for i in range(1024):
   off=int.from_bytes(hdr[4*i:4*i+4],'big')>>8
   if not off:continue
   f.seek(off*4096); header=f.read(5)
   if len(header)!=5:continue
   length,typ=struct.unpack('>IB',header);data=f.read(length-1)
   try: raw={1:gzip.decompress,2:zlib.decompress,3:lambda x:x}[typ](data); n=nbtlib.File.parse(io.BytesIO(raw))
   except Exception as e:raise RuntimeError(f'{path} chunk {i}') from e
   if str(n.get('Status','')).split(':')[-1]!='full':continue
   yield n

def terrain(n, unknown):
 ground=np.full((16,16),-32768,np.int16); surface=np.zeros((16,16),np.uint8); water=np.full((16,16),-32768,np.int16)
 for sec in sorted(n.get('sections',[]),key=lambda s:int(s['Y']),reverse=True):
  states=sec.get('block_states');
  if not states:continue
  names=[str(t['Name']) for t in states['palette']];classes=np.array([classify(x) for x in names],dtype=np.uint8)
  for name,c in zip(names,classes):
   if c==0:unknown[name]+=1
  if not np.any(classes):continue
  sy=int(sec['Y'])*16
  if len(names)==1: ids=np.zeros((16,16,16),np.uint16)
  else:
   bits=max(4,(len(names)-1).bit_length());per=64//bits
   longs=np.asarray(states['data'],dtype=np.int64).view(np.uint64); j=np.arange(4096,dtype=np.int64)
   ids=((longs[j//per] >> ((j%per)*bits).astype(np.uint64)) & ((1<<bits)-1)).astype(np.uint16).reshape(16,16,16)
  cats=classes[ids]
  for y in range(15,-1,-1):
   layer=cats[y];take=(ground==-32768)&(layer>1);ground[take]=sy+y;surface[take]=layer[take]
   wet=(water==-32768)&(layer==1);water[wet]=sy+y
  if (ground!=-32768).all() and (water!=-32768).all():break
 return ground,surface,water

def main():
 p=argparse.ArgumentParser();p.add_argument('region');p.add_argument('output');a=p.parse_args();parts={};unknown=collections.Counter()
 for f in Path(a.region).glob('r.*.*.mca'):
  for n in sections(f):parts[int(n['xPos']),int(n['zPos'])]=terrain(n,unknown)
 if not parts:raise ValueError('no full chunks')
 xs=[x for x,z in parts];zs=[z for x,z in parts];xmin=min(xs);zmin=min(zs);mx=max(xs)-xmin+1;mz=max(zs)-zmin+1
 ground=np.full((mz*16,mx*16),-32768,np.int16);water=ground.copy(); material=np.zeros_like(ground,dtype=np.uint8);valid=np.zeros((mz,mx),bool)
 for (x,z),(g,c,w) in parts.items():
  x-=xmin;z-=zmin;ss=(slice(z*16,z*16+16),slice(x*16,x*16+16));ground[ss]=g;material[ss]=c;water[ss]=w;valid[z,x]=True
 if (ground.reshape(mz,16,mx,16).transpose(0,2,1,3)[valid]==-32768).any():raise RuntimeError('missing ground in full chunks')
 with open(a.output,'wb') as f: np.savez_compressed(f,ground=ground,material=material,water_top=water,valid_chunks=valid,origin_chunk=np.array([xmin,zmin]))
 print(json.dumps({'chunks':len(parts),'shape':list(ground.shape),'unknown_palette':unknown.most_common(50)}),flush=True)
if __name__=='__main__':main()
