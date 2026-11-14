"""Audit completed seed datasets against their source world and report terrain coverage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def audit(root):
 root=Path(root);files=sorted((root/'natural_dataset').glob('*.npz'));summary=[];hashes=set()
 for f in files:
  seed=f.stem;d=np.load(f,allow_pickle=False);world=np.load(root/'natural_worlds'/f.name,allow_pickle=False);h=d['heights'];m=d['materials'];coords=d['chunk_coords'];g=world['ground'];w=world['water_top'];mat=world['material'];mask=world['valid_chunks'];origin=world['origin_chunk']
  assert str(d['seed'])==seed and h.shape[1:]==(2,64,64) and m.shape==(len(h),64,64) and coords.shape==(len(h),2),seed
  assert np.isin(m,np.arange(2,9)).all() and (h[:,0]>=h[:,1]).all(),seed
  assert len({tuple(p) for p in coords})==len(coords),seed
  for i in np.unique(np.linspace(0,len(h)-1,min(16,len(h)),dtype=int)):
   x,z=coords[i];dx=int(x-origin[0]);dz=int(z-origin[1]);assert dx>=0 and dz>=0 and mask[dz:dz+4,dx:dx+4].shape==(4,4) and mask[dz:dz+4,dx:dx+4].all(),(seed,i)
   sl=np.s_[dz*16:(dz+4)*16,dx*16:(dx+4)*16];assert np.array_equal(h[i,1],g[sl]) and np.array_equal(h[i,0],np.maximum(g[sl],w[sl])) and np.array_equal(m[i],mat[sl]),(seed,i)
  digest=hashlib.sha256(h.tobytes()).hexdigest();assert digest not in hashes,seed;hashes.add(digest)
  coarse=root/'natural_coarse'/f.name
  if coarse.exists():
   c=np.load(coarse); ch=c['heights'];cc=c['chunk_coords'];assert ch.shape[1:]==(2,64,64) and len(ch)==len(cc)
   for i in range(len(ch)):
    x,z=cc[i];dx=int(x-origin[0]);dz=int(z-origin[1]);assert mask[dz:dz+32,dx:dx+32].shape==(32,32) and mask[dz:dz+32,dx:dx+32].all(),(seed,i)
    sl=np.s_[dz*16:(dz+32)*16,dx*16:(dx+32)*16];source=np.stack([np.maximum(g[sl],w[sl]),g[sl]]).astype(np.float32);expected=source.reshape(2,64,8,64,8).mean((2,4));assert np.array_equal(ch[i],expected),(seed,i)
   count=len(ch)
  else:count=0
  summary.append({'seed':seed,'local_tiles':len(h),'global_contexts':count,'valid_chunks':int(mask.sum()),'min_height':int(h[:,1].min()),'max_height':int(h[:,1].max()),'water_fraction':round(float((h[:,0]>h[:,1]).mean()),4),'materials':{str(k):int((m==k).sum()) for k in range(2,9)}})
 return {'seed_count':len(summary),'local_tiles':sum(x['local_tiles'] for x in summary),'global_contexts':sum(x['global_contexts'] for x in summary),'worlds':summary}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='/root/autodl-tmp/minecraft-terrain');p.add_argument('--out',default='dataset-audit.json');a=p.parse_args();result=audit(a.root);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps({'seed_count':result['seed_count'],'local_tiles':result['local_tiles'],'global_contexts':result['global_contexts']}),flush=True)
if __name__=='__main__':main()
