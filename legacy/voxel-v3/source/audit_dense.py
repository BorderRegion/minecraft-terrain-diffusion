"""Cross-check 512-block overlapping contexts and material fractions against their source worlds."""
from pathlib import Path
import json,numpy as np

def audit(root='.',expected_worlds=255,min_contexts=5000):
 root=Path(root); files=sorted((root/'coarse_dataset').glob('*.npz'));count=0
 for f in files:
  with np.load(f,allow_pickle=False) as c, np.load(root/'natural_worlds'/f.name,allow_pickle=False) as w:
   h=c['heights'];p=c['material_probs'];coords=c['chunk_coords'];mask=w['valid_chunks'];origin=w['origin_chunk'];g=w['ground'];water=w['water_top'];m=w['material']
   assert str(c['seed'])==f.stem and h.shape==(len(coords),2,64,64) and p.shape==(len(coords),7,64,64) and len(coords)>=16,f.name
   assert np.isfinite(h).all() and np.isfinite(p).all() and np.allclose(p.sum(axis=1),1) and p.min()>=0 and p.max()<=1,f.name
   for i in np.unique(np.linspace(0,len(coords)-1,min(5,len(coords)),dtype=int)):
    x,z=coords[i];x=int(x-origin[0]);z=int(z-origin[1]);valid=mask[z:z+32,x:x+32];assert x>=0 and z>=0 and valid.shape==(32,32) and valid.all(),(f.name,i)
    sl=np.s_[z*16:(z+32)*16,x*16:(x+32)*16];ground=g[sl];ref=np.stack([np.maximum(ground,water[sl]),ground]).astype(np.float32).reshape(2,64,8,64,8).mean((2,4));materials=m[sl];pr=np.stack([(materials==k).reshape(64,8,64,8).mean((1,3)) for k in range(2,9)]).astype(np.float32)
    assert np.array_equal(h[i],ref) and np.array_equal(p[i],pr),(f.name,i)
   count+=len(coords)
 assert len(files)==expected_worlds and count>=min_contexts,(len(files),count)
 return {'worlds':len(files),'contexts':count,'checked_contexts_per_world':5}
if __name__=='__main__':print(json.dumps(audit()),flush=True)
