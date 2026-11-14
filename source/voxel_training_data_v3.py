"""Validate provenance and sample real, seed-separated voxel cubes."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def audit_dataset(root):
    root=Path(root);index_path=root/'index-pilot64-v1.json'
    index=json.loads(index_path.read_text());seen=set();splits={'train':set(),'validation':set()}
    hist={k:np.zeros(22,np.int64) for k in splits};unknown={};union={k:0 for k in splits}
    worlds=[];height_checks=0;underground_air=0
    shapes={'voxels.npy':((128,64,64,64),'uint8'),'biomes.npy':((128,16,16,16),'uint16'),
            'surfaces.npy':((128,2,64,64),'int16'),'surface_materials.npy':((128,64,64),'uint8'),
            'origins_xyz.npy':((128,3),'int32')}
    for record in index['worlds']:
        path=root/record['directory'];assert path.resolve().is_relative_to(root.resolve())
        assert sha(path/'manifest.json')==record['manifest_sha256']
        m=json.loads((path/'manifest.json').read_text());seed=m['seed'];split=m['split']
        assert seed not in seen and split in splits and seed==record['seed'] and split==record['split']
        seen.add(seed);splits[split].add(seed)
        assert m['schema']==index['schema']=='minecraft-semantic-voxel22-v1'
        assert m['axis_order']=='NYZX' and m['cubes']==128
        assert m['height_columns_crosschecked_with_archived_ground']>0
        height_checks+=m['height_columns_crosschecked_with_archived_ground']
        for name,(shape,dtype) in shapes.items():
            assert sha(path/name)==m['files'][name]['sha256']
            a=np.load(path/name,mmap_mode='r',allow_pickle=False)
            assert a.shape==shape and str(a.dtype)==dtype
            if name=='voxels.npy':
                counts=np.zeros(22,np.int64)
                for j,cube in enumerate(a):
                    assert cube.max()<22
                    c=np.bincount(cube.ravel(),minlength=22);counts+=c
                    assert c.tolist()==m['cube_records'][j]['class_counts']
                hist[split]+=counts
            if name=='biomes.npy':assert a.min()>0 and a.max()<len(m['biome_palette'])
            if name=='origins_xyz.npy':
                assert len(set(map(tuple,a)))==128 and (a[:,[0,2]]%64==0).all()
                assert (a[:,1]>=-64).all() and (a[:,1]<=256).all()
        union[split]+=m['source_union_voxel_positions']
        underground_air+=sum(r['underground_air_voxels'] for r in m['cube_records'])
        for k,v in m['unknown_semantic_palette_counts'].items():unknown[k]=unknown.get(k,0)+v
        worlds.append((path,m))
    assert len(seen)==64 and len(splits['train'])==56 and len(splits['validation'])==8
    assert not splits['train']&splits['validation']
    assert underground_air>0
    return worlds,{'index_sha256':sha(index_path),'worlds':64,'train_worlds':56,'validation_worlds':8,
                   'zero_seed_overlap':True,'all_file_hashes_and_cubes_verified':True,
                   'class_counts':{k:v.tolist() for k,v in hist.items()},'source_union_voxel_positions':union,
                   'height_columns_crosschecked':height_checks,'underground_air_voxels_nominal':underground_air,
                   'unknown_palette_counts':unknown}


class CubeDataset:
    def __init__(self, worlds, split):
        self.worlds=[(p,m) for p,m in worlds if m['split']==split]
        self.records=[(w,j) for w,(_,m) in enumerate(self.worlds) for j in range(m['cubes'])]
        self.cache=OrderedDict()

    def __len__(self):return len(self.records)

    def get(self,index):
        w,j=self.records[index]
        if w not in self.cache:
            path,_=self.worlds[w]
            self.cache[w]={n:np.load(path/(n+'.npy'),mmap_mode='r',allow_pickle=False)
                           for n in ('voxels','surfaces','origins_xyz')}
            if len(self.cache)>16:self.cache.popitem(last=False)
        self.cache.move_to_end(w);a=self.cache[w]
        target=np.array(a['voxels'][j],dtype=np.int64)
        underground=(np.arange(64)[:,None,None]+a['origins_xyz'][j,1] <= a['surfaces'][j,1,None]-4)
        return target,underground

    def batch(self,indices,rng=None):
        targets=[];underground=[]
        for i in indices:
            t,u=self.get(int(i))
            if rng is not None:
                k=int(rng.integers(4));t=np.rot90(t,k,axes=(1,2));u=np.rot90(u,k,axes=(1,2))
                if rng.integers(2):t=t[:,:,::-1];u=u[:,:,::-1]
            targets.append(t);underground.append(u)
        return np.stack(targets),np.stack(underground)

    def validation_indices(self):
        selected=[]
        for w,(_,m) in enumerate(self.worlds):
            for kind,count in [('subsurface',8),('surface',6),('other',2)]:
                candidates=[r['index'] for r in m['cube_records'] if r['sampling_stratum']==kind]
                candidates.sort(key=lambda j:hashlib.sha256((m['seed']+':fixed-vae-validation:'+str(j)).encode()).digest())
                selected.extend(w*128+j for j in candidates[:count])
        assert len(selected)==128 and len(set(selected))==128
        return selected
