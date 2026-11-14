"""Source-aligned rich latents and coarse3D/surface conditions for fine generation."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np

GROUPS=np.asarray([0,1,2,3,4,5,5,6,6,6,7,7,8,9,9,9,10,10,10,11,10,12],np.uint8)
CONDITION_CHANNELS=[
    'air','water','lava','stone','deepslate','soil_grass','sand_sandstone_gravel',
    'snow_ice','terracotta','wood_leaves_plants','cave_minerals','bedrock','other_solid',
    'relative_ground','relative_surface','absolute_y','condition_present']


def spatial(array,quarters=0,flip=False):
    value=np.rot90(array,quarters,axes=(-2,-1)) if quarters else array
    return value[...,::-1].copy() if flip else value.copy()


def encoder_inputs(data,index,quarters=0,flip=False):
    """Transform actual labels, including directed states, before rich encoding."""
    w,j=data.records[index];a=data.world(w)
    state_ids=spatial(a['states'][j],quarters,flip)
    state_map=a['maps'][(quarters,bool(flip))]['states']
    return dict(semantic=spatial(a['voxels'][j],quarters,flip).astype(np.int64),
                states=state_map[state_ids],
                biomes=a['biome_map'][spatial(a['biomes'][j],quarters,flip)])


def coarse_counts(semantic):
    """Exact13-group counts of non-overlapping8³ physical cells; shape C,Y,Z,X."""
    a=np.asarray(semantic);assert a.shape==(64,64,64) and a.min()>=0 and a.max()<22
    grouped=GROUPS[a]
    counts=np.stack([(grouped==g).reshape(8,8,8,8,8,8).sum((1,3,5),dtype=np.uint16) for g in range(13)])
    assert np.all(counts.sum(0)==512)
    return counts


def fine_condition(counts,surfaces,origin_y,quarters=0,flip=False):
    """Condition matches a64³ physical crop represented by a16³ latent field.

    At inference, supply the coarse generator's projected13-group probabilities
    via condition_from_probabilities; source-count conditions are training truth.
    """
    assert counts.shape==(13,8,8,8) and np.all(counts.sum(0)==512)
    return condition_from_probabilities(counts.astype(np.float32)/512,surfaces,origin_y,quarters,flip)


def condition_from_probabilities(probabilities,surfaces,origin_y,quarters=0,flip=False):
    p=np.asarray(probabilities,np.float32);s=np.asarray(surfaces)
    assert p.ndim==4 and p.shape[0]==13 and min(p.shape[1:])>0
    ny,nz,nx=[2*size for size in p.shape[1:]]
    assert s.shape==(2,nz*4,nx*4)
    assert np.isfinite(p).all() and p.min()>=0 and np.allclose(p.sum(0),1,atol=2e-5)
    assert np.isfinite(s).all() and np.all(s[0]>=s[1])
    assert -64<=origin_y and origin_y+ny*4<=320 and origin_y%8==0
    p=spatial(p,quarters,flip)
    if quarters%2:nz,nx=nx,nz
    s=spatial(s,quarters,flip).reshape(2,nz,4,nx,4).mean((2,4))
    y=origin_y+2+4*np.arange(ny,dtype=np.float32)
    result=np.empty((17,ny,nz,nx),np.float32)
    result[:13]=2*p.repeat(2,axis=1).repeat(2,axis=2).repeat(2,axis=3)-1
    result[13]=np.clip((s[1,None]-y[:,None,None])/128,-3,3)
    result[14]=np.clip((s[0,None]-y[:,None,None])/128,-3,3)
    result[15]=(y[:,None,None]+64)/192-1
    result[16]=1
    assert np.isfinite(result).all()
    return result


class FineLatentDataset:
    """Read an immutable complete latent cache; only training receives augmentation."""
    def __init__(self,index_path,split,cache_worlds=4):
        from voxel_training_data_v3 import sha
        self.path=Path(index_path);self.root=self.path.parent
        self.index=json.loads(self.path.read_text())
        assert self.index['complete'] and self.index['format']=='rich-fine-latent-cache-v1'
        self.worlds=[r for r in self.index['worlds'] if r['split']==split]
        self.split=split;self.cache=OrderedDict();self.cache_worlds=cache_worlds;self.verified={}
        self.records=[(w,j) for w in range(len(self.worlds)) for j in range(128)]
        self.mean=np.asarray(self.index['train_channel_mean'],np.float32)[:,None,None,None]
        self.std=np.asarray(self.index['train_channel_std'],np.float32)[:,None,None,None]
        assert np.isfinite(self.mean).all() and np.isfinite(self.std).all() and self.std.min()>=1e-4
        self.sha=sha

    def __len__(self):return len(self.records)

    def world(self,index):
        if index not in self.cache:
            r=self.worlds[index];d=self.root/r['directory']
            assert self.sha(d/'manifest.json')==r['manifest_sha256']
            manifest=json.loads((d/'manifest.json').read_text())
            arrays={}
            for name,entry in manifest['files'].items():
                path=d/name;info=path.stat();signature=(info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
                if str(path) not in self.verified:
                    assert self.sha(path)==entry['sha256'],('Corrupt latent input',path)
                    self.verified[str(path)]=signature
                else:assert self.verified[str(path)]==signature,('Immutable cache changed',path)
                arrays[name]=np.load(d/name,mmap_mode='r',allow_pickle=False)
            assert arrays['latents.npy'].shape==(8,128,32,16,16,16)
            assert arrays['latents.npy'].dtype==np.float16
            assert arrays['coarse_counts.npy'].shape==(128,13,8,8,8)
            assert arrays['surfaces.npy'].shape==(128,2,64,64)
            assert arrays['origins_xyz.npy'].shape==(128,3)
            self.cache[index]=arrays
            if len(self.cache)>self.cache_worlds:self.cache.popitem(last=False)
        self.cache.move_to_end(index);return self.cache[index]

    def get(self,index,quarters=0,flip=False):
        assert self.split=='train' or (quarters==0 and not flip)
        w,j=self.records[index];r=self.worlds[w];a=self.world(w)
        variant=quarters*2+int(flip)
        latent=np.array(a['latents.npy'][variant,j],dtype=np.float32)
        condition=fine_condition(a['coarse_counts.npy'][j],a['surfaces.npy'][j],int(a['origins_xyz.npy'][j,1]),quarters,flip)
        return dict(latent=(latent-self.mean)/self.std,condition=condition,
                    source=dict(seed=r['seed'],cube=j,profile_id=r['profile_id'],quarters=quarters,flip_x=bool(flip)))

    def batch(self,indices,rng=None):
        assert rng is None or self.split=='train'
        rows=[self.get(int(i),int(rng.integers(4)) if rng is not None else 0,
                       bool(rng.integers(2)) if rng is not None else False) for i in indices]
        return dict(latent=np.stack([r['latent'] for r in rows]),condition=np.stack([r['condition'] for r in rows]),
                    profile=np.asarray([r['source']['profile_id'] for r in rows],np.int64),source=[r['source'] for r in rows])
