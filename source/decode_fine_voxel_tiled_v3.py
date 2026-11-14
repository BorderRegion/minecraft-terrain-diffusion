"""Bounded-memory overlap-logit decoding of one shared regional rich latent field."""
from collections import OrderedDict
import itertools
import numpy as np
import torch
from rich_voxel_decode_v3 import decode_catalogue
from sample_fine_voxel_tiled_v3 import starts


def window(size):
    axis=np.sin(np.pi*(np.arange(size,dtype=np.float32)+.5)/size)**2
    return axis[:,None,None]*axis[None,:,None]*axis[None,None,:]


@torch.no_grad()
def iter_decoded_blocks(model,latent,vocabulary,output_block=(32,128,128),device='musa',cache_tiles=4):
    """Yield physical block origins and states/biomes from all intersecting tiles.

    Regional latent values must already be de-normalized using the selected
    codec cache's training statistics. Decode overlap remains an approximation;
    continuity, fluids and appearance require whole-world Minecraft review.
    """
    latent=np.asarray(latent,np.float32)
    assert latent.ndim==4 and latent.shape[0]==32 and min(latent.shape[1:])>=16
    assert np.isfinite(latent).all() and cache_tiles>=0
    assert all(size>0 and size%4==0 for size in output_block)
    shape=tuple(size*4 for size in latent.shape[1:])
    tiles=list(itertools.product(*(starts(size,stride=8) for size in latent.shape[1:])))
    weights=window(64);biome_weights=window(16);cache=OrderedDict()
    block_origins=itertools.product(*(range(0,size,step) for size,step in zip(shape,output_block)))
    for origin in block_origins:
        end=tuple(min(size,lo+step) for size,lo,step in zip(shape,origin,output_block))
        extent=tuple(hi-lo for lo,hi in zip(origin,end));bio_extent=tuple(size//4 for size in extent)
        sums={};mass=np.zeros(extent,np.float32);bio_mass=np.zeros(bio_extent,np.float32);used=0
        for tile in tiles:
            physical=tuple(i*4 for i in tile)
            lo=tuple(max(a,b) for a,b in zip(origin,physical));hi=tuple(min(a,b+64) for a,b in zip(end,physical))
            if any(a>=b for a,b in zip(lo,hi)):continue
            if tile not in cache:
                y,z,x=tile
                value=torch.from_numpy(latent[None,:,y:y+16,z:z+16,x:x+16].copy()).to(device)
                prediction=model.decode(value)
                cached={key:prediction[key][0].detach().float().cpu().numpy() for key in ['semantic','names','properties','biomes']}
                assert all(np.isfinite(a).all() for a in cached.values())
                assert all(cached[k].shape[1:]==(64,64,64) for k in ['semantic','names','properties'])
                assert cached['biomes'].shape[1:]==(16,16,16)
                cache[tile]=cached
            cache.move_to_end(tile);prediction=cache[tile]
            local=tuple(slice(a-b,c-b) for a,b,c in zip(lo,origin,hi))
            source=tuple(slice(a-b,c-b) for a,b,c in zip(lo,physical,hi))
            local_bio=tuple(slice((a-b)//4,(c-b)//4) for a,b,c in zip(lo,origin,hi))
            source_bio=tuple(slice((a-b)//4,(c-b)//4) for a,b,c in zip(lo,physical,hi))
            weight=weights[source];bio_weight=biome_weights[source_bio]
            mass[local]+=weight;bio_mass[local_bio]+=bio_weight
            for key,pred in prediction.items():
                if key not in sums:sums[key]=np.zeros((pred.shape[0],)+(bio_extent if key=='biomes' else extent),np.float32)
                target=(slice(None),)+(local_bio if key=='biomes' else local)
                select=(slice(None),)+(source_bio if key=='biomes' else source)
                sums[key][target]+=pred[select]*(bio_weight if key=='biomes' else weight)[None]
            used+=1
            while len(cache)>cache_tiles:cache.popitem(last=False)
        assert used>0 and np.all(mass>0) and np.all(bio_mass>0), 'Uncovered output positions'
        for key,value in sums.items():value/=((bio_mass if key=='biomes' else mass)[None])
        decoded=decode_catalogue({key:value[None] for key,value in sums.items()},vocabulary)
        decoded['projection']['intersecting_codec_tiles']=used
        yield origin,decoded
