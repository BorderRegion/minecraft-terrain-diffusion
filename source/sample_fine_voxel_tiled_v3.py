"""DDIM on one shared regional latent field, averaging overlapping tile velocities."""
import itertools
import math
import torch
from global_voxel_diffusion_training_v3 import alpha_schedule
from fine_voxel_diffusion_training_v3 import ddim_step


def starts(length,tile=16,stride=8):
    assert length>=tile and 1<=stride<=tile
    values=list(range(0,length-tile+1,stride))
    if values[-1]!=length-tile:values.append(length-tile)
    return values


def blend_window(device,tile=16):
    axis=torch.sin(math.pi*(torch.arange(tile,device=device,dtype=torch.float32)+.5)/tile).square()
    return axis[:,None,None]*axis[None,:,None]*axis[None,None,:]


@torch.no_grad()
def sample_tiled_ddim(model,condition,profile,noise,steps=100,guidance=1.,stride=8,batch=8,progress=None):
    """A regional update reads the same noisy field for every overlapping tile.

    Blending velocities avoids treating neighboring patches as independent
    generations. It does not prove codec tile equivalence or native continuity.
    """
    assert noise.ndim==condition.ndim==5 and noise.shape[:2]==(1,32)
    assert condition.shape[:2]==(1,17) and noise.shape[2:]==condition.shape[2:]
    assert 2<=steps<=1000 and batch>0 and guidance>0
    assert torch.isfinite(condition).all() and torch.isfinite(noise).all()
    axes=[starts(length,stride=stride) for length in noise.shape[2:]]
    coordinates=list(itertools.product(*axes));window=blend_window(noise.device)[None,None]
    mass=torch.zeros_like(noise[:,:1],dtype=torch.float32)
    for y,z,x in coordinates:mass[:,:,y:y+16,z:z+16,x:x+16]+=window
    assert torch.all(mass>0), 'Uncovered regional latent positions'
    alpha=alpha_schedule(noise.device);times=torch.linspace(999,0,steps).round().to(torch.int64).tolist()
    assert len(set(times))==steps
    value=noise.clone()
    for iteration,t in enumerate(times):
        accumulated=torch.zeros_like(value,dtype=torch.float32)
        for start in range(0,len(coordinates),batch):
            subset=coordinates[start:start+batch]
            patches=torch.cat([value[:,:,y:y+16,z:z+16,x:x+16] for y,z,x in subset])
            cond=torch.cat([condition[:,:,y:y+16,z:z+16,x:x+16] for y,z,x in subset])
            timestep=torch.full((len(subset),),t,device=value.device,dtype=torch.int64)
            profiles=torch.full_like(timestep,int(profile))
            velocity=model(patches,timestep,cond,profiles)
            if guidance!=1.:
                null=model(patches,timestep,torch.zeros_like(cond),torch.full_like(profiles,2))
                velocity=null+guidance*(velocity-null)
            assert velocity.shape==patches.shape and torch.isfinite(velocity).all()
            for k,(y,z,x) in enumerate(subset):
                accumulated[:,:,y:y+16,z:z+16,x:x+16]+=velocity[k:k+1].float()*window
        previous=alpha[times[iteration+1]] if iteration+1<len(times) else alpha.new_ones(())
        value=ddim_step(value,accumulated/mass,alpha[t],previous)
        assert torch.isfinite(value).all()
        if progress:progress(dict(iteration=iteration+1,steps=steps,timestep=t,tiles=len(coordinates)))
    return value
