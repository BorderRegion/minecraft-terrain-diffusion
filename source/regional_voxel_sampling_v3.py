"""Explicit surface/density bridges for the complete512x512x384 generation path."""
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from fine_voxel_diffusion_training_v3 import ddim_step
from global_voxel_diffusion_training_v3 import alpha_schedule
from fine_voxel_latent_data_v3 import condition_from_probabilities


def stage_seed(seed,stage):
    return int.from_bytes(hashlib.sha256(('voxel-v3:'+str(seed)+':'+stage+':v1').encode()).digest()[:8],'big')%(2**63)


@torch.no_grad()
def sample_surface(model,noise,profile,steps=100,guidance=1.,progress=None):
    assert noise.shape==(1,9,64,64) and profile in [0,1]
    assert 2<=steps<=1000 and guidance>0 and torch.isfinite(noise).all()
    alpha=alpha_schedule(noise.device);times=torch.linspace(999,0,steps).round().to(torch.int64).tolist()
    assert len(set(times))==steps
    value=noise.clone()
    for i,t in enumerate(times):
        time=torch.full((1,),t,device=noise.device,dtype=torch.int64)
        profiles=torch.full_like(time,profile);velocity=model(value,time,profiles)
        if guidance!=1.:
            null=model(value,time,torch.full_like(profiles,2));velocity=null+guidance*(velocity-null)
        assert velocity.shape==noise.shape and torch.isfinite(velocity).all()
        previous=alpha[times[i+1]] if i+1<len(times) else alpha.new_ones(())
        value=ddim_step(value,velocity,alpha[t],previous)
        assert torch.isfinite(value).all()
        if progress:progress(dict(iteration=i+1,steps=steps,timestep=t))
    return value


def project_guide(encoded,height_policy='bounded'):
    """Return physical surface/ground and material fractions; report all repairs."""
    value=np.asarray(encoded,np.float32)
    assert value.shape==(9,64,64) and np.isfinite(value).all()
    assert height_policy in ['bounded','strict']
    raw=value[:2]*128+64;physical=raw.copy()
    below=raw[0]<raw[1];outside=(raw<-64)|(raw>319)
    if height_policy=='strict':assert not below.any() and not outside.any(), 'Raw guide violates the requested strict height policy'
    else:
        np.clip(physical,-64,319,out=physical);physical[0]=np.maximum(physical[0],physical[1])
    raw_material=(value[2:]+1)*.5;clamped=np.clip(raw_material,0,1)
    mass=clamped.sum(0);zero=mass<=1e-8
    fractions=clamped/np.maximum(mass,1e-8)[None]
    fractions[:,zero]=1/7
    report=dict(height_policy=height_policy,raw_surface_below_ground_cells=int(below.sum()),
                raw_height_outside_world_values=int(outside.sum()),changed_height_values=int((physical!=raw).sum()),
                maximum_height_adjustment_blocks=float(np.abs(physical-raw).max()),
                raw_height_minimum=float(raw.min()),raw_height_maximum=float(raw.max()),
                material_values_outside_unit_interval=int(((raw_material<0)|(raw_material>1)).sum()),
                zero_material_mass_cells=int(zero.sum()),mean_raw_material_mass_error=float(np.abs(raw_material.sum(0)-1).mean()),
                mean_projected_material_absolute_change=float(np.abs(fractions-raw_material).mean()),
                scope='Explicit conditioning projection; raw guide is retained and projection is not a quality acceptance.')
    assert np.all(physical[0]>=physical[1]) and np.allclose(fractions.sum(0),1,atol=1e-6)
    return np.concatenate([physical,fractions]),report


def coarse_condition(guide):
    g=np.asarray(guide,np.float32)
    assert g.shape==(9,64,64) and np.isfinite(g).all()
    assert np.all(g[0]>=g[1]) and np.allclose(g[2:].sum(0),1,atol=1e-6)
    y=(-60+8*np.arange(48,dtype=np.float32))[:,None,None]
    result=np.empty((12,48,64,64),np.float32)
    result[0]=np.clip((g[1,None]-y)/128,-3,3);result[1]=np.clip((g[0,None]-y)/128,-3,3)
    result[2:9]=2*g[2:,None]-1;result[9]=(y+64)/192-1;result[10:]=1
    return result


def fine_condition(probabilities,guide):
    p=np.asarray(probabilities,np.float32);g=np.asarray(guide,np.float32)
    assert p.shape==(13,48,64,64) and g.shape==(9,64,64)
    # The guide represents8-block source averages. Fine training uses4-block
    # surface averages; bilinear interpolation supplies approximate inference
    # conditions without inventing extra independently generated terrain tiles.
    surfaces=F.interpolate(torch.from_numpy(g[None,:2].copy()),size=(512,512),mode='bilinear',align_corners=False)[0].numpy()
    return condition_from_probabilities(p,surfaces,-64),surfaces


def density_projection_report(raw,probabilities):
    raw=np.asarray(raw,np.float32);p=np.asarray(probabilities,np.float32)
    assert raw.shape==p.shape==(13,48,64,64) and np.isfinite(raw).all() and np.isfinite(p).all()
    amplitude=(raw+1)*.5;clamped=np.clip(amplitude,0,1)**2;mass=clamped.sum(0)
    assert p.min()>=0 and np.allclose(p.sum(0),1,atol=2e-5)
    return dict(raw_signed_values_outside_range=int(((raw<-1)|(raw>1)).sum()),
                raw_signed_minimum=float(raw.min()),raw_signed_maximum=float(raw.max()),
                mean_unclamped_squared_amplitude_mass_error=float(np.abs((amplitude**2).sum(0)-1).mean()),
                mean_clamped_mass_error=float(np.abs(mass-1).mean()),zero_clamped_mass_cells=int((mass<=1e-8).sum()),
                mean_projection_absolute_change=float(np.abs(p-clamped).mean()),
                scope='Coarse density projection diagnostics; do not infer sampled cave topology from probability normalization.')
