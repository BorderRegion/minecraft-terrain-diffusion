"""Unclipped normalized latent velocity objective and DDIM updates."""
import torch
from global_voxel_diffusion_training_v3 import alpha_schedule


def velocity_objective(prediction,x0,noise,alpha,condition):
    assert prediction.shape==x0.shape==noise.shape and condition.shape[1]==17
    a=alpha[:,None,None,None,None]
    target=a.sqrt()*noise-(1-a).sqrt()*x0
    error=(prediction.float()-target).square().mean(1)
    weight=torch.where(condition[:,13]>.0625,2.,torch.where(condition[:,13]<-.25,.2,1.))
    loss=(error*weight).sum()/weight.sum()
    return loss,dict(weighted_velocity_mse=loss,raw_velocity_mse=error.mean())


def ddim_step(noisy,velocity,alpha,previous_alpha):
    x0=alpha.sqrt()*noisy-(1-alpha).sqrt()*velocity
    epsilon=(1-alpha).sqrt()*noisy+alpha.sqrt()*velocity
    return previous_alpha.sqrt()*x0+(1-previous_alpha).sqrt()*epsilon


@torch.no_grad()
def sample_ddim(model,condition,profile,noise,steps=100,guidance=1.):
    assert 2<=steps<=1000 and guidance>0
    alpha=alpha_schedule(noise.device)
    times=torch.linspace(999,0,steps).round().to(torch.int64).tolist()
    assert len(set(times))==steps
    x=noise
    for index,t in enumerate(times):
        timestep=torch.full((len(x),),t,device=x.device,dtype=torch.int64)
        velocity=model(x,timestep,condition,profile)
        if guidance!=1.:
            null=model(x,timestep,torch.zeros_like(condition),torch.full_like(profile,2))
            velocity=null+guidance*(velocity-null)
        previous=alpha[times[index+1]] if index+1<len(times) else alpha.new_ones(())
        x=ddim_step(x,velocity,alpha[t],previous)
    return x
