"""Noise/velocity objective and explicit density projection for the3D generator."""
import torch
import torch.nn.functional as F


def alpha_schedule(device):
    return torch.cumprod(1-torch.linspace(.0001,.02,1000,dtype=torch.float32),0).to(device)


def density_probabilities(signed):
    amplitude=(signed.clamp(-1,1)+1)*.5;raw=amplitude.square();mass=raw.sum(1,keepdim=True)
    probabilities=raw/mass.clamp_min(1e-8)
    fallback=F.one_hot(signed.argmax(1),signed.shape[1]).movedim(-1,1).to(signed.dtype)
    return torch.where(mass>1e-8,probabilities,fallback)


def weights(condition):
    underground=condition[:,0]>.0625
    sky=condition[:,0]<-.25
    region=torch.where(underground,2.,torch.where(sky,.2,1.))*condition[:,10]
    return region,underground


def velocity_objective(prediction,x0,noisy,noise,alpha,condition):
    a=alpha[:,None,None,None,None]
    target=a.sqrt()*noise-(1-a).sqrt()*x0
    mse=(prediction.float()-target).square()
    region,underground=weights(condition)
    channel=torch.tensor([2.,1.,1.,.75,.75,1.,1.,1.,1.,1.,1.,1.,.5],device=x0.device)
    channel=channel/channel.mean()
    weighted=(mse*channel[None,:,None,None,None]*region[:,None]).sum()/(region.sum()*13)
    estimate=a.sqrt()*noisy-(1-a).sqrt()*prediction.float()
    probability=density_probabilities(estimate)
    target_probability=((x0+1)*.5).square()
    air_error=(probability[:,0]-target_probability[:,0]).abs()
    # This low-noise auxiliary term emphasizes cavity volume without replacing velocity learning.
    air_aux=(air_error*region*alpha[:,None,None,None]).sum()/region.sum()
    raw_mass=(((estimate.clamp(-1,1)+1)*.5).square().sum(1)-1).square()
    mass_aux=(raw_mass*region*alpha[:,None,None,None]).sum()/region.sum()
    valid_underground=underground&condition[:,10].bool()
    underground_mse=(mse.mean(1)*valid_underground).sum()/valid_underground.sum().clamp_min(1)
    return weighted+.1*air_aux+.01*mass_aux,{'weighted_velocity_mse':weighted,'raw_velocity_mse':mse.mean(),
            'underground_velocity_mse':underground_mse,'air_volume_auxiliary':air_aux,'density_mass_auxiliary':mass_aux,
            'air_l1_underground':(air_error*valid_underground).sum()/valid_underground.sum().clamp_min(1)}


@torch.no_grad()
def sample_ddim(model,condition,profile,noise,steps=100,guidance=1.):
    assert 2<=steps<=1000 and guidance>0
    alpha=alpha_schedule(noise.device);times=torch.linspace(999,0,steps).round().to(torch.int64).tolist()
    assert len(set(times))==steps
    x=noise
    for index,t in enumerate(times):
        timestep=torch.full((len(x),),t,device=x.device,dtype=torch.int64)
        velocity=model(x,timestep,condition,profile)
        if guidance!=1.:
            null=model(x,timestep,torch.zeros_like(condition),torch.full_like(profile,2))
            velocity=null+guidance*(velocity-null)
        at=alpha[t];previous=alpha[times[index+1]] if index+1<len(times) else torch.ones((),device=x.device)
        x0=at.sqrt()*x-(1-at).sqrt()*velocity
        epsilon=(1-at).sqrt()*x+at.sqrt()*velocity
        x=previous.sqrt()*x0+(1-previous).sqrt()*epsilon
    return x,density_probabilities(x)
