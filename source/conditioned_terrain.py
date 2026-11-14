"""Local depth diffusion conditioned on coarse physical terrain and materials."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import UNet

CONDITIONING = 'surface_ground_material_bilinear_v1'


class ConditionedUNet(UNet):
    def __init__(self, base=320, attention=True):
        super().__init__(base=base, attention=attention)
        self.input = nn.Conv2d(18, base, 3, padding=1)

    def forward(self, noisy, t, condition):
        if noisy.shape != condition.shape or noisy.shape[1] != 9:
            raise ValueError('expected matching 9-channel noisy field and condition')
        return super().forward(torch.cat((noisy, condition), dim=1), t)


def initialize_from_unconditional(net, weights):
    copied = dict(weights)
    old = copied['input.weight']
    if old.shape[1] != 9:
        raise ValueError('expected an unconditional nine-channel checkpoint')
    copied['input.weight'] = torch.cat((old, torch.zeros_like(old)), dim=1)
    net.load_state_dict(copied)


def initialize_from_checkpoint(net, state):
    """Preserve learned conditioning when starting a new dataset/training stage."""
    mode = state.get('conditioning')
    if mode is None:
        initialize_from_unconditional(net, state['ema'])
    elif mode == CONDITIONING and state.get('input_channels') == 18:
        net.load_state_dict(state['ema'])
    else:
        raise ValueError('incompatible warm-start conditioning')


def make_condition(clean):
    """Pool physical surface/ground before interpolation; preserve coarse materials."""
    depth = torch.expm1(((clean[:, 0:1]+1)*.5).clamp(0, 1)*np.log(384.))
    surface = clean[:, 1:2]+depth/128
    physical = torch.cat((surface, clean[:, 1:]), dim=1)
    return F.interpolate(F.avg_pool2d(physical, 8), size=clean.shape[-2:], mode='bilinear', align_corners=False)


def sample(model, alpha, condition, steps=100, seed=387, guidance_scale=1.):
    """Conditional DDIM; it applies no iterative mean projection or water filter."""
    torch.manual_seed(seed)
    x = torch.randn(condition.shape, device=condition.device)
    schedule = torch.linspace(len(alpha)-1, 0, steps, device=condition.device).long().tolist()
    with torch.no_grad():
        for index, t in enumerate(schedule):
            at = alpha[t]
            nxt = alpha[schedule[index+1]] if index+1 < len(schedule) else torch.ones((), device=x.device)
            times = torch.full((len(x),), t, device=x.device, dtype=torch.long)
            velocity = model(x, times, condition)
            if guidance_scale != 1:
                unconditional = model(x, times, torch.zeros_like(condition))
                velocity = unconditional+guidance_scale*(velocity-unconditional)
            x0 = at.sqrt()*x-(1-at).sqrt()*velocity
            eps = (1-at).sqrt()*x+at.sqrt()*velocity
            x = nxt.sqrt()*x0+(1-nxt).sqrt()*eps
    return x
