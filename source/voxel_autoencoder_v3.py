"""Factor-four semantic 3D VAE; topology is measured separately from materials."""
import torch
from torch import nn
import torch.nn.functional as F

CLASSES = 22


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv3d(width, width, 3, padding=1),
                                 nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv3d(width, width, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class VoxelVAE(nn.Module):
    def __init__(self, base=32, latent_channels=16, embedding_channels=16):
        super().__init__()
        self.config = dict(base=base, latent_channels=latent_channels,
                           embedding_channels=embedding_channels)
        self.embedding = nn.Embedding(CLASSES, embedding_channels)
        self.encoder = nn.Sequential(
            nn.Conv3d(embedding_channels, base, 3, padding=1), Residual(base),
            nn.Conv3d(base, base*2, 4, stride=2, padding=1), Residual(base*2),
            nn.Conv3d(base*2, base*4, 4, stride=2, padding=1), Residual(base*4),
            nn.GroupNorm(8, base*4), nn.SiLU(),
            nn.Conv3d(base*4, latent_channels*2, 1))
        self.from_latent = nn.Conv3d(latent_channels, base*4, 3, padding=1)
        self.low = Residual(base*4)
        self.mid = nn.Sequential(nn.Conv3d(base*4, base*2, 3, padding=1), Residual(base*2))
        self.high = nn.Sequential(nn.Conv3d(base*2, base, 3, padding=1), Residual(base),
                                  nn.GroupNorm(8, base), nn.SiLU(), nn.Conv3d(base, CLASSES, 1))

    def encode(self, target):
        embedded = self.embedding(target).permute(0, 4, 1, 2, 3).contiguous()
        mean, logvar = self.encoder(embedded).chunk(2, dim=1)
        return mean, logvar.clamp(-12, 8)

    def decode(self, latent):
        x = self.low(self.from_latent(latent))
        x = self.mid(F.interpolate(x, scale_factor=2, mode='nearest'))
        return self.high(F.interpolate(x, scale_factor=2, mode='nearest'))

    def forward(self, target, sample=True):
        mean, logvar = self.encode(target)
        latent = mean + torch.exp(logvar*.5)*torch.randn_like(mean) if sample else mean
        return self.decode(latent), mean, logvar


def reconstruction_loss(logits, target, underground, mean, logvar, class_weights):
    logits = logits.float()
    ce = F.cross_entropy(logits, target)
    balanced = F.cross_entropy(logits, target, weight=class_weights)
    probability = logits.softmax(1)
    air = probability[:, 0]
    air_target = (target == 0).float()
    air_bce = F.binary_cross_entropy(air.clamp(1e-6, 1-1e-6), air_target, reduction='none')
    air_weights = 1 + 2*underground.float()
    air_bce = (air_bce*air_weights).sum()/air_weights.sum()
    fluid = probability[:, 1:3].sum(1)
    fluid_target = ((target == 1) | (target == 2)).float()
    fluid_bce = F.binary_cross_entropy(fluid.clamp(1e-6, 1-1e-6), fluid_target)
    edges = []
    for dim in (1, 2, 3):
        predicted = torch.diff(air, dim=dim)
        expected = torch.diff(air_target, dim=dim)
        weight = 1 + 4*expected.abs()
        edges.append(((predicted-expected).abs()*weight).mean())
    boundary = torch.stack(edges).mean()
    kl = .5*(mean.float().square()+logvar.float().exp()-1-logvar.float()).mean()
    loss = ce + .15*balanced + .75*air_bce + .25*fluid_bce + .2*boundary + 1e-6*kl
    return loss, {'ce':ce, 'balanced_ce':balanced, 'air_bce':air_bce,
                  'fluid_bce':fluid_bce, 'air_boundary':boundary, 'kl':kl}
