"""Conditional3D diffusion backbone for full-height terrain/cave context."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def time_embedding(t,width):
    frequency=torch.exp(-math.log(10000)*torch.arange(width//2,device=t.device,dtype=torch.float32)/(width//2-1))
    phase=t.float()[:,None]*frequency[None]
    return torch.cat([phase.cos(),phase.sin()],dim=1)


class TimeResidual3D(nn.Module):
    def __init__(self,input_width,output_width,embedding_width):
        super().__init__()
        self.first=nn.Sequential(nn.GroupNorm(8,input_width),nn.SiLU(),nn.Conv3d(input_width,output_width,3,padding=1))
        self.time=nn.Sequential(nn.SiLU(),nn.Linear(embedding_width,output_width*2))
        self.norm=nn.GroupNorm(8,output_width)
        self.last=nn.Sequential(nn.SiLU(),nn.Conv3d(output_width,output_width,3,padding=1))
        self.skip=nn.Conv3d(input_width,output_width,1) if input_width!=output_width else nn.Identity()

    def forward(self,x,embedding):
        h=self.first(x);scale,shift=self.time(embedding)[:,:,None,None,None].chunk(2,1)
        return self.skip(x)+self.last(self.norm(h)*(1+scale)+shift)


class GlobalAttention3D(nn.Module):
    def __init__(self,width):
        super().__init__();self.heads=width//64;assert width%self.heads==0
        self.norm=nn.GroupNorm(8,width);self.qkv=nn.Conv1d(width,width*3,1);self.out=nn.Conv1d(width,width,1)

    def forward(self,x):
        b,c,y,z,w=x.shape;n=y*z*w;h=self.heads;d=c//h
        q,k,v=self.qkv(self.norm(x).flatten(2)).reshape(b,3,h,d,n).unbind(1)
        q=q.reshape(b*h,d,n).transpose(1,2);k=k.reshape(b*h,d,n);v=v.reshape(b*h,d,n)
        weights=(torch.bmm(q,k)/math.sqrt(d)).float().softmax(-1).to(v.dtype)
        result=torch.bmm(v,weights.transpose(1,2)).reshape(b,c,n)
        return x+self.out(result).reshape_as(x)


class ConditionalUNet3D(nn.Module):
    def __init__(self,channels=13,condition_channels=12,base=96,blocks=2,profiles=3):
        super().__init__();assert base%8==0 and blocks>=1
        self.config=dict(channels=channels,condition_channels=condition_channels,base=base,blocks=blocks,profiles=profiles)
        self.base=base;embedding=base*4;widths=[base,base*2,base*3,base*4]
        self.time=nn.Sequential(nn.Linear(base,embedding),nn.SiLU(),nn.Linear(embedding,embedding))
        self.profile=nn.Embedding(profiles,embedding)
        self.input=nn.Conv3d(channels+condition_channels,base,3,padding=1)
        self.down_blocks=nn.ModuleList();self.downsample=nn.ModuleList()
        for level,width in enumerate(widths):
            self.down_blocks.append(nn.ModuleList([TimeResidual3D(width,width,embedding) for _ in range(blocks)]))
            if level<3:self.downsample.append(nn.Conv3d(width,widths[level+1],3,stride=2,padding=1))
        self.middle=nn.ModuleList([TimeResidual3D(widths[-1],widths[-1],embedding),GlobalAttention3D(widths[-1]),TimeResidual3D(widths[-1],widths[-1],embedding)])
        self.up_blocks=nn.ModuleList();self.upsample=nn.ModuleList()
        for level in reversed(range(4)):
            width=widths[level]
            self.up_blocks.append(nn.ModuleList([TimeResidual3D(width*2,width,embedding)]+[TimeResidual3D(width,width,embedding) for _ in range(blocks-1)]))
            if level>0:self.upsample.append(nn.Conv3d(width,widths[level-1],3,padding=1))
        self.output=nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv3d(base,channels,3,padding=1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def forward(self,x,t,condition,profile):
        assert x.shape[0]==condition.shape[0] and x.shape[2:]==condition.shape[2:]
        assert all(size%8==0 for size in x.shape[2:])
        embedding=self.time(time_embedding(t,self.base))+self.profile(profile)
        h=self.input(torch.cat([x,condition],1));skips=[]
        for level,layer in enumerate(self.down_blocks):
            for block in layer:h=block(h,embedding)
            skips.append(h)
            if level<3:h=self.downsample[level](h)
        for block in self.middle:
            h=block(h) if isinstance(block,GlobalAttention3D) else block(h,embedding)
        for level,layer in enumerate(self.up_blocks):
            h=torch.cat([h,skips.pop()],1)
            for block in layer:h=block(h,embedding)
            if level<3:h=self.upsample[level](F.interpolate(h,scale_factor=2,mode='nearest'))
        return self.output(h)


def surface_condition(ground,water,material,valid):
    """Inputs are aligned full-resolution512² surface windows and64² valid cells."""
    assert ground.shape==water.shape==material.shape==(512,512) and valid.shape==(64,64)
    import numpy as np
    g=ground.reshape(64,8,64,8).mean((1,3));s=np.maximum(ground,water).reshape(64,8,64,8).mean((1,3))
    y=(-60+8*np.arange(48))[:,None,None]
    fractions=np.stack([(material==k).reshape(64,8,64,8).mean((1,3)) for k in range(2,9)])
    result=np.empty((12,48,64,64),np.float32)
    result[0]=np.clip((g[None]-y)/128,-3,3);result[1]=np.clip((s[None]-y)/128,-3,3)
    result[2:9]=2*fractions[:,None]-1;result[9]=(y+64)/192-1
    result[10]=valid[None];result[11]=1
    return result
