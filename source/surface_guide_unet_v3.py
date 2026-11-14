"""Profile-conditioned global surface guide,starting exactly from the retained2D model."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from train import UNet


class SurfaceGuideUNet(UNet):
    def __init__(self,base=320,profiles=3):
        super().__init__(base=base,attention=True)
        self.profile=nn.Embedding(profiles,256);nn.init.zeros_(self.profile.weight)
        self.config=dict(base=base,profiles=profiles)

    def load_surface_base(self,state):
        assert state['model_base']==self.base and state['channels']==9 and state['attention']
        assert state['height_representation']=='surface_ground' and state['prediction_type']=='v_prediction'
        assert state['normalization']=={'offset':64,'scale':128}
        result=self.load_state_dict(state['ema'],strict=False)
        assert result.missing_keys==['profile.weight'] and not result.unexpected_keys

    def forward(self,x,t,profile):
        half=self.base//2
        frequency=torch.exp(-math.log(10000)*torch.arange(half,device=x.device)/(half-1))
        value=t.float()[:,None]*frequency[None,:]
        embedding=self.time(torch.cat([value.sin(),value.cos()],dim=1))+self.profile(profile)
        a=self.r1(self.input(x),embedding);b=self.r2(self.d1(a),embedding)
        c=self.attn(self.mid(self.r3(self.d2(b),embedding),embedding))
        c=F.interpolate(c,scale_factor=2,mode='nearest');c=self.u2(torch.cat((c,b),1),embedding)
        c=F.interpolate(c,scale_factor=2,mode='nearest')
        return self.out(self.u1(torch.cat((c,a),1),embedding))
