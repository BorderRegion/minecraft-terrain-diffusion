"""Private learned appearance decoding from32-channel latents;semantic path stays separate."""
import torch
from torch import nn
import torch.nn.functional as F
from rich_voxel_autoencoder_v3 import RichVoxelVAE


class RichDetailVoxelVAE(RichVoxelVAE):
    def __init__(self,geometry_config,state_count,name_count,biome_count,property_sizes,detail_config=None):
        super().__init__(geometry_config,state_count,name_count,biome_count,property_sizes)
        detail_config=detail_config or {'hidden':geometry_config['base']*2}
        base=geometry_config['base'];latent=geometry_config['latent_channels'];hidden=detail_config['hidden']
        assert hidden%8==0
        self.name_detail=nn.Sequential(nn.Conv3d(latent,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU(),
            nn.Conv3d(hidden,hidden,3,padding=1),nn.GroupNorm(8,hidden),nn.SiLU(),
            nn.ConvTranspose3d(hidden,base,4,stride=2,padding=1),nn.GroupNorm(8,base),nn.SiLU(),
            nn.Conv3d(base,base,3,padding=1),nn.GroupNorm(8,base),nn.SiLU(),
            nn.ConvTranspose3d(base,base,4,stride=2,padding=1))
        self.biome_detail=nn.Sequential(nn.Conv3d(latent,base,3,padding=1),nn.GroupNorm(8,base),nn.SiLU(),nn.Conv3d(base,base,3,padding=1))
        for module in [self.name_detail[-1],self.biome_detail[-1]]:
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        self.config={**self.config,'detail_config':dict(detail_config)}

    def load_base(self,state):
        result=self.load_state_dict(state,strict=False)
        assert not result.unexpected_keys and all(n.startswith(('name_detail.','biome_detail.')) for n in result.missing_keys)

    def decode(self,latent):
        x=self.low(self.from_latent(latent));x=self.mid(F.interpolate(x,scale_factor=2,mode='nearest'))
        features=self.high[:-1](F.interpolate(x,scale_factor=2,mode='nearest'))
        semantic=self.high[-1](features)
        appearance=features+self.name_detail(latent)
        biome_features=F.avg_pool3d(features,4)+self.biome_detail(latent)
        return dict(semantic=semantic,names=self.name_head(appearance),properties=self.property_head(appearance),biomes=self.biome_head(biome_features))
