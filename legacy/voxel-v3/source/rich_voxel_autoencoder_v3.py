"""Geometry VAE with block-name, property and3D-biome supervision."""
import torch
from torch import nn
import torch.nn.functional as F
from voxel_autoencoder_v3 import VoxelVAE,reconstruction_loss


class RichVoxelVAE(VoxelVAE):
    def __init__(self,geometry_config,state_count,name_count,biome_count,property_sizes):
        super().__init__(**geometry_config)
        self.geometry_config=dict(geometry_config);base=geometry_config['base'];embedding=geometry_config['embedding_channels']
        self.state_embedding=nn.Embedding(state_count,embedding,padding_idx=0)
        self.biome_embedding=nn.Embedding(biome_count,embedding,padding_idx=0)
        nn.init.zeros_(self.state_embedding.weight);nn.init.zeros_(self.biome_embedding.weight)
        self.name_head=nn.Conv3d(base,name_count,1);self.biome_head=nn.Conv3d(base,biome_count,1)
        self.property_offsets={};offset=0
        for name,size in sorted(property_sizes.items()):
            self.property_offsets[name]=(offset,offset+size);offset+=size
        self.property_head=nn.Conv3d(base,offset,1)
        self.config={'geometry_config':dict(geometry_config),'state_count':state_count,'name_count':name_count,
                     'biome_count':biome_count,'property_sizes':dict(property_sizes)}

    def load_geometry(self,checkpoint):
        assert checkpoint['config']==self.geometry_config
        result=self.load_state_dict(checkpoint['model'],strict=False)
        prefixes=('state_embedding.','biome_embedding.','name_head.','biome_head.','property_head.')
        assert not result.unexpected_keys and all(k.startswith(prefixes) for k in result.missing_keys)

    def encode(self,semantic,states,biomes):
        b=self.biome_embedding(biomes).permute(0,4,1,2,3).contiguous()
        b=F.interpolate(b,size=semantic.shape[-3:],mode='nearest')
        x=(self.embedding(semantic)+self.state_embedding(states)).permute(0,4,1,2,3).contiguous()+b
        mean,logvar=self.encoder(x).chunk(2,1)
        return mean,logvar.clamp(-12,8)

    def decode(self,latent):
        x=self.low(self.from_latent(latent));x=self.mid(F.interpolate(x,scale_factor=2,mode='nearest'))
        features=self.high[:-1](F.interpolate(x,scale_factor=2,mode='nearest'))
        return {'semantic':self.high[-1](features),'names':self.name_head(features),
                'properties':self.property_head(features),'biomes':self.biome_head(F.avg_pool3d(features,4))}

    def forward(self,semantic,states,biomes,sample=True):
        mean,logvar=self.encode(semantic,states,biomes)
        z=mean+torch.exp(logvar*.5)*torch.randn_like(mean) if sample else mean
        prediction=self.decode(z);prediction.update(mean=mean,logvar=logvar)
        return prediction


def masked_ce(logits,targets,weight=None):
    value=F.cross_entropy(logits,targets,weight=weight,ignore_index=0,reduction='sum')
    valid=(targets!=0)
    denominator=valid.sum() if weight is None else (weight[targets]*valid).sum()
    return value/denominator.clamp_min(1)


def rich_reconstruction_loss(model,prediction,targets,semantic_weights,name_weights):
    geometry,parts=reconstruction_loss(prediction['semantic'],targets['semantic'],targets['underground'],
                                      prediction['mean'],prediction['logvar'],semantic_weights)
    names=masked_ce(prediction['names'],targets['names'])
    name_balanced=masked_ce(prediction['names'],targets['names'],weight=name_weights)
    numerator=torch.zeros((),device=names.device);denominator=torch.zeros_like(numerator)
    for key,(lo,hi) in model.property_offsets.items():
        label=targets['properties'][key];weight=2. if key in ('axis','facing','rotation','shape','orientation') else 1.
        numerator=numerator+weight*F.cross_entropy(prediction['properties'][:,lo:hi],label,ignore_index=0,reduction='sum')
        denominator=denominator+weight*(label!=0).sum()
    properties=numerator/denominator.clamp_min(1)
    biomes=masked_ce(prediction['biomes'],targets['biomes'])
    total=geometry+.5*names+.1*name_balanced+.15*properties+.1*biomes
    return total,{'geometry':geometry,'name_ce':names,'balanced_name_ce':name_balanced,'property_ce':properties,
                  'biome_ce':biomes,**{'geometry_'+k:v for k,v in parts.items()}}
