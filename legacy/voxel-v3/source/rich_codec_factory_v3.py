"""Instantiate the exact checkpoint codec variant without changing legacy model files."""
from rich_voxel_autoencoder_v3 import RichVoxelVAE
from rich_voxel_detail_decoder_v3 import RichDetailVoxelVAE


def create_codec(config):
    return RichDetailVoxelVAE(**config) if 'detail_config' in config else RichVoxelVAE(**config)
