"""Decode local depth fields and compare physical versus encoded overlap blending."""
import numpy as np


def decode_depth(field, scale=128., offset=64.):
    field = np.asarray(field)
    if field.shape[-3] != 9 or not np.isfinite(field).all():
        raise ValueError('expected finite nine-channel depth fields')
    result = field.copy()
    ground = field[..., 1, :, :]*scale+offset
    depth = np.expm1(np.clip((field[..., 0, :, :]+1)*.5, 0, 1)*np.log(384.))
    result[..., 1, :, :] = ground
    result[..., 0, :, :] = ground+depth
    return result


def compare_blends(patches, positions, size):
    patches = np.asarray(patches, dtype=np.float32)
    if patches.shape != (len(positions), 9, 64, 64):
        raise ValueError('expected one 64x64 field for each origin')
    total = np.zeros((9, size, size), np.float32)
    physical_total = np.zeros_like(total)
    weights = np.zeros((size, size), np.float32)
    ramp = np.maximum(np.sin((np.arange(64, dtype=np.float32)+.5)*np.pi/64)**2, 1e-3)
    window = ramp[:, None]*ramp[None, :]
    decoded = decode_depth(patches)
    for patch, decoded_patch, (z, x) in zip(patches, decoded, positions):
        if not (0 <= z <= size-64 and 0 <= x <= size-64):
            raise ValueError('patch exceeds output bounds')
        total[:, z:z+64, x:x+64] += patch*window
        physical_total[:, z:z+64, x:x+64] += decoded_patch*window
        weights[z:z+64, x:x+64] += window
    if np.any(weights <= 0):
        raise ValueError('output contains uncovered cells')
    return physical_total/weights, decode_depth(total/weights)
