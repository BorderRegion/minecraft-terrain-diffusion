"""Check every schematic terrain column against its generated height/material field."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import nbtlib


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


p = argparse.ArgumentParser()
p.add_argument('field')
p.add_argument('schematic')
p.add_argument('--min-y', type=int, required=True, help='generated height represented by schematic layer zero')
p.add_argument('--wet-policy', choices=['rounded', 'float'], default='rounded')
p.add_argument('--out', required=True)
a = p.parse_args()
field = np.load(a.field, allow_pickle=False)
if field.ndim != 3 or field.shape[0] != 9 or not np.isfinite(field).all():
    raise ValueError('expected a finite 9-channel field')
schematic = nbtlib.load(a.schematic, gzipped=True)
width, height, length = (int(schematic[k]) for k in ('Width', 'Height', 'Length'))
if field.shape[1:] != (length, width):
    raise ValueError('schematic dimensions do not match field')
palette = {str(k): int(v) for k, v in schematic['Palette'].items()}
if max(palette.values()) >= 128:
    raise ValueError('verification requires single-byte palette values')
blocks = np.asarray(schematic['BlockData'], dtype=np.uint8).reshape(height, length, width)
floor = np.clip(np.rint(field[1]), -64, 319).astype(np.int32)
surface = np.clip(np.rint(field[0]), -64, 319).astype(np.int32)
wet_float = field[0] > field[1] + 1
wet_rounded = surface > floor + 1
wet = wet_float if a.wet_policy == 'float' else wet_rounded
if np.any(wet & (surface <= floor)):
    raise ValueError('a water column has no room above its rounded ground')
kind = np.argmax(field[2:], axis=0) + 2
floor_layer = floor - a.min_y
surface_layer = surface - a.min_y
if floor_layer.min() < 0 or floor_layer.max() >= height or (wet.any() and surface_layer[wet].max() >= height):
    raise ValueError('height lies outside schematic')
z, x = np.indices((length, width))
name_by_class = {2: 'minecraft:grass_block[snowy=false]', 3: 'minecraft:dirt', 4: 'minecraft:sand',
                 5: 'minecraft:stone', 6: 'minecraft:snow_block', 7: 'minecraft:gravel', 8: 'minecraft:terracotta'}
lookup = np.zeros(9, dtype=np.uint8)
for k, name in name_by_class.items():
    lookup[k] = palette[name]
expected_top = lookup[kind]
actual_top = blocks[floor_layer, z, x]
top_mismatch = int(np.count_nonzero(actual_top != expected_top))
water_id = palette['minecraft:water[level=0]']
water_mismatch = int(np.count_nonzero(blocks[surface_layer[wet], z[wet], x[wet]] != water_id)) if wet.any() else 0
floor_above = np.minimum(floor_layer + 1, height - 1)
wet_bottom_mismatch = int(np.count_nonzero(blocks[floor_above[wet], z[wet], x[wet]] != water_id)) if wet.any() else 0
stone_id = palette['minecraft:stone']
below = floor_layer >= 4
deep_mismatch = int(np.count_nonzero(blocks[(floor_layer - 4)[below], z[below], x[below]] != stone_id))
result = {'columns': length * width, 'water_columns': int(wet.sum()),
          'float_field_water_columns': int(wet_float.sum()),
          'float_water_columns_excluded_by_rounded_policy': int((wet_float & ~wet_rounded).sum()),
          'exported_water_columns_lost_vs_float': int((wet_float & ~wet).sum()),
          'exported_water_columns_added_vs_float': int((~wet_float & wet).sum()),
          'wet_policy': a.wet_policy,
          'top_material_mismatches': top_mismatch,
          'water_surface_mismatches': water_mismatch, 'water_bottom_mismatches': wet_bottom_mismatch,
          'stone_layer_mismatches': deep_mismatch, 'field_sha256': sha(a.field), 'schematic_sha256': sha(a.schematic),
          'note': 'Checks every column at specified layers, not all interior blocks or in-game appearance.'}
Path(a.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
if any((top_mismatch, water_mismatch, wet_bottom_mismatch, deep_mismatch)):
    raise SystemExit('schematic differs from generated terrain')
