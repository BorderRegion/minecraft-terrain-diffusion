"""Generate a continuous 512x512 diagnostic region using learned local conditioning."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
try:
    import torch_musa
except ImportError:
    torch_musa = None

from conditioned_terrain import CONDITIONING, ConditionedUNet, sample as sample_local
from sample import export_schematic
from sample_global_depth_encoding import load_model, sample as sample_global
from terrain_overlap import compare_blends


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--global-checkpoint', required=True)
    p.add_argument('--local-checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--gpu-memory-fraction', type=float)
    a = p.parse_args()
    if not 2 <= a.steps <= 1000 or a.batch < 1:
        p.error('invalid diffusion steps or batch size')
    if a.gpu_memory_fraction is not None:
        if not 0 < a.gpu_memory_fraction <= 1:
            p.error('GPU memory fraction must be in (0, 1]')
        torch.musa.set_per_process_memory_fraction(a.gpu_memory_fraction, 0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed)
    device = torch.device('musa')
    global_model, global_alpha, gs = load_model(a.global_checkpoint, device)
    if gs.get('height_representation', 'surface_ground') != 'surface_ground':
        raise ValueError('global checkpoint must represent physical surface and ground')
    coarse = sample_global(global_model, global_alpha, 1, a.steps, device, prediction_type=gs['prediction_type'])
    del global_model
    torch.musa.empty_cache()
    state = torch.load(a.local_checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if state.get('conditioning') != CONDITIONING or state.get('height_representation') != 'log_depth_ground_v1' or state.get('prediction_type') != 'v_prediction':
        raise ValueError('expected a learned conditional depth checkpoint')
    if state['normalization'] != gs['normalization'] or state['normalization'] != {'offset': 64, 'scale': 128}:
        raise ValueError('incompatible normalization')
    local_model = ConditionedUNet(base=state['model_base'], attention=True).to(device)
    local_model.load_state_dict(state['ema']); local_model.eval()
    local_alpha = torch.cumprod(1-torch.linspace(state['beta_start'], state['beta_end'], state['timesteps'], device=device), 0)
    raw_coarse = coarse.cpu().numpy()[0].copy()
    raw_coarse[:2] = raw_coarse[:2]*128+64
    np.save(out/'global_coarse_raw.npy', raw_coarse)
    # Match the physical invariants of the training conditions once, at the coarse level.
    below = (coarse[:, 0] < coarse[:, 1]).float().mean().item()
    coarse[:, 0] = torch.maximum(coarse[:, 0], coarse[:, 1])
    probabilities = ((coarse[:, 2:]+1)*.5).clamp(0, 1)
    probabilities = (probabilities+1e-8)/(probabilities.sum(1, keepdim=True)+7e-8)
    coarse[:, 2:] = probabilities*2-1
    guide = F.interpolate(coarse, size=(512, 512), mode='bilinear', align_corners=False)
    positions = [(z, x) for z in range(0, 449, 32) for x in range(0, 449, 32)]
    patches = []
    for start in range(0, len(positions), a.batch):
        current = positions[start:start+a.batch]
        condition = torch.cat([guide[:, :, z:z+64, x:x+64] for z, x in current])
        encoded = sample_local(local_model, local_alpha, condition, a.steps, a.seed+1+start)
        patches.append(encoded.cpu().numpy())
        print(json.dumps({'patches': start+len(current), 'total': len(positions)}), flush=True)
    patches = np.concatenate(patches)
    np.save(out/'encoded_patches.npy', patches)
    field, legacy_blend = compare_blends(patches, positions, 512)
    np.save(out/'global_field.npy', field)
    np.save(out/'encoded_blend_reference.npy', legacy_blend)
    if not np.isfinite(field).all() or np.any(field[0] < field[1]-1e-4):
        raise ValueError('invalid physical terrain field')
    if field[:2].min() < -64 or field[:2].max() > 319:
        raise ValueError('generated field exceeds Minecraft height bounds; raw field retained for diagnosis')
    physical_coarse = coarse.cpu().numpy()[0]
    physical_coarse[:2] = physical_coarse[:2]*128+64
    np.save(out/'global_coarse.npy', physical_coarse)
    whole = export_schematic(field, out/'terrain_full.schem', decorate=True, seed=a.seed)
    manifest = {'seed': a.seed, 'global_step': gs['step'], 'local_probe_step': state['step'],
                'warm_start': state.get('warm_start'), 'conditioning': CONDITIONING,
                'blending': 'physical_surface_ground', 'batch': a.batch, 'steps': a.steps,
                'noise_seed_rule': 'seed + 1 + batch_start_index',
                'gpu_memory_fraction': a.gpu_memory_fraction,
                'gpu_peak_allocated_bytes': torch.musa.max_memory_allocated(),
                'gpu_peak_reserved_bytes': torch.musa.max_memory_reserved(),
                'coarse_surface_below_ground_fraction_before_projection': below,
                'coarse_projection': 'surface=max(surface,ground); clamp and normalize material probabilities',
                'full': whole, 'note': 'Diagnostic generated region; schematic validity, novelty, game behavior and aesthetics still require verification.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Saved conditional global diagnostic '+str(out), flush=True)


if __name__ == '__main__':
    main()
