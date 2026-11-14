"""Generate a 512x512 heightfield with the expanded-v2 global/local EMA pair."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def load(role, directory, device):
    import torch
    from train import UNet
    from conditioned_terrain import CONDITIONING, ConditionedUNet

    filename, step = ('global2047-v2-ema.pt', 98000) if role == 'global' else ('conditional2048-v2-ema.pt', 100000)
    path = Path(directory) / filename
    manifest = json.loads((Path(directory) / 'WEIGHTS.json').read_text())
    row = next(r for r in manifest['weights'] if r['role'] == role)
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Weight integrity mismatch: ' + filename)
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state['step'] != step or state['normalization'] != {'offset': 64, 'scale': 128}:
        raise ValueError('Wrong expanded-v2 checkpoint identity')
    if role == 'local' and (state.get('conditioning') != CONDITIONING or
                            state.get('height_representation') != 'log_depth_ground_v1' or
                            state.get('prediction_type') != 'v_prediction'):
        raise ValueError('Incompatible local checkpoint')
    if role == 'global' and state.get('height_representation', 'surface_ground') != 'surface_ground':
        raise ValueError('Incompatible global checkpoint')
    cls = UNet if role == 'global' else ConditionedUNet
    model = cls(base=state['model_base'], attention=state['attention'])
    model.load_state_dict(state['ema'], strict=True)
    model.to(device).eval().requires_grad_(False)
    if sum(t.numel() for t in model.parameters()) != row['parameters']:
        raise ValueError('Parameter count mismatch')
    alpha = torch.cumprod(1 - torch.linspace(state['beta_start'], state['beta_end'], state['timesteps'], device=device), 0)
    meta = {key: value for key, value in state.items() if key != 'ema'}
    del state
    return model, alpha, meta


def release(device):
    import torch
    gc.collect()
    if device.type in ('cuda', 'musa'):
        getattr(torch, device.type).empty_cache()


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from conditioned_terrain import sample as local_sample
    from sample_global_depth_encoding import sample as global_sample
    from sample import export_schematic
    from terrain_overlap import compare_blends
    from height_domain import project_height_domain

    if args.device == 'musa':
        import torch_musa  # Registers the original backend only when selected.
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def progress(stage, **extra):
        value = dict(stage=stage, elapsed_seconds=time.time()-started, **extra)
        write(out / 'status.json', value)
        print(json.dumps(value), flush=True)

    try:
        torch.manual_seed(args.seed)
        model, alpha, global_meta = load('global', args.weights, device)
        progress('global_sampling', step=global_meta['step'])
        coarse = global_sample(model, alpha, 1, args.steps, device,
                               prediction_type=global_meta['prediction_type'])
        del model, alpha
        release(device)
        coarse[:, 0] = torch.maximum(coarse[:, 0], coarse[:, 1])
        probabilities = ((coarse[:, 2:]+1)*.5).clamp(0, 1)
        probabilities = (probabilities+1e-8)/(probabilities.sum(1, keepdim=True)+7e-8)
        coarse[:, 2:] = probabilities*2-1
        guide = F.interpolate(coarse, size=(512, 512), mode='bilinear', align_corners=False)
        physical_coarse = coarse[0].cpu().numpy().copy()
        physical_coarse[:2] = physical_coarse[:2]*128+64
        np.save(out / 'global_coarse.npy', physical_coarse)
        del coarse, probabilities
        model, alpha, local_meta = load('local', args.weights, device)
        positions = [(z, x) for z in range(0, 449, 32) for x in range(0, 449, 32)]
        if args.smoke:
            positions = positions[:1]
        patches = []
        for start in range(0, len(positions), args.batch):
            current = positions[start:start+args.batch]
            condition = torch.cat([guide[:, :, z:z+64, x:x+64] for z, x in current])
            encoded = local_sample(model, alpha, condition, args.steps, args.seed+1+start)
            patches.append(encoded.cpu().numpy())
            progress('local_sampling', completed=start+len(current), total=len(positions))
            del encoded, condition
        del model, alpha, guide
        release(device)
        patches = np.concatenate(patches)
        np.save(out / 'encoded_patches.npy', patches)
        field, reference = compare_blends(patches, positions, 64 if args.smoke else 512)
        if not np.isfinite(field).all():
            raise ValueError('Non-finite generated field')
        np.save(out / 'global_field_raw.npy', field)
        del patches, reference
        if args.smoke:
            # The smoke path tests neural sampling and export; it is not a quality sample.
            field[1] = np.clip(field[1], -64, 319)
            field[0] = np.maximum(np.clip(field[0], -64, 319), field[1])
            projection = {'smoke_only': True, 'quality_assessed': False}
        else:
            field, projection = project_height_domain(field)
        write(out / 'height-domain-projection.json', projection)
        np.save(out / 'global_field_bounded.npy', field)
        if args.water == 'hydro':
            from project_global_water import project
            prior = json.loads((ROOT / 'source/training-water-level-prior.json').read_text())
            field, report = project(field, prior['modal_water_top_block_y'])
            write(out / 'water-projection.json', report)
        np.save(out / 'global_field.npy', field)
        progress('schematic_export')
        exported = export_schematic(field, out / 'terrain_full.schem', decorate=not args.no_decorate, seed=args.seed)
        write(out / 'manifest.json', dict(
            model='heightfield-expanded-v2', global_step=global_meta['step'],
            local_step=local_meta['step'], seed=args.seed, steps=args.steps, batch=args.batch,
            device=str(device), shape=list(field.shape), water=args.water,
            blending='physical_surface_ground', noise_seed_rule='seed + 1 + batch_start_index',
            full=exported, smoke_only=args.smoke, elapsed_seconds=time.time()-started,
            cross_backend_bitwise_equality_claimed=False, quantized=False,
            memory_2GB_validated=False))
        progress('completed', schematic=str(out / 'terrain_full.schem'))
    except BaseException as error:
        progress('failed', error=type(error).__name__ + ': ' + str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'musa'], default='cpu')
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--water', choices=['hydro', 'raw'], default='hydro')
    parser.add_argument('--no-decorate', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Only one 64x64 patch; requires --steps 2 explicitly; not quality validation')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.steps <= 1000 or args.batch < 1 or args.cpu_threads < 1:
        parser.error('steps must be 2..1000; batch and cpu-threads must be positive')
    if args.smoke and args.steps != 2:
        parser.error('Use --smoke --steps 2 for a quick integration check')
    run(args)


if __name__ == '__main__':
    main()
