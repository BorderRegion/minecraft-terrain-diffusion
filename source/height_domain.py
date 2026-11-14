"""Original height-domain projection, extracted unchanged from the delivery adapter."""
import numpy as np

def project_height_domain(field):
    """Project physical heights into the representable world, without smoothing."""
    if field.shape != (9, 512, 512) or not np.isfinite(field).all():
        raise ValueError('Expected a complete finite physical field')
    if np.any(field[0] < field[1]-1e-4):
        raise ValueError('Input surface is below its ground')
    result = field.copy()
    result[1] = np.clip(field[1], -64, 319)
    result[0] = np.maximum(np.clip(field[0], -64, 319), result[1])
    assert np.array_equal(field[2:], result[2:])
    assert result[:2].min() >= -64 and result[:2].max() <= 319
    changed = np.any(result != field, axis=0)
    positions = np.argwhere(changed)
    report = {
        'policy': 'G=clip(G,-64,319); S=max(clip(S,-64,319),G); materials unchanged',
        'shape': list(field.shape), 'source_all_finite': True, 'result_all_finite': True,
        'changed_columns': int(changed.sum()),
        'changed_surface_columns': int(np.count_nonzero(result[0] != field[0])),
        'changed_ground_columns': int(np.count_nonzero(result[1] != field[1])),
        'maximum_height_change_blocks': float(np.abs(result[:2]-field[:2]).max()),
        'source_height_range': [float(field[:2].min()), float(field[:2].max())],
        'result_height_range': [float(result[:2].min()), float(result[:2].max())],
        'material_channels_bitwise_unchanged': True,
        'all_channels_at_other_columns_bitwise_unchanged': bool(np.array_equal(result[:, ~changed], field[:, ~changed])),
        'float_wet_mask_unchanged': bool(np.array_equal(result[0]-result[1] > 1, field[0]-field[1] > 1)),
        'changed_column_values': [
            {'zx': [int(z), int(x)], 'source_surface': float(field[0,z,x]),
             'source_ground': float(field[1,z,x]), 'result_surface': float(result[0,z,x]),
             'result_ground': float(result[1,z,x])} for z,x in positions],
        'note': 'A format-domain projection, not learned output, terrain smoothing or an aesthetic repair. The original unbounded field remains preserved.',
    }
    return result, report
