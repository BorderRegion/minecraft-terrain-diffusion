"""Apply the same final density math on CPU,avoiding amplified MUSA division error."""
import numpy as np
import torch
from global_voxel_diffusion_training_v3 import density_probabilities


def project_density_cpu(raw,device_probabilities=None):
    signed=np.asarray(raw);assert signed.dtype==np.float32 and signed.ndim==4 and signed.shape[0]==13 and np.isfinite(signed).all()
    # This is the original projection function on CPU,not a new spatial rule.
    with torch.no_grad():probability=density_probabilities(torch.from_numpy(np.ascontiguousarray(signed[None])))[0].numpy()
    assert np.isfinite(probability).all() and probability.min()>=0 and probability.max()<=1
    assert np.allclose(probability.sum(0),1,rtol=0,atol=2e-6)
    report=dict(backend='CPU float32; original clamp/square/normalize/argmax-fallback function',
                maximum_probability_sum_error=float(np.abs(probability.sum(0)-1).max()),learned_signed_field_modified=False,spatial_repair_applied=False)
    if device_probabilities is not None:
        old=np.asarray(device_probabilities,np.float32);assert old.shape==probability.shape and np.isfinite(old).all()
        report.update(device_probability_sum_maximum_error=float(np.abs(old.sum(0)-1).max()),
                      device_probability_sum_cells_outside_original_tolerance=int((np.abs(old.sum(0)-1)>2e-5).sum()),
                      maximum_cpu_vs_device_probability_difference=float(np.abs(probability-old).max()),
                      dominant_class_changed_cells=int((probability.argmax(0)!=old.argmax(0)).sum()))
    return probability,report
