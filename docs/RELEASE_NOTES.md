# Heightfield expanded-v2 deployment release

The default deployment is now the user-confirmed early 2.5D heightfield model, selected global EMA step 98000 plus conditional local EMA step 100000. The previously published fragmented 3D voxel source remains only under `legacy/voxel-v3/`.

This release includes actual inference weight assets, source-bound game screenshots, a portable inference entrypoint, a mandatory world-height projection and the original optional hydrostatic water decoder. The default uses that water decoder, matching the historical reviewed screenshot pipeline.

EMA storage bytes are copied unchanged from the preserved formal training checkpoint ZIP records and checked by ZIP CRC during extraction. Containers and metadata are repacked, so use the new SHA-256 values in `WEIGHTS.json`; do not substitute historical inference export container checksums. No quantization or extra training is performed.

CPU integration checks exercise both real selected networks and schematic export. A two-step 64x64 smoke output is a deployment check, not a quality sample or a new full-region visual acceptance. Full CPU/CUDA cross-backend parity, a 2 GB peak process-memory limit and seamless unlimited generation are not claimed.
