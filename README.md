# Spinal cord multimodal lesion segmentation (scl)

A PyTorch project for **multimodal spinal cord lesion segmentation** from MRI (T1, T2, T2FS, and optionally cord-based inputs). The default model is a multimodal U–Net-style architecture with dual CBAM, Swin blocks, deep supervision, etc. See `models/` and `configs/`.

**Training volumes are not included** in this repository. Bring your own NIfTI files and JSON splits as described in the documentation.

## Documentation

- **[Dataset format and JSON schema](docs/DATASET.md)** — dataset layout, `train.json` / `val.json` / `test.json` fields, and how paths join with `data_dir` and the `raw/` subfolder. Read this before wiring your data.
- **[Method and figures](docs/method.md)** — framework overview, model architecture, and uncertainty-boundary visualization.

## Framework Overview

![Framework](docs/images/framework.png)

## Environment

- Python 3.10+ (pinned to 3.10 in `environment.yml`)
- Create env: `conda env create -f environment.yml && conda activate sci`

### PyTorch install (CUDA required)

This project is configured for GPU training. `pip install -r requirements.txt` may install the CPU wheel of PyTorch by default, so install CUDA PyTorch explicitly first, then install the remaining requirements.

```bash
# Example for CUDA 12.4
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Verify current torch build:**

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda)"
```

## Quick start

```bash
# Train (prepare data and paths per docs/DATASET.md)
python train_lesion_unet.py --config configs/multimodal_lesion_unet.yaml

# Evaluate
python test_lesion_unet.py --config configs/multimodal_lesion_unet.yaml --checkpoint <path/to/best_*.pt>

# Resume
python resume_training.py --checkpoint <path/to/weights/*.pt>
# or pick the latest run under experiments/*/weights
python resume_training.py --auto
```

## `inference/` (optional uncertainty)

- **`inference/uncertainty.py`** — Monte Carlo Dropout and epistemic-boundary helpers used **after** training, not by the default `train_lesion_unet.py` loop.
- **`tools/generate_uncertainty_boundaries.py`** — CLI that loads a checkpoint, calls `inference.uncertainty`, and writes NIfTI maps under `output_dir/<split>/uncertainty_boundaries/` for the optional `uncertainty_boundary` input channel (enable with `model.include_uncertainty` and `data.uncertainty_boundary_dir`).

## `tools/`

- `tools/create_spinal_cord_dataset.py` — **example** layout/split script from the original multi-center setup; it may not match your folders. Use `docs/DATASET.md` and `data/dataloader.py` as the source of truth to build your own JSON.
- `tools/analyze_dataset.py` — NIfTI statistics (spacing, shape, intensity) to sanity-check data against `spatial_size`, `target_spacing`, and related YAML settings.

## Analysis and Visualization

- `3d_sci_visualization_tool/` — postprocessing scripts used for SCI/MSCC quantification and 3D rendering in the paper.
- See `docs/3d_visualization_of_acute_cervical_spinal_cord_injury.md` for expected inputs, CLI examples, and output files.
- Canonical scripts: `compute_sci_biomarkers.py`, `compute_mscc_biomarkers.py`, `render_spine_biomarker_3d.py`.

## Third-Party Dependencies (Analysis Stage)

The analysis pipeline can depend on external preprocessing tools (for example, SpinalTotalSeg and Spinal Cord Toolbox straightening) to generate intermediate files such as straightened cord masks and vertebral-level labels.

- This repository does **not** redistribute those third-party binaries, models, or weights.
- Install and use those tools separately under their own licenses.
- In publications, report their exact version/commit and key command parameters.

## Configuration

- Main config: `configs/multimodal_lesion_unet.yaml` (`model.type: multimodal_lesion_unet`)

## License

Add a `LICENSE` file of your choice in the repository root.
