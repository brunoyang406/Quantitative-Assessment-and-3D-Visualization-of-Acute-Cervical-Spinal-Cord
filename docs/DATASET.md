# Custom datasets and JSON format

This repository **does not ship full medical volumes**. You must provide your own NIfTI files and build `train.json` / `val.json` / `test.json` as described below. In the config, set `data.data_dir` to the **dataset root directory**.

## 1. Directories and path resolution

- `data.data_dir` in YAML (e.g. `./spinal_cord_dataset`) is the **dataset root**.
- Image paths in the list are resolved in code as:

  `Full path = <data_dir> / "raw" / <relative path from JSON>`

  Store volumes under **`<data_dir>/raw/`** and in JSON use paths **relative to `raw`** (forward or backward slashes; they are normalized).

Example: if `data_dir` is `spinal_cord_dataset` and the JSON path is:

`center_a/sub001/T1.nii.gz`

the file on disk should be:

`spinal_cord_dataset/raw/center_a/sub001/T1.nii.gz`

## 2. Top-level JSON structure

The file must be a **single object** with a **`"data"`** key. `load_data_list` loads the `data` field as a list of samples. Other top-level fields (e.g. `name`, `description`) are optional documentation and are **not** read by the loader.

```json
{
  "name": "My dataset",
  "data": [ /* array, see below */ ]
}
```

## 3. Fields per sample

| Field | Required | Description |
|-------|----------|-------------|
| `subject_id` | Strongly recommended | Unique case ID (logging and debugging). |
| `T1` | Yes* | T1 path relative to `raw`. |
| `T2` | Yes* | T2 path. |
| `T2FS` | Yes* | T2 fat-sat path. |
| `cord_mask` | Yes* | Spinal-cord label/mask, **spatially aligned** with T1, etc.; used as the 4th input channel and for mask-based normalization (see config). |
| `lesion_mask` | Yes* | Lesion mask, segmentation target. |
| `center` | No | Optional site/center metadata; may be empty. |
| `injury_grade` | No | Arbitrary string; currently passed through only. |

\*In `prepare_data_dicts`, empty string skips path joining. For the default **multimodal lesion** setup, **T1, T2, T2FS, cord_mask, and lesion_mask** must resolve to existing files; otherwise the pipeline will fail.

## 4. Volumes and labels

- **Format:** 3D NIfTI (`.nii` / `.nii.gz`) is recommended, as loaded by MONAI / nibabel.
- **Alignment:** Modalities and both masks should already be in a **common physical space** consistent with your preprocessing. The code applies `Orientation` (e.g. RAS) and resampling per `data.target_spacing`, `data.spatial_size`, and `roi_crop` in the YAML.
- **Lesion / cord labels:** They are **binarized** in augmentation/loss. Multi-class data must be merged to binary, or you must change loss/outputs (default is a single binary lesion channel).

## 5. Link to `configs/multimodal_lesion_unet.yaml`

- `data.spatial_size`, `data.target_spacing`, and `roi_crop` define the voxel grid and ROI. When switching datasets, run `tools/analyze_dataset.py` to review **spacing / shape** and tune these fields.
- If `data.uncertainty_boundary_dir` is `null`, the `uncertainty_boundary` modality is off. If you enable it, place extra volumes as expected in `data/dataloader.py`. To **generate** those maps from a trained checkpoint, use `tools/generate_uncertainty_boundaries.py` (MC Dropout + boundary logic in `inference/uncertainty.py`).

## 6. Optional tool scripts

| Script | Role |
|--------|------|
| `tools/create_spinal_cord_dataset.py` | Example for multi-center **folder layout and splits** used by the authors; it may not match your paths. Use it as a **reference** for writing JSON/ splits. |
| `tools/analyze_dataset.py` | NIfTI statistics (shape, spacing, intensity) to check consistency with the YAML. |
| `tools/generate_uncertainty_boundaries.py` | Optional: writes `uncertainty_boundary` NIfTIs; library code is under `inference/`, not `utils/`. |

You do **not** have to run these to train: JSON + valid files under `raw/` that follow this spec are enough.

## 7. Minimal example layout

```text
my_dataset/
  train.json
  val.json
  test.json
  raw/
    case_001/
      001_T1.nii.gz
      001_T2.nii.gz
      001_T2FS.nii.gz
      001_cord_mask.nii.gz
      001_lesion_mask.nii.gz
```

One `train.json` record (paths relative to `raw`):

```json
{
  "data": [
    {
      "subject_id": "001",
      "center": "",
      "injury_grade": "",
      "T1": "case_001/001_T1.nii.gz",
      "T2": "case_001/001_T2.nii.gz",
      "T2FS": "case_001/001_T2FS.nii.gz",
      "cord_mask": "case_001/001_cord_mask.nii.gz",
      "lesion_mask": "case_001/001_lesion_mask.nii.gz"
    }
  ]
}
```

Set `data.data_dir: "./my_dataset"` in the YAML and point `train_json` / `val_json` / `test_json` to the JSON files.

## 8. Code reference

- List loading and path join: `data/dataloader.py` — `load_data_list`, `prepare_data_dicts`
- Keys and transforms: `data/multimodal_transforms.py` — `get_multimodal_lesion_unet_transforms`
