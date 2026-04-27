# 3D Visualization of Acute Cervical Spinal Cord Injury

This document describes the postprocessing scripts under `3d_sci_visualization_tool/`.

## Scope

These scripts are for **analysis and visualization** after segmentation and preprocessing. They are separate from model training.

## Expected Preprocessing Inputs

For each subject folder, the scripts expect files such as:

- `<subject_id>_cord_mask_straight.nii.gz`
- `<subject_id>_labels_straight.nii.gz`
- `<subject_id>_lesion_mask_straight.nii.gz` (SCI script only)

How those files are produced (for example with SCT straightening and/or SpinalTotalSeg labeling) is outside this repository.

## Scripts

### 1) SCI biomarkers per slice / segment

`3d_sci_visualization_tool/compute_sci_biomarkers.py`

Outputs per subject:
- `sci_ratio_by_slice.csv`
- `sci_auc_by_segment.csv`
- `sci_ratio_curve.png`

Global output:
- `sci_auc_summary_all_subjects.csv`

Example:

```bash
python 3d_sci_visualization_tool/compute_sci_biomarkers.py \
  --base_dir "D:/spinal_cord_lesion/spinal_cord_dataset/raw" \
  --centers center_jlu center_PUTH center_tongji
```

### 2) MSCC biomarkers per slice

`3d_sci_visualization_tool/compute_mscc_biomarkers.py`

Outputs per subject:
- `mscc_by_slice.csv`
- `mscc_curve.png`

Global output:
- `mscc_summary_all_subjects.csv`

Example:

```bash
python 3d_sci_visualization_tool/compute_mscc_biomarkers.py \
  --base_dir "D:/spinal_cord_lesion/spinal_cord_dataset/raw" \
  --centers center_jlu center_PUTH center_tongji
```

### 3) 3D combined rendering (SCI + MSCC)

`3d_sci_visualization_tool/render_spine_biomarker_3d.py`

Input per subject:
- `sci_ratio_by_slice.csv`
- `mscc_by_slice.csv`

Output per subject:
- default `spine_biomarker_3d.png` (customizable)

Example:

```bash
python 3d_sci_visualization_tool/render_spine_biomarker_3d.py \
  --base_dir "D:/spinal_cord_lesion/spinal_cord_dataset/raw/center_tongji" \
  --output_name "spine_biomarker_3d.png"
```

## Reproducibility Notes for Papers

- Report third-party tool versions (for example SCT, SpinalTotalSeg) and command settings.
- Report sampling/smoothing parameters used by these scripts.
- Keep a frozen copy of generated CSV outputs used in figures/tables.
