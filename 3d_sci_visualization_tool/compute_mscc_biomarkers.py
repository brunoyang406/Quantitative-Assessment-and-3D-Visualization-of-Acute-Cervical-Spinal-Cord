"""
Batch MSCC analysis from straightened cord mask + vertebral level labels.

Per subject outputs:
- mscc_by_slice.csv
- mscc_curve.png

Global output:
- mscc_summary_all_subjects.csv (under base_dir)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def get_label_name(label: int) -> str:
    if label <= 2:
        return ""
    if label <= 7:
        return f"C{label-1}/{label}"
    if label == 8:
        return "C7/T1"
    return f"T{label-8}/{label-7}"


def _nearest_nonzero(arr: List[float], start: int, stop: int, step: int) -> float:
    for idx in range(start, stop, step):
        if arr[idx] > 0:
            return arr[idx]
    return 0.0


def analyze_subject(subject_dir: Path) -> Optional[List[Dict]]:
    subject_id = subject_dir.name
    cord_path = subject_dir / f"{subject_id}_cord_mask_straight.nii.gz"
    label_path = subject_dir / f"{subject_id}_labels_straight.nii.gz"
    if not all(p.exists() for p in [cord_path, label_path]):
        return None
    try:
        cord_img = nib.load(cord_path)
        label_img = nib.load(label_path)
        cord_data = cord_img.get_fdata()
        label_data = label_img.get_fdata()
        pixdim_z = float(cord_img.header.get_zooms()[2])

        ap_diameters: List[float] = []
        for z in range(cord_data.shape[2]):
            y_idx = np.where(cord_data[:, :, z] > 0)[1]
            ap_diameters.append(float(np.max(y_idx) - np.min(y_idx) + 1) if len(y_idx) > 0 else 0.0)

        ap = np.asarray(ap_diameters, dtype=float)
        valid = ap > 0
        if np.sum(valid) > 11:
            ap[valid] = savgol_filter(ap[valid], window_length=11, polyorder=2)
        ap_diameters = ap.tolist()

        reference_offset = int(round(5.0 / pixdim_z))
        exclusion_offset = int(round(20.0 / pixdim_z))
        start_idx = exclusion_offset
        end_idx = len(ap_diameters) - 1 - exclusion_offset

        mscc_full = np.full(len(ap_diameters), np.nan)
        for i in range(start_idx, end_idx + 1):
            if ap_diameters[i] == 0:
                continue
            idx_a, idx_b = i - reference_offset, i + reference_offset
            d_a = ap_diameters[idx_a]
            d_b = ap_diameters[idx_b]
            if d_a == 0:
                d_a = _nearest_nonzero(ap_diameters, idx_a, i, 1)
            if d_b == 0:
                d_b = _nearest_nonzero(ap_diameters, idx_b, i, -1)
            d_mean = (d_a + d_b) / 2.0
            if d_mean > 0:
                mscc_full[i] = (1 - ap_diameters[i] / d_mean) * 100.0

        coords = np.where(label_data > 0)
        labels_found = label_data[coords]
        z_positions = coords[2]
        label_to_z = {int(l): int(z) for l, z in zip(labels_found, z_positions)}
        sorted_labels = sorted(label_to_z.keys())
        if len(sorted_labels) < 2:
            return None

        rows: List[Dict] = []
        boundaries = [0]
        for i in range(len(sorted_labels) - 1):
            l_start, l_end = sorted_labels[i], sorted_labels[i + 1]
            z_start, z_end = int(round(label_to_z[l_start])), int(round(label_to_z[l_end]))
            segment_name = f"{get_label_name(l_start)}-{get_label_name(l_end)}"
            step = 1 if z_end >= z_start else -1
            z_range = range(z_start, z_end + step, step) if i == len(sorted_labels) - 2 else range(z_start, z_end, step)
            for z_idx in z_range:
                if z_idx < start_idx or z_idx > end_idx:
                    continue
                rows.append(
                    {
                        "Subject": subject_id,
                        "Segment": segment_name,
                        "Slice_Z": z_idx,
                        "MSCC_Percent": mscc_full[z_idx],
                    }
                )
            boundaries.append(len(rows))

        df = pd.DataFrame(rows)
        valid_series = df["MSCC_Percent"].values
        finite = np.isfinite(valid_series)
        vals = valid_series[finite]
        if len(vals) > 11:
            smooth_vals = savgol_filter(vals, window_length=11, polyorder=3)
            smooth_full = np.full_like(valid_series, np.nan, dtype=float)
            smooth_full[finite] = smooth_vals
            df["MSCC_Smoothed"] = smooth_full
        else:
            df["MSCC_Smoothed"] = df["MSCC_Percent"]

        df.to_csv(subject_dir / "mscc_by_slice.csv", index=False)
        _plot_mscc_curve(df, subject_id, sorted_labels, boundaries, subject_dir / "mscc_curve.png")
        return rows
    except Exception as e:
        logging.exception("Failed %s: %s", subject_id, e)
        return None


def _plot_mscc_curve(df: pd.DataFrame, subject_id: str, sorted_labels: List[int], segment_boundaries: List[int], output_png: Path) -> None:
    plt.figure(figsize=(12, 6))
    color = "#C06C84"
    plt.plot(df["MSCC_Percent"].values, color=color, linewidth=1, alpha=0.3, label="Original MSCC")
    plt.plot(df["MSCC_Smoothed"].values, color=color, linewidth=2.5, label="Smoothed MSCC")
    plt.axhline(0, color="#2C3E50", linestyle="-", linewidth=1, alpha=0.3)

    max_idx = df["MSCC_Smoothed"].idxmax(skipna=True)
    max_val = float(df.loc[max_idx, "MSCC_Smoothed"])
    if max_val > 0:
        plt.plot(max_idx, max_val, "o", color="#E84A5F", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        text_y = max_val + 2 if max_val < 22 else max_val - 4
        va = "bottom" if max_val < 22 else "top"
        plt.text(max_idx, text_y, f"Max MSCC: {max_val:.1f}%", fontsize=12, fontweight="bold", color="#E84A5F", ha="center", va=va)

    line_color = "#A9A9A9"
    for i, boundary in enumerate(segment_boundaries[:-1]):
        plt.axvline(x=boundary, color=line_color, linestyle="--", alpha=0.4)
        plt.text(boundary, plt.ylim()[1] * 0.95, get_label_name(sorted_labels[i]), rotation=90, va="top", fontsize=10, fontweight="bold")
    plt.axvline(x=segment_boundaries[-1], color=line_color, linestyle="--", alpha=0.4)
    plt.text(segment_boundaries[-1], plt.ylim()[1] * 0.95, get_label_name(sorted_labels[-1]), rotation=90, va="top", fontsize=10, fontweight="bold")

    plt.title(f"MSCC Analysis (Per Slice): {subject_id}", fontsize=16, fontweight="bold")
    plt.ylabel("MSCC (%)")
    plt.xlabel("Slices (label-defined axis)")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch MSCC biomarker analysis from straightened cord masks.")
    parser.add_argument("--base_dir", type=Path, required=True, help="Raw dataset directory containing center_* folders")
    parser.add_argument("--centers", nargs="+", default=["center_jlu", "center_PUTH", "center_tongji"], help="Center folder names to process")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    all_summary: List[Dict] = []
    for center in args.centers:
        center_path = args.base_dir / center
        if not center_path.exists():
            logging.warning("Center not found: %s", center_path)
            continue
        logging.info("Analyzing MSCC center: %s", center)
        for subject_dir in sorted([p for p in center_path.iterdir() if p.is_dir()]):
            result = analyze_subject(subject_dir)
            if result:
                all_summary.extend(result)
                logging.info("  done: %s", subject_dir.name)
    if all_summary:
        out_csv = args.base_dir / "mscc_summary_all_subjects.csv"
        pd.DataFrame(all_summary).to_csv(out_csv, index=False)
        logging.info("Summary written: %s", out_csv)


if __name__ == "__main__":
    main()
