"""
Batch SCI burden analysis from straightened masks.

Per subject outputs:
- sci_ratio_by_slice.csv
- sci_auc_by_segment.csv
- sci_ratio_curve.png

Global output:
- sci_auc_summary_all_subjects.csv (under base_dir)
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.integrate import simpson


def get_label_name(label: int) -> str:
    if label <= 2:
        return ""
    if label <= 7:
        return f"C{label-1}/{label}"
    if label == 8:
        return "C7/T1"
    return f"T{label-8}/{label-7}"


def label_to_z_mapping(label_data: np.ndarray) -> Dict[int, int]:
    coords = np.where(label_data > 0)
    labels_found = label_data[coords]
    z_positions = coords[2]
    return {int(l): int(z) for l, z in zip(labels_found, z_positions)}


def segment_ranges(sorted_labels: List[int], label_to_z: Dict[int, int]) -> List[Tuple[str, range]]:
    ranges = []
    for i in range(len(sorted_labels) - 1):
        l_start, l_end = sorted_labels[i], sorted_labels[i + 1]
        z_start, z_end = int(round(label_to_z[l_start])), int(round(label_to_z[l_end]))
        segment_name = f"{get_label_name(l_start)}-{get_label_name(l_end)}"
        step = 1 if z_end >= z_start else -1
        if i == len(sorted_labels) - 2:
            z_range = range(z_start, z_end + step, step)
        else:
            z_range = range(z_start, z_end, step)
        ranges.append((segment_name, z_range))
    return ranges


def analyze_subject(subject_dir: Path) -> Optional[List[Dict]]:
    subject_id = subject_dir.name
    cord_path = subject_dir / f"{subject_id}_cord_mask_straight.nii.gz"
    lesion_path = subject_dir / f"{subject_id}_lesion_mask_straight.nii.gz"
    label_path = subject_dir / f"{subject_id}_labels_straight.nii.gz"

    if not all(p.exists() for p in [cord_path, lesion_path, label_path]):
        return None

    try:
        cord_data = nib.load(cord_path).get_fdata()
        lesion_data = nib.load(lesion_path).get_fdata()
        label_data = nib.load(label_path).get_fdata()

        label_to_z = label_to_z_mapping(label_data)
        sorted_labels = sorted(label_to_z.keys())
        if len(sorted_labels) < 2:
            logging.warning("Skip %s: not enough level labels", subject_id)
            return None

        all_ratios: List[Dict] = []
        segment_metrics: List[Dict] = []
        segment_boundaries = [0]

        for segment_name, z_range in segment_ranges(sorted_labels, label_to_z):
            segment_ratios = []
            for z_idx in z_range:
                cord_area = float(np.sum(cord_data[:, :, z_idx]))
                lesion_area = float(np.sum(lesion_data[:, :, z_idx]))
                ratio = lesion_area / cord_area if cord_area > 0 else 0.0
                ratio = min(ratio, 1.0)
                segment_ratios.append(ratio)
                all_ratios.append(
                    {
                        "Subject": subject_id,
                        "Segment": segment_name,
                        "Slice_Z": z_idx,
                        "Ratio": ratio,
                    }
                )

            if len(segment_ratios) > 1:
                try:
                    seg_auc = float(simpson(segment_ratios))
                except Exception:
                    seg_auc = float(np.trapz(segment_ratios))
            else:
                seg_auc = float(segment_ratios[0]) if segment_ratios else 0.0

            segment_metrics.append(
                {
                    "Subject": subject_id,
                    "Segment": segment_name,
                    "AUC": seg_auc,
                    "Slices_Count": len(segment_ratios),
                }
            )
            segment_boundaries.append(len(all_ratios))

        pd.DataFrame(all_ratios).to_csv(subject_dir / "sci_ratio_by_slice.csv", index=False)
        pd.DataFrame(segment_metrics).to_csv(subject_dir / "sci_auc_by_segment.csv", index=False)
        plot_sci_curve(subject_id, all_ratios, sorted_labels, segment_boundaries, subject_dir / "sci_ratio_curve.png")
        return segment_metrics
    except Exception as e:
        logging.exception("Failed %s: %s", subject_id, e)
        return None


def plot_sci_curve(subject_id: str, all_ratios: List[Dict], sorted_labels: List[int], segment_boundaries: List[int], output_png: Path) -> None:
    ratios = [r["Ratio"] for r in all_ratios]
    plt.figure(figsize=(12, 6))
    main_color = "#5B7DB1"
    plt.plot(ratios, color=main_color, linewidth=2.5)
    plt.fill_between(range(len(ratios)), ratios, color=main_color, alpha=0.2)
    plt.ylim(0, 1.0)
    line_color = "#A9A9A9"
    for i, boundary in enumerate(segment_boundaries[:-1]):
        plt.axvline(x=boundary, color=line_color, linestyle="--", alpha=0.5)
        plt.text(boundary, 0.95, get_label_name(sorted_labels[i]), rotation=90, va="top", fontsize=12, fontweight="bold", color="#333333")
    plt.axvline(x=segment_boundaries[-1], color=line_color, linestyle="--", alpha=0.5)
    plt.text(segment_boundaries[-1], 0.95, get_label_name(sorted_labels[-1]), rotation=90, va="top", fontsize=12, fontweight="bold", color="#333333")
    plt.title(f"Lesion Distribution (Per Slice): {subject_id}", fontsize=16, fontweight="bold", color="#2C3E50")
    plt.xlabel("Slices (label-defined axis)")
    plt.ylabel("Lesion/Cord Area Ratio")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SCI biomarker analysis from straightened masks.")
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
        logging.info("Analyzing SCI center: %s", center)
        for subject_dir in sorted([p for p in center_path.iterdir() if p.is_dir()]):
            result = analyze_subject(subject_dir)
            if result:
                all_summary.extend(result)
                logging.info("  done: %s", subject_dir.name)
    if all_summary:
        out_csv = args.base_dir / "sci_auc_summary_all_subjects.csv"
        pd.DataFrame(all_summary).to_csv(out_csv, index=False)
        logging.info("Summary written: %s", out_csv)


if __name__ == "__main__":
    main()
