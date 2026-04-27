#!/usr/bin/env python3
"""
Resume multimodal lesion U-Net training from a checkpoint.

Usage:
    python resume_training.py --checkpoint experiments/multimodal_lesion_unet_xxx/weights/best_model_lesion_dice=0.xxx_50.pt
    python resume_training.py --auto
    python resume_training.py --auto --experiments_dir ./experiments
"""

import argparse
import glob
import os
import subprocess
import sys


def _latest_checkpoint_in_experiment(exp_dir: str):
    """Return path to the newest .pt / .pth under exp_dir/weights, or None."""
    weights_dir = os.path.join(exp_dir, "weights")
    if not os.path.isdir(weights_dir):
        return None
    candidates = []
    for name in os.listdir(weights_dir):
        if name.endswith((".pt", ".pth")):
            p = os.path.join(weights_dir, name)
            if os.path.isfile(p):
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def find_latest_checkpoint(experiments_dir: str = "experiments", name_prefix: str = "multimodal_lesion_unet_"):
    """
    Pick the most recently modified experiment under experiments_dir whose
    name starts with name_prefix, then take the newest checkpoint in weights/.
    If name_prefix is empty, use any direct subdirectory. If no match, fall
    back to any subdirectory with a weights/ checkpoint.
    """
    if not os.path.isdir(experiments_dir):
        print(f"No such directory: {experiments_dir}")
        return None

    if name_prefix:
        exp_globs = [os.path.join(experiments_dir, name_prefix + "*")]
        exp_dirs = []
        for g in exp_globs:
            exp_dirs.extend(p for p in glob.glob(g) if os.path.isdir(p))
    else:
        exp_dirs = []

    if not exp_dirs:
        exp_dirs = [
            p
            for p in glob.glob(os.path.join(experiments_dir, "*"))
            if os.path.isdir(p) and not os.path.basename(p).startswith(".")
        ]

    if not exp_dirs:
        print(f"No experiment subdirectories found under {experiments_dir}")
        return None

    # Prefer latest experiment by folder mtime
    exp_dirs.sort(key=os.path.getmtime, reverse=True)
    for exp in exp_dirs:
        ckpt = _latest_checkpoint_in_experiment(exp)
        if ckpt is not None:
            return ckpt, exp

    print(f"No checkpoint files found under {experiments_dir}/*/weights/")
    return None


def main():
    default_config = "configs/multimodal_lesion_unet.yaml"
    parser = argparse.ArgumentParser(description="Resume multimodal lesion U-Net training")
    parser.add_argument("--checkpoint", type=str, help="Path to a .pt / .pth checkpoint")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Use latest experiment (by time) and newest file in its weights/ folder",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help=f"Config YAML (default: {default_config})",
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Root directory containing per-run experiment folders",
    )
    parser.add_argument(
        "--exp_prefix",
        type=str,
        default="multimodal_lesion_unet_",
        help="Only consider experiment dirs under experiments_dir/ whose names start with this (empty = any subdir)",
    )
    args = parser.parse_args()

    checkpoint_path = None
    exp_dir = None

    if args.auto:
        print("Searching for latest checkpoint...")
        result = find_latest_checkpoint(
            experiments_dir=args.experiments_dir,
            name_prefix=args.exp_prefix,
        )
        if result is None:
            sys.exit(1)
        checkpoint_path, exp_dir = result
        print(f"Found experiment: {exp_dir}")
        print(f"Latest weights file: {checkpoint_path}")
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        print("Error: use --checkpoint <path> or --auto")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)

    cmd_list = [
        sys.executable,
        "train_lesion_unet.py",
        "--config",
        args.config,
        "--resume",
        checkpoint_path,
    ]

    print("\n" + "=" * 70)
    print("Resuming with:")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Config:     {args.config}")
    print(f"\nCommand: {' '.join(cmd_list)}\n" + "=" * 70 + "\n")

    raise SystemExit(subprocess.call(cmd_list, cwd=os.path.dirname(os.path.abspath(__file__)) or "."))


if __name__ == "__main__":
    main()
