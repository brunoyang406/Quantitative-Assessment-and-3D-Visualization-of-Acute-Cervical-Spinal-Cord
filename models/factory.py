"""
Build segmentation models from config.

Only production path: ``model.type`` ``multimodal_lesion_unet`` → :class:`MultimodalLesionUNet`.
Legacy config alias: ``single_task_lesion`` (mapped to the same class).
"""

import os
import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .multimodal_lesion_unet import MultimodalLesionUNet


def create_model(
    model_config: Dict[str, Any],
    img_size: Tuple[int, int, int],
) -> nn.Module:
    """
    Instantiate model from ``model_config`` and move to CUDA if available.

    ``model_config`` must include ``type`` (default ``multimodal_lesion_unet``).
    ``img_size`` is ``(D, H, W)`` passed to the network as ``input_size``.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    model_type = model_config.get("type", "multimodal_lesion_unet").lower()
    if model_type == "single_task_lesion":
        model_type = "multimodal_lesion_unet"
        logging.info("model.type 'single_task_lesion' is deprecated; use 'multimodal_lesion_unet'.")

    logging.info("Creating model: %s", model_type)

    if model_type != "multimodal_lesion_unet":
        raise ValueError(
            f"Unsupported model type: {model_type!r}. "
            "Supported: 'multimodal_lesion_unet' (alias: 'single_task_lesion')."
        )

    spatial_dims = model_config.get("spatial_dims", 3)
    in_channels = model_config.get("in_channels", 4)
    lesion_out_channels = model_config.get("lesion_out_channels", 1)
    include_cord = model_config.get("include_cord", True)
    include_uncertainty = model_config.get("include_uncertainty", False)
    channels = model_config.get(
        "channels", [32, 64, 128, 256, 320, 320, 320, 320]
    )
    strides = model_config.get(
        "strides",
        [(1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
    )
    kernel_sizes = model_config.get("kernel_sizes", None)
    num_res_units = model_config.get("num_res_units", 2)
    norm = model_config.get("norm_name", "instance")
    dropout = model_config.get("dropout_rate", 0.0)
    cbam_reduction = model_config.get("cbam_reduction", 16)
    deep_supervision = model_config.get("deep_supervision", True)
    deep_supervision_heads = model_config.get("deep_supervision_heads", 7)
    use_swin = model_config.get("use_swin_bottleneck", True)

    if model_config.get("use_dynunet", False):
        logging.warning("DynUNet is ignored; MultimodalLesionUNet uses fixed U-Net-style blocks.")

    model = MultimodalLesionUNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=lesion_out_channels,
        channels=tuple(channels),
        strides=tuple(strides),
        kernel_sizes=tuple(kernel_sizes) if kernel_sizes is not None else None,
        num_res_units=num_res_units,
        norm=norm,
        dropout=dropout,
        cbam_reduction=cbam_reduction,
        deep_supervision=deep_supervision,
        deep_supervision_heads=deep_supervision_heads,
        include_cord=include_cord,
        include_uncertainty=include_uncertainty,
        use_swin_bottleneck=use_swin,
        swin_from_stage=model_config.get("swin_from_stage", 4),
        input_size=img_size,
    ).to(device)

    modalities = ["T1", "T2", "T2FS"]
    if include_cord:
        modalities.append("cord_mask")
    if include_uncertainty:
        modalities.append("uncertainty_boundary")

    logging.info(
        "MultimodalLesionUNet: modalities=%s in_ch=%d out_ch=%d ds=%s heads=%d swin=%s bottleneck_res=%s",
        "+".join(modalities),
        in_channels,
        lesion_out_channels,
        deep_supervision,
        deep_supervision_heads,
        use_swin,
        model.bottleneck_res,
    )
    logging.info("  channels=%s strides=%s cbam_reduction=%s", channels, strides, cbam_reduction)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Parameters: total=%s trainable=%s", f"{n_total:,}", f"{n_train:,}")

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Return parameter counts and ``model.__class__.__name__``."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_type": model.__class__.__name__,
    }
