"""MONAI transforms for multimodal lesion segmentation (T1, T2, T2FS, cord_mask → image; lesion_mask target)."""

import logging
import os
from typing import Any, Dict, Hashable, Mapping, Tuple

import numpy as np
import torch
from monai.transforms import (
    AsDiscreted,
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandScaleIntensityd,
    SpatialPadd,
    Spacingd,
)

# --- spacing: config [X,Y,Z] mm → Spacingd pixdim for RAS (D,H,W) order
def _reorder_spacing_for_ras(target_spacing):
    if len(target_spacing) != 3:
        raise ValueError(f"target_spacing must have 3 elements, got {len(target_spacing)}")
    return (target_spacing[0], target_spacing[2], target_spacing[1])


class CropBasedOnCordTransform(MapTransform):
    """ROI crop + pad to target size from cord mask bbox + margin."""

    def __init__(
        self,
        keys: tuple,
        cord_key: str = "cord_mask",
        target_size: tuple = (16, 512, 256),
        margin: tuple = (1, 27, 27),
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.cord_key = cord_key
        self.target_size = np.array(target_size)
        self.margin = np.array(margin)

    def _get_cord_bbox(self, cord_data: np.ndarray) -> tuple:
        coords = np.where(cord_data > 0)
        if len(coords[0]) == 0:
            return np.array([0, 0, 0]), np.array(cord_data.shape) - 1
        min_c = np.array([coords[0].min(), coords[1].min(), coords[2].min()])
        max_c = np.array([coords[0].max(), coords[1].max(), coords[2].max()])
        return min_c, max_c

    def _compute_crop_indices(
        self, min_coords: np.ndarray, max_coords: np.ndarray, img_shape: np.ndarray
    ) -> tuple:
        crop_start = np.zeros(3, dtype=int)
        crop_end = np.zeros(3, dtype=int)
        pad_before = np.zeros(3, dtype=int)
        pad_after = np.zeros(3, dtype=int)
        for i in range(3):
            roi_start = max(0, min_coords[i] - self.margin[i])
            roi_end = min(img_shape[i], max_coords[i] + self.margin[i] + 1)
            roi_actual_size = roi_end - roi_start
            if roi_actual_size >= self.target_size[i]:
                center = (roi_start + roi_end) // 2
                crop_start[i] = max(0, center - self.target_size[i] // 2)
                crop_end[i] = min(img_shape[i], crop_start[i] + self.target_size[i])
                if crop_end[i] - crop_start[i] < self.target_size[i]:
                    crop_start[i] = max(0, crop_end[i] - self.target_size[i])
            else:
                crop_start[i] = roi_start
                crop_end[i] = roi_end
                total_pad = self.target_size[i] - roi_actual_size
                pad_before[i] = total_pad // 2
                pad_after[i] = total_pad - pad_before[i]
        return crop_start, crop_end, pad_before, pad_after

    def _crop_and_pad(
        self,
        data: np.ndarray,
        crop_start: np.ndarray,
        crop_end: np.ndarray,
        pad_before: np.ndarray,
        pad_after: np.ndarray,
        is_label: bool = False,
    ) -> np.ndarray:
        assert len(data.shape) == 4, f"Expected 4D (C,D,H,W), got {data.shape}"
        cropped = data[
            :,
            crop_start[0] : crop_end[0],
            crop_start[1] : crop_end[1],
            crop_start[2] : crop_end[2],
        ]
        pad_width = [
            (0, 0),
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2]),
        ]
        padded = np.pad(cropped, pad_width, mode="constant", constant_values=0)
        assert len(padded.shape) == 4
        return padded

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        cord_data = d[self.cord_key]
        if isinstance(cord_data, torch.Tensor):
            cord_data = cord_data.numpy()
        if cord_data.ndim == 4:
            cord_for_bbox = cord_data[0]
            spatial_dims = cord_data.shape[1:]
        elif cord_data.ndim == 3:
            cord_for_bbox = cord_data
            spatial_dims = cord_data.shape
        else:
            raise ValueError(f"Unexpected cord_data shape: {cord_data.shape}")

        min_coords, max_coords = self._get_cord_bbox(cord_for_bbox)
        img_shape = np.array(spatial_dims)
        crop_start, crop_end, pad_before, pad_after = self._compute_crop_indices(
            min_coords, max_coords, img_shape
        )

        for key in self.keys:
            if key not in d:
                continue
            data_item = d[key]
            was_torch = isinstance(data_item, torch.Tensor)
            if was_torch:
                data_item = data_item.numpy()
            if data_item.ndim == 3:
                data_item = data_item[np.newaxis, ...]
            elif data_item.ndim != 4:
                raise ValueError(f"Unexpected shape for {key}: {data_item.shape}")

            data_spatial_dims = np.array(data_item.shape[1:])
            if not np.array_equal(data_spatial_dims, img_shape):
                logging.warning(
                    f"Dimension mismatch for {key}: expected {img_shape}, got {data_spatial_dims}; using safe crop."
                )
                safe_shape = np.minimum(img_shape, data_spatial_dims)
                safe_min = np.minimum(min_coords, safe_shape - 1)
                safe_max = np.minimum(max_coords, safe_shape - 1)
                actual_crop_start, actual_crop_end, actual_pad_before, actual_pad_after = (
                    self._compute_crop_indices(safe_min, safe_max, safe_shape)
                )
                actual_crop_start = np.clip(actual_crop_start, 0, data_spatial_dims - 1)
                actual_crop_end = np.clip(actual_crop_end, actual_crop_start + 1, data_spatial_dims)
            else:
                actual_crop_start, actual_crop_end = crop_start, crop_end
                actual_pad_before, actual_pad_after = pad_before, pad_after

            is_label = "mask" in key.lower()
            cropped = self._crop_and_pad(
                data_item,
                actual_crop_start,
                actual_crop_end,
                actual_pad_before,
                actual_pad_after,
                is_label=is_label,
            )
            if not is_label and cropped.dtype != np.float32:
                cropped = cropped.astype(np.float32)
            if was_torch:
                cropped = torch.from_numpy(cropped)
                if not is_label and cropped.dtype != torch.float32:
                    cropped = cropped.float()
            d[key] = cropped
        return d


class ImprovedNormalizeIntensityWithMaskd(MapTransform):
    """Intensity norm on `keys` using optional mask region; method: percentile_clip_minmax | zscore | robust."""

    def __init__(
        self,
        keys: tuple,
        mask_key: str = "cord_mask",
        use_mask: bool = True,
        channel_wise: bool = True,
        nonzero: bool = False,
        method: str = "percentile_clip_minmax",
        lower_percentile: float = 0.5,
        upper_percentile: float = 99.5,
        robust_scale: float = 1.35,
        minmax_range: tuple = (0, 1),
    ):
        super().__init__(keys)
        self.mask_key = mask_key
        self.use_mask = use_mask
        self.channel_wise = channel_wise
        self.nonzero = nonzero
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.robust_scale = robust_scale
        self.minmax_range = minmax_range

    def _get_region_data(self, channel_data, mask=None):
        if mask is not None:
            return channel_data[mask]
        if self.nonzero:
            nz = channel_data != 0
            return channel_data[nz] if nz.sum() > 0 else channel_data.flatten()
        return channel_data.flatten()

    def _percentile_clip_zscore(self, channel_data, region_data):
        p_lo, p_hi = np.percentile(region_data, [self.lower_percentile, self.upper_percentile])
        channel_clipped = np.clip(channel_data, p_lo, p_hi)
        region_clipped = np.clip(region_data, p_lo, p_hi)
        mean, std = region_clipped.mean(), region_clipped.std()
        return (channel_clipped - mean) / std if std > 1e-8 else channel_clipped - mean

    def _robust_normalize(self, channel_data, region_data):
        median = np.median(region_data)
        q25, q75 = np.percentile(region_data, [25, 75])
        iqr = q75 - q25
        return (channel_data - median) / (iqr / self.robust_scale) if iqr > 1e-8 else channel_data - median

    def _percentile_clip_minmax(self, channel_data, region_data):
        p_lo, p_hi = np.percentile(region_data, [self.lower_percentile, self.upper_percentile])
        channel_clipped = np.clip(channel_data, p_lo, p_hi)
        if p_hi > p_lo:
            normalized = (channel_clipped - p_lo) / (p_hi - p_lo)
            lo, hi = self.minmax_range
            return normalized * (hi - lo) + lo
        return channel_clipped - p_lo

    def _normalize_channel(self, channel_data, region_data):
        if len(region_data) == 0:
            return channel_data
        method_map = {
            "percentile_clip_zscore": self._percentile_clip_zscore,
            "robust": self._robust_normalize,
            "percentile_clip_minmax": self._percentile_clip_minmax,
        }
        fn = method_map.get(self.method)
        if fn:
            return fn(channel_data, region_data)
        mean, std = region_data.mean(), region_data.std()
        return (channel_data - mean) / std if std > 1e-8 else channel_data - mean

    def _get_mask(self, data):
        if not self.use_mask or self.mask_key not in data:
            return None, False
        mask_data = data[self.mask_key]
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.numpy()
        if mask_data.ndim == 4:
            mask_data = mask_data[0]
        elif mask_data.ndim != 3:
            return None, False
        mask = mask_data > 0
        return mask, mask.sum() > 0

    def _normalize_image(self, image, mask, use_mask_for_norm):
        if self.channel_wise:
            for c in range(image.shape[0]):
                region = self._get_region_data(image[c], mask if use_mask_for_norm else None)
                image[c] = self._normalize_channel(image[c], region)
        else:
            if use_mask_for_norm:
                region_data = np.concatenate([image[c][mask] for c in range(image.shape[0])])
            else:
                region_data = self._get_region_data(image, None)
            for c in range(image.shape[0]):
                image[c] = self._normalize_channel(image[c], region_data)
        return image

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        mask, use_mask_for_norm = self._get_mask(d)
        for key in self.keys:
            if key not in d:
                continue
            image = d[key]
            was_torch = isinstance(image, torch.Tensor)
            if was_torch:
                image = image.numpy()
            if image.ndim == 3:
                image = image[np.newaxis, ...]
            image = self._normalize_image(image, mask, use_mask_for_norm)
            image = image.astype(np.float32)
            if was_torch:
                image = torch.from_numpy(image)
            d[key] = image
        return d


class ConcatMultimodald(MapTransform):
    """Stack T1,T2,T2FS (+ optional cord, uncertainty) into `output_key` (C,D,H,W)."""

    def __init__(
        self,
        keys: Tuple[str, ...] = ("T1", "T2", "T2FS"),
        cord_key: str = "cord_mask",
        uncertainty_key: str = "uncertainty_boundary",
        output_key: str = "image",
        include_cord: bool = True,
        include_uncertainty: bool = True,
    ):
        super().__init__(keys)
        self.cord_key = cord_key
        self.uncertainty_key = uncertainty_key
        self.output_key = output_key
        self.include_cord = include_cord
        self.include_uncertainty = include_uncertainty

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        images = []
        for key in self.keys:
            if key not in d:
                raise KeyError(f"Key '{key}' not found")
            img = d[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            elif img.ndim != 3:
                raise ValueError(f"Unexpected image shape for {key}: {img.shape}")
            images.append(img)

        if self.include_cord:
            cord_ok = (
                self.cord_key in d
                and d[self.cord_key] is not None
                and (not isinstance(d[self.cord_key], str) or d[self.cord_key].strip() != "")
            )
            if cord_ok:
                cord_img = d[self.cord_key]
                if isinstance(cord_img, torch.Tensor):
                    cord_img = cord_img.numpy()
                if cord_img.ndim == 4 and cord_img.shape[0] == 1:
                    cord_img = cord_img[0]
                elif cord_img.ndim != 3:
                    raise ValueError(f"Unexpected cord_mask shape: {cord_img.shape}")
                if cord_img.dtype != np.float32:
                    cord_img = cord_img.astype(np.float32)
                images.append(cord_img)
            else:
                if not images:
                    raise KeyError(f"Cord key '{self.cord_key}' missing and cannot infer shape")
                images.append(np.zeros_like(images[0], dtype=np.float32))

        if self.include_uncertainty:
            u_ok = (
                self.uncertainty_key in d
                and d[self.uncertainty_key] is not None
                and (not isinstance(d[self.uncertainty_key], str) or d[self.uncertainty_key].strip() != "")
            )
            if u_ok:
                u_img = d[self.uncertainty_key]
                if isinstance(u_img, torch.Tensor):
                    u_img = u_img.numpy()
                if u_img.ndim == 4 and u_img.shape[0] == 1:
                    u_img = u_img[0]
                elif u_img.ndim != 3:
                    raise ValueError(f"Unexpected uncertainty shape: {u_img.shape}")
                if u_img.dtype != np.float32:
                    u_img = u_img.astype(np.float32)
                images.append(u_img)
            else:
                if not images:
                    raise KeyError(f"Uncertainty key '{self.uncertainty_key}' missing")
                images.append(np.zeros_like(images[0], dtype=np.float32))

        stacked = np.stack(images, axis=0).astype(np.float32)
        d[self.output_key] = stacked
        return d


class ValidatePathsTransform(MapTransform):
    """Fail fast on bad file paths before LoadImaged (string paths only)."""

    def __init__(self, keys: Tuple[str, ...]):
        super().__init__(keys)

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)
        sid = d.get("subject_id", "UNKNOWN_SUBJECT")
        for key in self.keys:
            path = d.get(key)
            if not isinstance(path, str):
                continue
            if path.strip() == "":
                raise RuntimeError(f"Empty path for '{key}' (subject_id={sid})")
            if path.strip() == "..":
                raise RuntimeError(f"Invalid path '..' for '{key}' (subject_id={sid})")
            if not os.path.exists(path):
                raise RuntimeError(f"File not found for '{key}' (subject_id={sid}): {path}")
        return d


def _transforms_load_through_concat(
    *,
    all_keys: list,
    image_keys: list,
    mask_key: str,
    norm_mask_key: str,
    label_key: str,
    uncertainty_key: str,
    include_uncertainty: bool,
    target_spacing: tuple,
    use_roi_crop: bool,
    roi_target_size: tuple,
    roi_margin: tuple,
    normalize_config: dict,
    use_mask_normalization: bool,
    include_cord: bool,
    include_uncertainty_concat: bool,
) -> list:
    pixdim = _reorder_spacing_for_ras(target_spacing)
    modes = tuple("bilinear" if k in image_keys else "nearest" for k in all_keys)
    mask_cast = [mask_key, label_key]
    if include_uncertainty:
        mask_cast.insert(1, uncertainty_key)

    norm_method = normalize_config.get("normalization_method", "percentile_clip_minmax")
    lower_p = normalize_config.get("lower_percentile", 0.5)
    upper_p = normalize_config.get("upper_percentile", 99.5)
    robust_scale = normalize_config.get("robust_scale", 1.35)
    minmax_range = normalize_config.get("minmax_range", [0, 1])

    blocks = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        CastToTyped(keys=image_keys, dtype=np.float32),
        CastToTyped(keys=mask_cast, dtype=np.float32),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=all_keys, pixdim=pixdim, mode=modes),
    ]
    if use_roi_crop:
        blocks.append(
            CropBasedOnCordTransform(
                keys=all_keys,
                cord_key=mask_key,
                target_size=roi_target_size,
                margin=roi_margin,
            )
        )
    blocks.append(
        ImprovedNormalizeIntensityWithMaskd(
            keys=tuple(image_keys),
            mask_key=norm_mask_key,
            use_mask=use_mask_normalization,
            channel_wise=True,
            nonzero=False,
            method=norm_method,
            lower_percentile=lower_p,
            upper_percentile=upper_p,
            robust_scale=robust_scale,
            minmax_range=tuple(minmax_range),
        )
    )
    blocks.append(
        ConcatMultimodald(
            keys=tuple(image_keys),
            cord_key=mask_key,
            uncertainty_key=uncertainty_key,
            output_key="image",
            include_cord=include_cord,
            include_uncertainty=include_uncertainty_concat,
        )
    )
    return blocks


def _augmentation_list(
    label_key: str,
    aug_config: dict,
) -> list:
    flip_prob = aug_config.get("flip_prob", 0.5)
    use_rotate = aug_config.get("use_rotate", True)
    rotate_prob = aug_config.get("rotate_prob", 0.3)
    rotate_range = aug_config.get("rotate_range", 0.15)
    use_contrast = aug_config.get("use_contrast", True)
    contrast_prob = aug_config.get("contrast_prob", 0.3)
    gamma_range = aug_config.get("gamma_range", (0.7, 1.5))
    use_intensity_scale = aug_config.get("use_intensity_scale", True)
    intensity_scale_prob = aug_config.get("intensity_scale_prob", 0.4)
    intensity_scale_factor = aug_config.get("intensity_scale_factor", 0.2)
    use_gaussian_noise = aug_config.get("use_gaussian_noise", True)
    gaussian_noise_prob = aug_config.get("gaussian_noise_prob", 0.2)
    gaussian_noise_std = aug_config.get("gaussian_noise_std", 0.01)

    aug = [
        RandFlipd(keys=["image", label_key], spatial_axis=[0], prob=flip_prob),
        RandFlipd(keys=["image", label_key], spatial_axis=[1], prob=flip_prob),
        RandFlipd(keys=["image", label_key], spatial_axis=[2], prob=flip_prob),
    ]
    if use_rotate:
        aug.append(
            RandRotated(
                keys=["image", label_key],
                range_x=rotate_range,
                range_y=rotate_range,
                range_z=rotate_range,
                prob=rotate_prob,
                mode=("bilinear", "nearest"),
                padding_mode="constant",
                keep_size=True,
            )
        )
    if use_contrast:
        aug.append(RandAdjustContrastd(keys=["image"], gamma=gamma_range, prob=contrast_prob))
    if use_intensity_scale:
        aug.append(
            RandScaleIntensityd(keys=["image"], factors=intensity_scale_factor, prob=intensity_scale_prob)
        )
    if use_gaussian_noise:
        aug.append(RandGaussianNoised(keys=["image"], std=gaussian_noise_std, prob=gaussian_noise_prob))
    return aug


def get_multimodal_lesion_unet_transforms(
    config: Dict[str, Any],
) -> Tuple[Compose, Compose]:
    """Train and val Compose: outputs `image`, `lesion_mask` (binarized)."""
    spatial_size = config.get("spatial_size", (16, 512, 256))
    target_spacing = config.get("target_spacing", (3.3, 0.54, 0.54))

    model_config = config.get("model", {})
    include_cord = model_config.get("include_cord", True)
    include_uncertainty = model_config.get("include_uncertainty", False)

    roi_config = config.get("roi_crop", {})
    use_roi_crop = roi_config.get("use_roi_crop", False)
    roi_target_size = tuple(roi_config.get("roi_target_size", spatial_size))
    roi_margin = tuple(roi_config.get("roi_margin", (1, 27, 27)))

    image_keys = ["T1", "T2", "T2FS"]
    mask_key = "cord_mask"
    uncertainty_key = "uncertainty_boundary"
    label_key = "lesion_mask"

    all_keys = image_keys + [mask_key]
    if include_uncertainty:
        all_keys.append(uncertainty_key)
    all_keys.append(label_key)

    normalize_config = config.get("normalization", {})
    use_mask_normalization = normalize_config.get("use_mask", False)
    aug_config = config.get("augmentation", {})

    norm_mask_key = normalize_config.get("mask_key", mask_key)
    shared_kw = dict(
        all_keys=all_keys,
        image_keys=image_keys,
        mask_key=mask_key,
        norm_mask_key=norm_mask_key,
        label_key=label_key,
        uncertainty_key=uncertainty_key,
        include_uncertainty=include_uncertainty,
        target_spacing=target_spacing,
        use_roi_crop=use_roi_crop,
        roi_target_size=roi_target_size,
        roi_margin=roi_margin,
        normalize_config=normalize_config,
        use_mask_normalization=use_mask_normalization,
        include_cord=include_cord,
        include_uncertainty_concat=include_uncertainty,
    )

    train_list = _transforms_load_through_concat(**shared_kw)
    train_list.extend(
        [
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.int16),
        ]
    )
    if not use_roi_crop or roi_target_size != spatial_size:
        train_list.append(
            SpatialPadd(
                keys=["image", label_key],
                spatial_size=spatial_size,
                mode="constant",
                constant_values=0,
            )
        )
    train_list.extend(_augmentation_list(label_key, aug_config))
    train_list.append(AsDiscreted(keys=[label_key], threshold=0.5))
    train_transforms = Compose(train_list)

    val_list = _transforms_load_through_concat(**shared_kw)
    val_list.extend(
        [
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=[label_key], dtype=torch.int16),
        ]
    )
    if not use_roi_crop or roi_target_size != spatial_size:
        val_list.append(
            SpatialPadd(
                keys=["image", label_key],
                spatial_size=spatial_size,
                mode="constant",
                constant_values=0,
            )
        )
    val_list.append(AsDiscreted(keys=[label_key], threshold=0.5))
    val_transforms = Compose(val_list)

    modalities = ["T1", "T2", "T2FS"]
    n_ch = 3
    if include_cord:
        modalities.append("cord_mask")
        n_ch += 1
    if include_uncertainty:
        modalities.append("uncertainty_boundary")
        n_ch += 1
    logging.info("Multimodal lesion U-Net transforms:")
    logging.info(f"  Input: {', '.join(modalities)} ({n_ch} ch) | target: lesion | spatial_size={spatial_size}")
    logging.info(f"  target_spacing={target_spacing} | ROI crop={use_roi_crop}")
    if use_roi_crop:
        logging.info(f"    ROI size={roi_target_size} margin={roi_margin}")
    logging.info(f"  Mask normalization={use_mask_normalization} (norm mask_key={norm_mask_key})")

    return train_transforms, val_transforms
