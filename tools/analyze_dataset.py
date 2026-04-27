"""
Dataset analysis script
Statistics for shape, spacing, orientation, intensity range, etc. of all images in the dataset
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict


def analyze_spacing_distribution(physical_sizes, target_spacing, total_files):
    """
    Analyze shape distribution at a specific target spacing
    
    Args:
        physical_sizes: Array of physical sizes
        target_spacing: Target spacing, e.g., [1.0, 1.0, 1.0]
        total_files: Total number of files
    """
    from collections import Counter
    
    print("="*80)
    print(f"Target Spacing: [{target_spacing[0]:.4f}, {target_spacing[1]:.4f}, {target_spacing[2]:.4f}] mm")
    print("="*80)
    
    # Calculate resampled sizes
    resampled_shapes = physical_sizes / target_spacing
    
    print("\nResampled size distribution:")
    for dim_idx, dim_name in enumerate(['Width (X)', 'Height (Y)', 'Depth (Z)']):
        dim_values = resampled_shapes[:, dim_idx]
        print(f"{dim_name}:")
        print(f"  Min: {np.min(dim_values):.1f}")
        print(f"  Max: {np.max(dim_values):.1f}")
        print(f"  Mean: {np.mean(dim_values):.1f}")
        print(f"  Median: {np.median(dim_values):.1f}")
        print(f"  75th percentile: {np.percentile(dim_values, 75):.1f}")
        print(f"  90th percentile: {np.percentile(dim_values, 90):.1f}")
        print(f"  95th percentile: {np.percentile(dim_values, 95):.1f}")
        print()
    
    # Detailed shape combination distribution
    print("-" * 80)
    print("Complete Shape Combination Statistics:")
    print("-" * 80)
    
    # Round resampled sizes to integers
    resampled_shapes_int = np.round(resampled_shapes).astype(int)
    
    # Count occurrences of each shape combination
    shape_combinations = Counter(map(tuple, resampled_shapes_int))
    
    print(f"\nUnique shape combinations: {len(shape_combinations)}")
    print(f"Total files: {total_files}\n")
    
    # Sort by occurrence count
    sorted_shapes = sorted(shape_combinations.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_shapes) <= 20:
        print("All shape combinations and frequencies:")
        for shape_tuple, count in sorted_shapes:
            percentage = count / total_files * 100
            shape_str = f"({shape_tuple[0]}, {shape_tuple[1]}, {shape_tuple[2]})"
            print(f"  {shape_str}: {count} times ({percentage:.1f}%)")
    else:
        print("Top 20 most common shape combinations:")
        for shape_tuple, count in sorted_shapes[:20]:
            percentage = count / total_files * 100
            shape_str = f"({shape_tuple[0]}, {shape_tuple[1]}, {shape_tuple[2]})"
            print(f"  {shape_str}: {count} times ({percentage:.1f}%)")
        
        # Count remaining combinations
        remaining_count = sum(count for _, count in sorted_shapes[20:])
        remaining_percentage = remaining_count / total_files * 100
        print(f"  Other {len(sorted_shapes) - 20} combinations: {remaining_count} times ({remaining_percentage:.1f}%)")
    
    print()
    
    # Detailed percentile distribution for each dimension
    print("-" * 80)
    print("Detailed Percentile Distribution by Dimension:")
    print("-" * 80)
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    
    for dim_idx, dim_name in enumerate(['Width (X)', 'Height (Y)', 'Depth (Z)']):
        dim_values = resampled_shapes[:, dim_idx]
        print(f"\n{dim_name}:")
        print("  Percentile | Size")
        print("  " + "-" * 30)
        for p in percentiles:
            value = np.percentile(dim_values, p)
            print(f"  P{p:2d}         | {value:7.1f}")
    
    print()
    
    # Recommended target sizes
    target_size_p90 = np.percentile(resampled_shapes, 90, axis=0).astype(int)
    target_size_p95 = np.percentile(resampled_shapes, 95, axis=0).astype(int)
    target_size_max = np.max(resampled_shapes, axis=0).astype(int)
    
    print("Recommended target size options:")
    print(f"  Option 1 (90th percentile): {target_size_p90} - covers 90% of data")
    print(f"  Option 2 (95th percentile): {target_size_p95} - covers 95% of data")
    print(f"  Option 3 (maximum): {target_size_max} - covers 100% of data")
    print()
    
    # Calculate memory usage for different target sizes
    print("Memory usage estimate (batch_size=1, float32):")
    for name, size in [("Option 1", target_size_p90), ("Option 2", target_size_p95), ("Option 3", target_size_max)]:
        voxels = np.prod(size)
        memory_mb = voxels * 4 / 1024 / 1024  # 4 bytes per float32
        print(f"  {name} {size}: ~{memory_mb:.2f} MB/sample")
    print()
    
    return {
        'target_spacing': target_spacing,
        'resampled_shapes': resampled_shapes,
        'target_size_p90': target_size_p90,
        'target_size_p95': target_size_p95,
        'target_size_max': target_size_max,
    }


def analyze_nifti_dataset(data_dir, custom_target_spacings=None, max_files=None, sample_intensity=True, intensity_sample_ratio=0.1):
    """
    Analyze statistics of NIfTI dataset
    
    Args:
        data_dir: Data directory path
        custom_target_spacings: Custom target spacing list, e.g., [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]
        max_files: Maximum number of files to process (None for all files)
        sample_intensity: Whether to sample intensity values instead of storing all (saves memory)
        intensity_sample_ratio: Ratio of voxels to sample for intensity analysis (0.1 = 10%)
    """
    # Get all NIfTI files (recursively search subdirectories)
    nii_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.nii.gz"), recursive=True))
    
    if not nii_files:
        print(f"No .nii.gz files found in directory: {data_dir}")
        return
    
    # Limit number of files if specified
    if max_files is not None and len(nii_files) > max_files:
        print(f"Limiting to first {max_files} files (out of {len(nii_files)} total)")
        nii_files = nii_files[:max_files]
    
    print("="*80)
    print(f"Dataset Analysis Report")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Number of files: {len(nii_files)}\n")
    
    # Store statistics
    shapes = []
    spacings = []
    orientations = []
    affines = []
    
    # Store intensity information (categorized by modality)
    intensity_data = {
        'T1': [],
        'T2': [],
        'T2FS': [],
        'all': []  # All modalities
    }
    
    # Store cord mask bounding box information for ROI cropping analysis
    cord_bbox_sizes = []  # List of (bbox_size_x, bbox_size_y, bbox_size_z) for each cord mask
    cord_bbox_starts = []  # List of (start_x, start_y, start_z) for each cord mask
    cord_bbox_ends = []  # List of (end_x, end_y, end_z) for each cord mask
    cord_mask_shapes = []  # Original image shapes for cord masks
    cord_mask_spacings = []  # Spacings for cord masks
    
    print("Analyzing files...")
    for i, file_path in enumerate(nii_files, 1):
        try:
            nii = nib.load(file_path)
            
            # Get image shape
            shape = nii.shape
            shapes.append(shape)
            
            # Get spacing (voxel size)
            spacing = nii.header.get_zooms()[:3]  # Only take first 3 dimensions
            spacings.append(spacing)
            
            # Get affine
            affine = nii.affine
            affines.append(affine)
            
            # Get orientation
            orientation = nib.aff2axcodes(affine)
            orientations.append(orientation)
            
            # Read image data for intensity analysis (only for image files, not masks)
            filename_lower = os.path.basename(file_path).lower()
            is_image_file = any(mod in filename_lower for mod in ['t1', 't2', 't2fs']) and 'mask' not in filename_lower
            is_cord_mask = 'cord_mask' in filename_lower or 'cord' in filename_lower and 'mask' in filename_lower
            
            if is_image_file:
                try:
                    # Use dataobj instead of get_fdata() for memory efficiency
                    # dataobj is a memory-mapped array that doesn't load everything at once
                    dataobj = nii.dataobj
                    shape = nii.shape
                    
                    # For large images, sample slices instead of loading entire volume
                    # This is much more memory efficient
                    if len(shape) == 3:
                        # Sample middle slices (e.g., every 5th slice or 3-5 slices from middle)
                        if shape[2] > 10:
                            # Sample 5 slices from different parts of the volume
                            slice_indices = [
                                shape[2] // 4,
                                shape[2] // 2 - 1,
                                shape[2] // 2,
                                shape[2] // 2 + 1,
                                3 * shape[2] // 4
                            ]
                        else:
                            # Use all slices if volume is small
                            slice_indices = list(range(shape[2]))
                        
                        # Extract sampled slices
                        sampled_data = []
                        for z_idx in slice_indices:
                            slice_data = np.array(dataobj[:, :, z_idx])
                            sampled_data.append(slice_data)
                        
                        # Combine sampled slices
                        data = np.concatenate([s.flatten() for s in sampled_data])
                    else:
                        # For 2D or other formats, use dataobj directly but convert to array
                        data = np.array(dataobj).flatten()
                    
                    # Only consider nonzero voxels (exclude background)
                    nonzero_data = data[data != 0]
                    
                    if len(nonzero_data) > 0:
                        # Further sampling if still too large (for very large datasets)
                        if sample_intensity and len(nonzero_data) > 50000:
                            sample_size = max(5000, int(len(nonzero_data) * intensity_sample_ratio))
                            nonzero_data = np.random.choice(nonzero_data, size=sample_size, replace=False)
                        
                        # Identify modality based on filename
                        if 't1' in filename_lower and 't2' not in filename_lower:
                            intensity_data['T1'].extend(nonzero_data)
                        elif 't2fs' in filename_lower or 't2-fs' in filename_lower:
                            intensity_data['T2FS'].extend(nonzero_data)
                        elif 't2' in filename_lower:
                            intensity_data['T2'].extend(nonzero_data)
                        
                        # Add to total data
                        intensity_data['all'].extend(nonzero_data)
                        
                        # Clear local variables to free memory
                        del data, nonzero_data, sampled_data
                        
                except Exception as e:
                    # Skip intensity analysis if memory error or other issues
                    if i % 100 == 0:
                        print(f"  Warning: Skipped intensity analysis for {os.path.basename(file_path)}: {str(e)[:50]}")
            
            # Analyze cord mask bounding box for ROI cropping
            if is_cord_mask:
                try:
                    # Use memory-efficient approach: process slice by slice for large volumes
                    # This avoids loading entire volume into memory at once
                    if shape[2] > 20:  # For large volumes, process in chunks
                        # Process every Nth slice to find bbox (faster and more memory efficient)
                        step = max(1, shape[2] // 20)  # Sample up to 20 slices
                        slice_indices = list(range(0, shape[2], step))
                        if slice_indices[-1] != shape[2] - 1:
                            slice_indices.append(shape[2] - 1)
                    else:
                        # For small volumes, process all slices
                        slice_indices = list(range(shape[2]))
                    
                    # Find bbox by processing slices
                    min_coords = None
                    max_coords = None
                    dataobj = nii.dataobj
                    
                    for z_idx in slice_indices:
                        slice_data = np.array(dataobj[:, :, z_idx])
                        coords_2d = np.where(slice_data > 0)
                        
                        if len(coords_2d[0]) > 0:
                            if min_coords is None:
                                min_coords = np.array([coords_2d[0].min(), coords_2d[1].min(), z_idx])
                                max_coords = np.array([coords_2d[0].max(), coords_2d[1].max(), z_idx])
                            else:
                                min_coords[0] = min(min_coords[0], coords_2d[0].min())
                                min_coords[1] = min(min_coords[1], coords_2d[1].min())
                                min_coords[2] = min(min_coords[2], z_idx)
                                max_coords[0] = max(max_coords[0], coords_2d[0].max())
                                max_coords[1] = max(max_coords[1], coords_2d[1].max())
                                max_coords[2] = max(max_coords[2], z_idx)
                        
                        del slice_data, coords_2d
                    
                    if min_coords is not None and max_coords is not None:
                        # Calculate bounding box size
                        bbox_size = max_coords - min_coords + 1  # +1 because coordinates are inclusive
                        
                        # Store information
                        cord_bbox_sizes.append(bbox_size)
                        cord_bbox_starts.append(min_coords)
                        cord_bbox_ends.append(max_coords)
                        cord_mask_shapes.append(shape)
                        cord_mask_spacings.append(spacing)
                    
                    # Clear memory
                    del dataobj, min_coords, max_coords
                    
                except Exception as e:
                    if i % 100 == 0:
                        print(f"  Warning: Skipped cord mask analysis for {os.path.basename(file_path)}: {str(e)[:50]}")
            
            if i % 20 == 0:
                print(f"  Processed: {i}/{len(nii_files)}")
                
        except Exception as e:
            print(f"  ✗ Failed to read: {os.path.basename(file_path)}")
            print(f"    Error: {str(e)}")
    
    print(f"  Processed: {len(nii_files)}/{len(nii_files)}\n")
    
    # Convert to numpy arrays for statistics
    shapes = np.array(shapes)
    spacings = np.array(spacings)
    
    # ========== 1. Shape Statistics ==========
    print("="*80)
    print("1. Image Size (Shape) Statistics")
    print("="*80)
    print(f"Dimensions: 3D\n")
    
    for dim_idx, dim_name in enumerate(['Width (X)', 'Height (Y)', 'Depth (Z)']):
        dim_values = shapes[:, dim_idx]
        print(f"{dim_name}:")
        print(f"  Min: {np.min(dim_values)}")
        print(f"  Max: {np.max(dim_values)}")
        print(f"  Mean: {np.mean(dim_values):.2f}")
        print(f"  Median: {np.median(dim_values):.2f}")
        print(f"  Std: {np.std(dim_values):.2f}")
        
        # Statistics of different size distributions
        unique_values, counts = np.unique(dim_values, return_counts=True)
        print(f"  Unique values: {len(unique_values)}")
        if len(unique_values) <= 10:
            print(f"  Distribution: {dict(zip(unique_values.astype(int), counts.astype(int)))}")
        else:
            print(f"  Top 5 most common values:")
            top5_idx = np.argsort(counts)[-5:][::-1]
            for idx in top5_idx:
                print(f"    {int(unique_values[idx])}: {int(counts[idx])} times")
        print()
    
    # ========== 2. Spacing Statistics ==========
    print("="*80)
    print("2. Voxel Spacing Statistics")
    print("="*80)
    
    for dim_idx, dim_name in enumerate(['X direction', 'Y direction', 'Z direction']):
        dim_values = spacings[:, dim_idx]
        print(f"{dim_name}:")
        print(f"  Min: {np.min(dim_values):.6f} mm")
        print(f"  Max: {np.max(dim_values):.6f} mm")
        print(f"  Mean: {np.mean(dim_values):.6f} mm")
        print(f"  Median: {np.median(dim_values):.6f} mm")
        print(f"  Std: {np.std(dim_values):.6f} mm")
        
        # Statistics of different spacing distributions
        unique_values, counts = np.unique(dim_values.round(6), return_counts=True)
        print(f"  Unique values: {len(unique_values)}")
        if len(unique_values) <= 10:
            print(f"  Distribution:")
            for val, cnt in zip(unique_values, counts):
                print(f"    {val:.6f} mm: {int(cnt)} times")
        else:
            print(f"  Top 5 most common values:")
            top5_idx = np.argsort(counts)[-5:][::-1]
            for idx in top5_idx:
                print(f"    {unique_values[idx]:.6f} mm: {int(counts[idx])} times")
        print()
    
    # ========== 3. Physical Size Statistics ==========
    print("="*80)
    print("3. Physical Size Statistics")
    print("="*80)
    print("Physical Size = Shape × Spacing\n")
    
    physical_sizes = shapes * spacings
    
    for dim_idx, dim_name in enumerate(['X direction', 'Y direction', 'Z direction']):
        dim_values = physical_sizes[:, dim_idx]
        print(f"{dim_name}:")
        print(f"  Min: {np.min(dim_values):.2f} mm")
        print(f"  Max: {np.max(dim_values):.2f} mm")
        print(f"  Mean: {np.mean(dim_values):.2f} mm")
        print(f"  Median: {np.median(dim_values):.2f} mm")
        print(f"  Std: {np.std(dim_values):.2f} mm")
        print()
    
    # ========== 4. Orientation Statistics ==========
    print("="*80)
    print("4. Image Orientation Statistics")
    print("="*80)
    
    orientation_counts = defaultdict(int)
    for orient in orientations:
        orientation_counts[orient] += 1
    
    print("Orientation distribution:")
    for orient, count in sorted(orientation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {''.join(orient)}: {count} files ({count/len(nii_files)*100:.1f}%)")
    print()
    
    # ========== 5. Intensity Range Statistics ==========
    print("="*80)
    print("5. Intensity Range Statistics")
    print("="*80)
    
    def analyze_intensity_range(intensity_values, modality_name):
        """Analyze intensity range for a single modality"""
        if len(intensity_values) == 0:
            print(f"{modality_name}: No data")
            return None
        
        values = np.array(intensity_values)
        
        print(f"\n{modality_name} modality:")
        print(f"  Voxel count: {len(values):,}")
        print(f"  Global min: {np.min(values):.2f}")
        print(f"  Global max: {np.max(values):.2f}")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std: {np.std(values):.2f}")
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n  Percentile distribution:")
        print(f"  Percentile | Intensity Value")
        print(f"  " + "-" * 30)
        for p in percentiles:
            value = np.percentile(values, p)
            print(f"  P{p:2d}         | {value:10.2f}")
        
        # Recommended range (using 1% and 99% percentiles to exclude outliers)
        recommended_min = np.percentile(values, 1)
        recommended_max = np.percentile(values, 99)
        
        print(f"\n  Recommended intensity range (1%-99% percentile):")
        print(f"    min: {recommended_min:.0f}")
        print(f"    max: {recommended_max:.0f}")
        
        # More conservative range (5%-95% percentile)
        conservative_min = np.percentile(values, 5)
        conservative_max = np.percentile(values, 95)
        
        print(f"\n  Conservative intensity range (5%-95% percentile):")
        print(f"    min: {conservative_min:.0f}")
        print(f"    max: {conservative_max:.0f}")
        
        return {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'recommended_min': recommended_min,
            'recommended_max': recommended_max,
            'conservative_min': conservative_min,
            'conservative_max': conservative_max,
        }
    
    intensity_stats = {}
    
    # Analyze each modality
    for modality in ['T1', 'T2', 'T2FS', 'all']:
        if len(intensity_data[modality]) > 0:
            stats = analyze_intensity_range(intensity_data[modality], modality)
            if stats:
                intensity_stats[modality] = stats
    
    # If only 'all' data is available, use 'all' statistics
    if 'all' in intensity_stats and len(intensity_stats) == 1:
        print("\nNote: Cannot distinguish modalities, using statistics from all data")
    
    print()
    
    # ========== 5.5. Cord Mask Bounding Box Analysis (for ROI Cropping) ==========
    if len(cord_bbox_sizes) > 0:
        print("="*80)
        print("5.5. Cord Mask Bounding Box Analysis (for ROI Cropping)")
        print("="*80)
        
        cord_bbox_sizes_arr = np.array(cord_bbox_sizes)
        cord_bbox_starts_arr = np.array(cord_bbox_starts)
        cord_bbox_ends_arr = np.array(cord_bbox_ends)
        cord_mask_shapes_arr = np.array(cord_mask_shapes)
        cord_mask_spacings_arr = np.array(cord_mask_spacings)
        
        print(f"\nAnalyzed {len(cord_bbox_sizes_arr)} cord masks\n")
        
        # Analyze bounding box size in voxels
        print("Bounding Box Size (in voxels):")
        print("-" * 80)
        for dim_idx, dim_name in enumerate(['X (Width)', 'Y (Height)', 'Z (Depth)']):
            dim_values = cord_bbox_sizes_arr[:, dim_idx]
            print(f"\n{dim_name}:")
            print(f"  Min: {np.min(dim_values)}")
            print(f"  Max: {np.max(dim_values)}")
            print(f"  Mean: {np.mean(dim_values):.2f}")
            print(f"  Median: {np.median(dim_values):.2f}")
            print(f"  Std: {np.std(dim_values):.2f}")
            
            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            print(f"  Percentiles:")
            for p in percentiles:
                value = np.percentile(dim_values, p)
                print(f"    P{p:2d}: {value:.1f}")
        
        # Analyze bounding box position (relative to image center)
        print("\n" + "-" * 80)
        print("Bounding Box Position Analysis:")
        print("-" * 80)
        
        # Calculate center of each image
        image_centers = cord_mask_shapes_arr / 2.0
        
        # Calculate bbox centers
        bbox_centers = (cord_bbox_starts_arr + cord_bbox_ends_arr) / 2.0
        
        # Calculate offset from image center
        offsets = bbox_centers - image_centers
        
        print("\nOffset from image center (in voxels):")
        for dim_idx, dim_name in enumerate(['X', 'Y', 'Z']):
            dim_offsets = offsets[:, dim_idx]
            print(f"\n{dim_name} direction:")
            print(f"  Mean offset: {np.mean(dim_offsets):.2f}")
            print(f"  Std offset: {np.std(dim_offsets):.2f}")
            print(f"  Min offset: {np.min(dim_offsets):.2f}")
            print(f"  Max offset: {np.max(dim_offsets):.2f}")
        
        # Analyze bounding box size in physical space (mm)
        print("\n" + "-" * 80)
        print("Bounding Box Size (in physical space, mm):")
        print("-" * 80)
        
        # Convert voxel sizes to physical sizes
        bbox_physical_sizes = cord_bbox_sizes_arr * cord_mask_spacings_arr
        
        for dim_idx, dim_name in enumerate(['X (Width)', 'Y (Height)', 'Z (Depth)']):
            dim_values = bbox_physical_sizes[:, dim_idx]
            print(f"\n{dim_name}:")
            print(f"  Min: {np.min(dim_values):.2f} mm")
            print(f"  Max: {np.max(dim_values):.2f} mm")
            print(f"  Mean: {np.mean(dim_values):.2f} mm")
            print(f"  Median: {np.median(dim_values):.2f} mm")
            print(f"  Std: {np.std(dim_values):.2f} mm")
            
            # Percentiles
            percentiles = [50, 75, 90, 95, 99]
            print(f"  Percentiles:")
            for p in percentiles:
                value = np.percentile(dim_values, p)
                print(f"    P{p:2d}: {value:.2f} mm")
        
        # ROI Cropping Recommendations
        print("\n" + "="*80)
        print("ROI Cropping Configuration Recommendations")
        print("="*80)
        
        # Calculate recommended target size based on percentiles
        p95_sizes = np.percentile(cord_bbox_sizes_arr, 95, axis=0)
        p99_sizes = np.percentile(cord_bbox_sizes_arr, 99, axis=0)
        max_sizes = np.max(cord_bbox_sizes_arr, axis=0)
        
        # Calculate recommended margin (based on typical spacing)
        median_spacing_cord = np.median(cord_mask_spacings_arr, axis=0)
        # Margin in mm: typically 10-20mm in X/Y, 5-10mm in Z
        margin_mm = np.array([15.0, 15.0, 5.0])  # X, Y, Z in mm
        margin_voxels = (margin_mm / median_spacing_cord).astype(int)
        
        print("\nRecommended roi_target_size (in voxels):")
        print(f"  Option 1 (95th percentile): [{int(p95_sizes[0])}, {int(p95_sizes[1])}, {int(p95_sizes[2])}]")
        print(f"    - Covers 95% of cord masks")
        print(f"  Option 2 (99th percentile): [{int(p99_sizes[0])}, {int(p99_sizes[1])}, {int(p99_sizes[2])}]")
        print(f"    - Covers 99% of cord masks")
        print(f"  Option 3 (maximum): [{int(max_sizes[0])}, {int(max_sizes[1])}, {int(max_sizes[2])}]")
        print(f"    - Covers 100% of cord masks (may be too large)")
        
        print("\nRecommended roi_margin (in voxels):")
        print(f"  Based on median spacing [{median_spacing_cord[0]:.3f}, {median_spacing_cord[1]:.3f}, {median_spacing_cord[2]:.3f}] mm")
        print(f"  Suggested margin: [{margin_voxels[0]}, {margin_voxels[1]}, {margin_voxels[2]}]")
        print(f"    - Physical size: [{margin_mm[0]:.1f}, {margin_mm[1]:.1f}, {margin_mm[2]:.1f}] mm")
        print(f"    - Note: Add this margin to bbox_size to get final roi_target_size")
        
        # Calculate final recommended size (bbox + margin)
        recommended_size_with_margin = p95_sizes + margin_voxels
        print(f"\nFinal recommended roi_target_size (bbox_95th + margin):")
        print(f"  [{int(recommended_size_with_margin[0])}, {int(recommended_size_with_margin[1])}, {int(recommended_size_with_margin[2])}]")
        
        # If target spacing is provided, calculate size at target spacing
        if custom_target_spacings is not None and len(custom_target_spacings) > 0:
            target_spacing = np.array(custom_target_spacings[0])
            print(f"\nAt target spacing [{target_spacing[0]:.3f}, {target_spacing[1]:.3f}, {target_spacing[2]:.3f}] mm:")
            # Convert physical size to voxels at target spacing
            bbox_physical_p95 = p95_sizes * median_spacing_cord
            margin_physical = margin_mm
            total_physical = bbox_physical_p95 + margin_physical
            size_at_target = (total_physical / target_spacing).astype(int)
            print(f"  Recommended roi_target_size: [{size_at_target[0]}, {size_at_target[1]}, {size_at_target[2]}]")
            print(f"  Recommended roi_margin: [{int(margin_physical[0]/target_spacing[0])}, {int(margin_physical[1]/target_spacing[1])}, {int(margin_physical[2]/target_spacing[2])}]")
        
        print()
    else:
        print("="*80)
        print("5.5. Cord Mask Bounding Box Analysis")
        print("="*80)
        print("\nNo cord mask files found. Skipping ROI cropping analysis.")
        print("Note: Cord mask files should contain 'cord_mask' or 'cord' and 'mask' in filename.\n")
    
    # ========== 6. Recommended Preprocessing ==========
    print("="*80)
    print("6. Recommended Preprocessing")
    print("="*80)
    
    # Calculate target spacing (using median)
    median_spacing = np.median(spacings, axis=0)
    print(f"Recommended target spacing (median): [{median_spacing[0]:.4f}, {median_spacing[1]:.4f}, {median_spacing[2]:.4f}] mm\n")
    
    # Analyze using median spacing
    median_stats = analyze_spacing_distribution(physical_sizes, median_spacing, len(nii_files))
    
    # ========== Custom Spacing Analysis ==========
    custom_stats_list = []
    if custom_target_spacings is not None and len(custom_target_spacings) > 0:
        print("\n" + "="*80)
        print("Custom Spacing Analysis")
        print("="*80)
        
        for custom_spacing in custom_target_spacings:
            custom_spacing_arr = np.array(custom_spacing)
            print()
            custom_stats = analyze_spacing_distribution(physical_sizes, custom_spacing_arr, len(nii_files))
            custom_stats_list.append(custom_stats)
    
    # Use median statistics
    target_spacing = median_spacing
    target_size_p90 = median_stats['target_size_p90']
    target_size_p95 = median_stats['target_size_p95']
    target_size_max = median_stats['target_size_max']
    
    # ========== 7. Complete Preprocessing Recommendations ==========
    print("="*80)
    print("7. Complete Preprocessing Recommendations")
    print("="*80)
    print("""
Option A: Preserve Original Physical Size (Recommended for Precise Medical Segmentation)
---------------------------------------------------------
1. Resample to uniform spacing: [{:.4f}, {:.4f}, {:.4f}] mm
2. Use uniform target size: {} (covers 95% of data)
3. For images smaller than target size: use padding (mode='constant', value=0)
4. For images larger than target size: use center crop or random crop
5. Interpolation methods:
   - Images: trilinear (order=1)
   - Labels: nearest (order=0)

Advantages: Maintains physical spatial consistency, suitable for medical images
Disadvantages: May require larger memory

Option B: Direct Resize (Fast but may lose information)
---------------------------------------------------------
1. Directly resize to fixed size, ignoring spacing
2. Target size: [64, 128, 128] or [96, 192, 192]
3. Interpolation methods:
   - Images: trilinear
   - Labels: nearest

Advantages: Simple implementation, controllable memory usage
Disadvantages: Breaks physical spatial relationships, may affect segmentation accuracy

Recommended: Use Option A with target configuration:
- target_spacing: [{:.4f}, {:.4f}, {:.4f}]
- target_size: {}
""".format(
        target_spacing[0], target_spacing[1], target_spacing[2],
        target_size_p95,
        target_spacing[0], target_spacing[1], target_spacing[2],
        target_size_p95
    ))
    
    print("="*80)
    print("Analysis complete!")
    print("="*80)
    
    # Output intensity range configuration recommendations
    if intensity_stats:
        print("\n" + "="*80)
        print("Intensity Range Configuration Recommendations")
        print("="*80)
        
        # Prefer T2 statistics (because config notes "based on T2 modality")
        if 'T2' in intensity_stats:
            stats = intensity_stats['T2']
            print("\nRecommended configuration based on T2 modality:")
            print(f"intensity_range:")
            print(f"  min: {stats['recommended_min']:.0f}  # 1% percentile")
            print(f"  max: {stats['recommended_max']:.0f}  # 99% percentile")
            print(f"\nOr use a more conservative range:")
            print(f"intensity_range:")
            print(f"  min: {stats['conservative_min']:.0f}  # 5% percentile")
            print(f"  max: {stats['conservative_max']:.0f}  # 95% percentile")
        elif 'all' in intensity_stats:
            stats = intensity_stats['all']
            print("\nRecommended configuration based on all data:")
            print(f"intensity_range:")
            print(f"  min: {stats['recommended_min']:.0f}  # 1% percentile")
            print(f"  max: {stats['recommended_max']:.0f}  # 99% percentile")
        
        print()
    
    return {
        'shapes': shapes,
        'spacings': spacings,
        'physical_sizes': physical_sizes,
        'median_spacing': median_spacing,
        'median_stats': median_stats,
        'custom_stats_list': custom_stats_list,
        'target_spacing': target_spacing,
        'target_size_p90': target_size_p90,
        'target_size_p95': target_size_p95,
        'target_size_max': target_size_max,
        'intensity_stats': intensity_stats,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze NIfTI dataset statistics")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory path (default: ./spinal_cord_dataset/raw)"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all files, use this to limit memory usage)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable intensity sampling (uses more memory but more accurate)"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="Ratio of voxels to sample for intensity analysis (default: 0.1 = 10%%)"
    )
    
    args = parser.parse_args()
    
    # Data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Use absolute path or relative path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, "spinal_cord_dataset", "raw")
    
    # Custom target spacing list (can add multiple)
    # Example: [[0.54, 0.54, 3.3], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]
    custom_spacings = [
        [0.54, 0.54, 3.3],  # Target spacing from config (matches multitask_multimodal.yaml)
        # [1.0, 1.0, 1.0],  # Isotropic 1mm (optional, for comparison)
        # [0.5, 0.5, 0.5],  # Isotropic 0.5mm (optional, for comparison)
    ]
    
    print(f"Configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Max files: {args.max_files if args.max_files else 'All files'}")
    print(f"  Intensity sampling: {not args.no_sample}")
    if not args.no_sample:
        print(f"  Sample ratio: {args.sample_ratio*100:.1f}%")
    print()
    
    # Execute analysis
    stats = analyze_nifti_dataset(
        data_dir,
        custom_target_spacings=custom_spacings,
        max_files=args.max_files,
        sample_intensity=not args.no_sample,
        intensity_sample_ratio=args.sample_ratio
    )

