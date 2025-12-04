"""
Module for converting images to use only specific allowed colors.
"""

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

from parameters_reader import ParametersImageSizeInMm
from generate_image.utils import merge_small_regions


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        RGB tuple (R, G, B)
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Calculate Euclidean distance between two RGB colors.

    Args:
        c1: First color as (R, G, B) tuple
        c2: Second color as (R, G, B) tuple

    Returns:
        Squared Euclidean distance between the colors
    """
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2


def _find_nearest_color(color: Tuple[int, int, int], palette: list[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    """
    Find the nearest color in the palette to the given color.

    Args:
        color: RGB color tuple to match
        palette: List of RGB color tuples to search

    Returns:
        The nearest color from the palette
    """
    return min(palette, key=lambda c: _color_distance(c, color))


def generate_image_in_the_specific_colors(
    image: Image.Image,
    allowed_colors: list[str],
    image_size_in_mm: ParametersImageSizeInMm,
    min_region_size_in_mm: int
) -> Image.Image:
    """
    Convert the image to use only colors from the allowed colors list.

    The function:
    1. Resizes the image to the specified size in mm (assuming 3 pixels per mm)
    2. Converts hex color strings to RGB tuples
    3. Maps each pixel to the nearest allowed color
    4. Applies median filtering to smooth the result

    Args:
        image: Input PIL Image in RGB mode
        allowed_colors: List of hex color strings (e.g., ["#FF0000", "#00FF00"])
        image_size_in_mm: Target image size in millimeters
        min_region_size_in_mm: Minimum region size in millimeters (used for filter size)

    Returns:
        PIL Image with colors mapped to the allowed palette
    """
    # Resize image to specified dimensions (2 pixels per mm for manageable file size)
    # Note: For very large canvases, using 2 pixels/mm provides good quality while keeping memory usage reasonable
    PIXELS_PER_MM = 2
    target_width = image_size_in_mm.width * PIXELS_PER_MM
    target_height = image_size_in_mm.height * PIXELS_PER_MM
    print(f"  Resizing image to {target_width} x {target_height} pixels ({PIXELS_PER_MM} pixels/mm)")
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert hex colors to RGB
    palette = np.array([_hex_to_rgb(color) for color in allowed_colors])
    print(f"  Converting to {len(palette)} allowed colors...")

    # Convert image to numpy array for faster processing
    print(f"  Mapping {image.width * image.height:,} pixels to nearest colors...")
    img_array = np.array(image)

    # Process in batches to avoid memory issues with large images
    pixels_flat = img_array.reshape(-1, 3)
    batch_size = 100000  # Process 100k pixels at a time
    nearest_indices = np.zeros(len(pixels_flat), dtype=np.int32)

    for i in range(0, len(pixels_flat), batch_size):
        end_idx = min(i + batch_size, len(pixels_flat))
        batch = pixels_flat[i:end_idx]

        # Calculate distances to all palette colors for this batch
        distances = np.sum((batch[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2)
        nearest_indices[i:end_idx] = np.argmin(distances, axis=1)

        if (i // batch_size) % 10 == 0:
            progress = 100 * end_idx // len(pixels_flat)
            print(f"    Progress: {progress}%")

    # Map pixels to nearest colors
    img_array = palette[nearest_indices].reshape(img_array.shape).astype(np.uint8)
    image = Image.fromarray(img_array, 'RGB')

    # Calculate filter size based on min_region_size_in_mm
    filter_size = max(3, int(min_region_size_in_mm * PIXELS_PER_MM * 0.03))
    if filter_size % 2 == 0:
        filter_size += 1  # MedianFilter requires odd size

    # Apply median filter to smooth colors and reduce noise
    print(f"  Applying median filter (size={filter_size}) to smooth colors...")
    median_filter = ImageFilter.MedianFilter(size=filter_size)
    image = image.filter(median_filter)

    # Re-apply palette mapping after filtering
    print(f"  Re-mapping pixels after filtering...")
    img_array = np.array(image)
    pixels_flat = img_array.reshape(-1, 3)
    nearest_indices = np.zeros(len(pixels_flat), dtype=np.int32)

    for i in range(0, len(pixels_flat), batch_size):
        end_idx = min(i + batch_size, len(pixels_flat))
        batch = pixels_flat[i:end_idx]
        distances = np.sum((batch[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2)
        nearest_indices[i:end_idx] = np.argmin(distances, axis=1)

        if (i // batch_size) % 10 == 0:
            progress = 100 * end_idx // len(pixels_flat)
            print(f"    Progress: {progress}%")

    img_array = palette[nearest_indices].reshape(img_array.shape).astype(np.uint8)
    image = Image.fromarray(img_array, 'RGB')

    print(f"  Color conversion complete!")

    # Merge small regions to create larger paintable areas
    min_region_size_in_pixels = int(min_region_size_in_mm * PIXELS_PER_MM)
    image = merge_small_regions(image, min_region_size_in_pixels)

    return image
