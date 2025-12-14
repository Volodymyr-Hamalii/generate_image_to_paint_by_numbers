"""
Module for converting images to optimal colors using k-means clustering.
"""

import random

import numpy as np
from PIL import Image, ImageFilter

from utils.logger import logger
from parameters_reader import ParametersImageSizeInMm
from generate_image.utils import merge_small_regions


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Convert RGB to LAB color space for perceptually uniform color comparison.

    Args:
        rgb: RGB tuple (R, G, B) with values 0-255

    Returns:
        LAB tuple (L, a, b)
    """
    # Normalize RGB to 0-1
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0

    # Convert to linear RGB
    def srgb_to_linear(c):
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4

    r_linear = srgb_to_linear(r)
    g_linear = srgb_to_linear(g)
    b_linear = srgb_to_linear(b)

    # Convert to XYZ using sRGB D65 conversion matrix
    x = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
    y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750
    z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041

    # Normalize by D65 white point
    x = x / 0.95047
    y = y / 1.00000
    z = z / 1.08883

    # Convert to LAB
    def f(t):
        delta = 6.0 / 29.0
        if t > delta ** 3:
            return t ** (1.0 / 3.0)
        return t / (3 * delta ** 2) + 4.0 / 29.0

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return (L, a, b)


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """
    Calculate perceptual distance between two RGB colors using LAB color space (deltaE).

    Args:
        c1: First color as (R, G, B) tuple
        c2: Second color as (R, G, B) tuple

    Returns:
        Perceptual distance between the colors (deltaE)
    """
    lab1 = _rgb_to_lab(c1)
    lab2 = _rgb_to_lab(c2)

    # Calculate Euclidean distance in LAB space (deltaE)
    return ((lab1[0] - lab2[0]) ** 2 +
            (lab1[1] - lab2[1]) ** 2 +
            (lab1[2] - lab2[2]) ** 2) ** 0.5


def _mean_color(colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """
    Calculate the mean color from a list of colors.

    Args:
        colors: List of RGB color tuples

    Returns:
        Mean color as (R, G, B) tuple
    """
    n = len(colors)
    if n == 0:
        return (0, 0, 0)
    r = sum(c[0] for c in colors) // n
    g = sum(c[1] for c in colors) // n
    b = sum(c[2] for c in colors) // n
    return (r, g, b)


def _find_nearest_color(color: tuple[int, int, int], palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """
    Find the nearest color in the palette to the given color.

    Args:
        color: RGB color tuple to match
        palette: List of RGB color tuples to search

    Returns:
        The nearest color from the palette
    """
    return min(palette, key=lambda c: _color_distance(c, color))


def _k_means_cluster(colors: list[tuple[int, int, int]], k: int, max_samples: int = 10000, max_iterations: int = 10) -> list[tuple[int, int, int]]:
    """
    Perform k-means clustering on colors to find k representative colors.

    Args:
        colors: List of all RGB color tuples
        k: Number of clusters to create
        max_samples: Maximum number of color samples to use for clustering
        max_iterations: Maximum number of clustering iterations

    Returns:
        List of k cluster centroids as RGB tuples
    """
    # Sample colors if there are too many
    if len(colors) > max_samples:
        colors = random.sample(colors, max_samples)

    # Initialize centroids with random colors
    centroids = random.sample(colors, k)
    iteration = 0
    old_centroids = None

    while iteration < max_iterations and centroids != old_centroids:
        old_centroids = centroids
        iteration += 1

        # Assign each color to nearest centroid
        labels = [_find_nearest_color(c, centroids) for c in colors]

        # Update centroids as mean of assigned colors
        new_centroids = []
        for centroid in centroids:
            assigned_colors = [c for c, label in zip(colors, labels) if label == centroid]
            if assigned_colors:
                new_centroids.append(_mean_color(assigned_colors))
            else:
                new_centroids.append(centroid)
        centroids = new_centroids

    return centroids


def generate_image_in_optimal_colors(
    image: Image.Image,
    max_number_of_colors: int,
    image_size_in_mm: ParametersImageSizeInMm,
    min_region_size_in_mm: int
) -> Image.Image:
    """
    Convert the image to use optimal colors found through k-means clustering.

    The function:
    1. Resizes the image to the specified size in mm (assuming 3 pixels per mm)
    2. Extracts all unique colors from the image
    3. Uses k-means clustering to find the most representative colors
    4. Maps each pixel to the nearest cluster color
    5. Applies median filtering to smooth the result

    Args:
        image: Input PIL Image in RGB mode
        max_number_of_colors: Maximum number of colors to use in the output
        image_size_in_mm: Target image size in millimeters
        min_region_size_in_mm: Minimum region size in millimeters (used for filter size)

    Returns:
        PIL Image with reduced color palette
    """
    # Resize image to specified dimensions (2 pixels per mm for manageable file size)
    PIXELS_PER_MM = 2
    width_in_mm, height_in_mm = image_size_in_mm.get_dimensions(image.width, image.height)
    target_width = width_in_mm * PIXELS_PER_MM
    target_height = height_in_mm * PIXELS_PER_MM
    logger.info(f"  Resizing image to {target_width} x {target_height} pixels ({PIXELS_PER_MM} pixels/mm, {width_in_mm} x {height_in_mm} mm)")
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Convert image to numpy array for faster processing
    logger.info(f"  Collecting colors from {image.width * image.height:,} pixels...")
    img_array = np.array(image)
    pixels_flat = img_array.reshape(-1, 3)
    all_colors = [tuple(p) for p in pixels_flat.tolist()]

    # Find optimal palette using k-means clustering
    logger.info(f"  Finding {max_number_of_colors} optimal colors using k-means clustering...")
    palette = _k_means_cluster(all_colors, max_number_of_colors)
    palette_array = np.array(palette)
    logger.info(f"  Found {len(palette)} colors")

    # Map each pixel to nearest palette color using batch processing
    logger.info(f"  Mapping pixels to nearest colors...")
    batch_size = 100000  # Process 100k pixels at a time
    nearest_indices = np.zeros(len(pixels_flat), dtype=np.int32)

    for i in range(0, len(pixels_flat), batch_size):
        end_idx = min(i + batch_size, len(pixels_flat))
        batch = pixels_flat[i:end_idx]
        distances = np.sum((batch[:, np.newaxis, :] - palette_array[np.newaxis, :, :]) ** 2, axis=2)
        nearest_indices[i:end_idx] = np.argmin(distances, axis=1)

        if (i // batch_size) % 10 == 0:
            progress = 100 * end_idx // len(pixels_flat)
            logger.info(f"    Progress: {progress}%")

    img_array = palette_array[nearest_indices].reshape(img_array.shape).astype(np.uint8)
    image = Image.fromarray(img_array, 'RGB')

    # Calculate filter size based on min_region_size_in_mm
    filter_size = max(3, int(min_region_size_in_mm * PIXELS_PER_MM * 0.03))
    if filter_size % 2 == 0:
        filter_size += 1  # MedianFilter requires odd size

    # Apply median filter to smooth colors and reduce noise
    logger.info(f"  Applying median filter (size={filter_size}) to smooth colors...")
    median_filter = ImageFilter.MedianFilter(size=filter_size)
    image = image.filter(median_filter)

    # Re-apply palette mapping after filtering
    logger.info(f"  Re-mapping pixels after filtering...")
    img_array = np.array(image)
    pixels_flat = img_array.reshape(-1, 3)
    nearest_indices = np.zeros(len(pixels_flat), dtype=np.int32)

    for i in range(0, len(pixels_flat), batch_size):
        end_idx = min(i + batch_size, len(pixels_flat))
        batch = pixels_flat[i:end_idx]
        distances = np.sum((batch[:, np.newaxis, :] - palette_array[np.newaxis, :, :]) ** 2, axis=2)
        nearest_indices[i:end_idx] = np.argmin(distances, axis=1)

        if (i // batch_size) % 10 == 0:
            progress = 100 * end_idx // len(pixels_flat)
            logger.info(f"    Progress: {progress}%")

    img_array = palette_array[nearest_indices].reshape(img_array.shape).astype(np.uint8)
    image = Image.fromarray(img_array, 'RGB')

    logger.info(f"  Color conversion complete!")

    # Merge small regions to create larger paintable areas
    min_region_size_in_pixels = int(min_region_size_in_mm * PIXELS_PER_MM)
    image = merge_small_regions(image, min_region_size_in_pixels)

    return image
