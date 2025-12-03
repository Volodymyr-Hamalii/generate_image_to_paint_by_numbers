"""
Module for converting images to optimal colors using k-means clustering.
"""

import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter


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


def _mean_color(colors: list[Tuple[int, int, int]]) -> Tuple[int, int, int]:
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


def _k_means_cluster(colors: list[Tuple[int, int, int]], k: int, max_samples: int = 10000, max_iterations: int = 10) -> list[Tuple[int, int, int]]:
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


def generate_image_in_optimal_colors(image: Image.Image, max_number_of_colors: int) -> Image.Image:
    """
    Convert the image to use optimal colors found through k-means clustering.

    The function:
    1. Extracts all unique colors from the image
    2. Uses k-means clustering to find the most representative colors
    3. Maps each pixel to the nearest cluster color
    4. Applies median filtering to smooth the result

    Args:
        image: Input PIL Image in RGB mode
        max_number_of_colors: Maximum number of colors to use in the output

    Returns:
        PIL Image with reduced color palette
    """
    # Get image size and pixels
    width, height = image.size
    pixels = image.load()

    # Collect all colors
    all_coords = [(x, y) for x in range(width) for y in range(height)]
    all_colors = [pixels[x, y] for x, y in all_coords]

    # Find optimal palette using k-means clustering
    palette = _k_means_cluster(all_colors, max_number_of_colors)

    # Map each pixel to nearest palette color
    for x, y in all_coords:
        pixels[x, y] = _find_nearest_color(pixels[x, y], palette)

    # Apply median filter to smooth colors and reduce noise
    median_filter = ImageFilter.MedianFilter(size=3)
    image = image.filter(median_filter)
    pixels = image.load()

    # Re-apply palette mapping after filtering
    for x, y in all_coords:
        pixels[x, y] = _find_nearest_color(pixels[x, y], palette)

    return image
