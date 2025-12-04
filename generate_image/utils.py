"""
Utility functions shared across image generation modules.
"""

from typing import Tuple, Set, Dict
import numpy as np
from PIL import Image


def flood_fill_region(
    label_image: np.ndarray,
    start_x: int,
    start_y: int,
    visited: np.ndarray,
    width: int,
    height: int
) -> Set[Tuple[int, int]]:
    """
    Use flood fill to find all connected pixels of the same color.

    Args:
        label_image: 2D array of color labels
        start_x: Starting x coordinate
        start_y: Starting y coordinate
        visited: 2D array tracking visited pixels
        width: Image width
        height: Image height

    Returns:
        Set of (x, y) coordinates belonging to this region
    """
    region = set()
    stack = [(start_x, start_y)]
    target_label = label_image[start_y, start_x]

    while stack:
        x, y = stack.pop()

        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if visited[y, x]:
            continue
        if label_image[y, x] != target_label:
            continue

        visited[y, x] = True
        region.add((x, y))

        # Add neighbors
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return region


def merge_small_regions(
    image: Image.Image,
    min_region_size_in_pixels: int
) -> Image.Image:
    """
    Merge small regions with their neighbors to create larger paintable areas.

    Args:
        image: Input PIL Image in RGB mode
        min_region_size_in_pixels: Minimum size for a region (smaller ones will be merged)

    Returns:
        PIL Image with small regions merged
    """
    width, height = image.size
    pixels = image.load()

    # Create label image mapping each unique color to a number
    print(f"  Segmenting into regions for merging...")
    unique_colors = {}
    label_image = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            color = pixels[x, y]
            if color not in unique_colors:
                unique_colors[color] = len(unique_colors)
            label_image[y, x] = unique_colors[color]

    # Map labels back to colors
    label_to_color = {label: color for color, label in unique_colors.items()}

    # Segment image into regions using flood fill
    visited = np.zeros((height, width), dtype=bool)
    regions = []

    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                region = flood_fill_region(label_image, x, y, visited, width, height)
                if region:
                    regions.append(region)

    print(f"  Found {len(regions)} initial regions")

    # Sort regions by size (smallest first for merging)
    regions = sorted(regions, key=len)

    # Create a region map for quick neighbor lookup
    region_map = {}
    for idx, region in enumerate(regions):
        for pixel in region:
            region_map[pixel] = idx

    merged = [False] * len(regions)

    print(f"  Merging small regions (min size: {min_region_size_in_pixels} pixels)...")
    for idx in range(len(regions)):
        if merged[idx] or len(regions[idx]) >= min_region_size_in_pixels:
            continue

        # Find neighboring regions
        neighbors = set()
        for x, y in regions[idx]:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_pixel = (nx, ny)
                    if neighbor_pixel in region_map:
                        neighbor_idx = region_map[neighbor_pixel]
                        if neighbor_idx != idx and not merged[neighbor_idx]:
                            neighbors.add(neighbor_idx)

        if neighbors:
            # Merge with the largest neighbor
            target_idx = max(neighbors, key=lambda i: len(regions[i]))
            regions[target_idx] = regions[target_idx].union(regions[idx])
            merged[idx] = True

            # Update region map
            for pixel in regions[idx]:
                region_map[pixel] = target_idx

    # Keep only non-merged regions
    final_regions = [regions[i] for i in range(len(regions)) if not merged[i]]
    print(f"  After merging: {len(final_regions)} regions")

    # Update image pixels to reflect merged regions
    print(f"  Applying merged regions to image...")
    for region in final_regions:
        # Get the color for this region (use the label from any pixel in the region)
        sample_x, sample_y = next(iter(region))
        region_label = label_image[sample_y, sample_x]
        region_color = label_to_color[region_label]

        # Set all pixels in this region to the same color
        for x, y in region:
            pixels[x, y] = region_color

    return image
