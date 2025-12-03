"""
Module for converting color-reduced images to paint-by-numbers templates.
"""

import random
from typing import Tuple, Set, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _flood_fill_region(
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


def _create_border_image(regions: list[Set[Tuple[int, int]]], width: int, height: int) -> np.ndarray:
    """
    Create a binary image with borders between regions.

    Args:
        regions: List of sets, each containing (x, y) coordinates of a region
        width: Image width
        height: Image height

    Returns:
        2D numpy array with 1 for borders, 0 for interior
    """
    # Create region map
    region_map = np.full((height, width), -1, dtype=np.int32)
    for region_id, region in enumerate(regions):
        for x, y in region:
            region_map[y, x] = region_id

    # Find borders
    border = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            current_region = region_map[y, x]
            # Check neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if region_map[ny, nx] != current_region:
                        border[y, x] = 1
                        break

    return border


def _find_label_center(region: Set[Tuple[int, int]], border: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    """
    Find the best position to place a label in a region (as central as possible).

    Args:
        region: Set of (x, y) coordinates in the region
        border: 2D array with 1 for borders
        width: Image width
        height: Image height

    Returns:
        (x, y) coordinate for label placement
    """
    # Filter out border pixels
    interior_pixels = [(x, y) for x, y in region if border[y, x] == 0]

    if not interior_pixels:
        # If no interior pixels, use any pixel from region
        interior_pixels = list(region)

    if not interior_pixels:
        return (0, 0)

    # Find the most central point using distance transform approach
    # Calculate the centroid first
    centroid_x = sum(x for x, y in interior_pixels) // len(interior_pixels)
    centroid_y = sum(y for x, y in interior_pixels) // len(interior_pixels)

    # Find the closest interior pixel to the centroid
    best_pixel = min(interior_pixels,
                    key=lambda p: (p[0] - centroid_x)**2 + (p[1] - centroid_y)**2)

    return best_pixel


def _merge_small_regions(
    regions: list[Set[Tuple[int, int]]],
    label_image: np.ndarray,
    width: int,
    height: int,
    min_region_size: int
) -> list[Set[Tuple[int, int]]]:
    """
    Merge small regions with their neighbors to create larger paintable areas.

    Args:
        regions: List of region sets
        label_image: 2D array of color labels
        width: Image width
        height: Image height
        min_region_size: Minimum size for a region (smaller ones will be merged)

    Returns:
        List of merged regions
    """
    # Sort regions by size (smallest first for merging)
    regions = sorted(regions, key=len)

    # Create a region map for quick neighbor lookup
    region_map = {}
    for idx, region in enumerate(regions):
        for pixel in region:
            region_map[pixel] = idx

    merged = [False] * len(regions)

    for idx in range(len(regions)):
        if merged[idx] or len(regions[idx]) >= min_region_size:
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

    # Return only non-merged regions
    return [regions[i] for i in range(len(regions)) if not merged[i]]


def generate_image_to_paint_by_numbers(image: Image.Image, min_region_size: int = 100) -> Image.Image:
    """
    Convert a color-reduced image to a B&W paint-by-numbers template.

    The function:
    1. Segments the image into regions of the same color
    2. Merges small regions to create larger paintable areas
    3. Creates a color-to-label mapping
    4. Draws borders between regions
    5. Places numeric labels in each region

    Args:
        image: Input PIL Image with reduced color palette
        min_region_size: Minimum size for regions (smaller ones get merged)

    Returns:
        PIL Image in RGB mode with black borders and color numbers on white background
    """
    width, height = image.size
    pixels = image.load()

    # Create label image mapping each unique color to a number
    unique_colors = {}
    label_image = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            color = pixels[x, y]
            if color not in unique_colors:
                unique_colors[color] = len(unique_colors)
            label_image[y, x] = unique_colors[color]

    # Segment image into regions using flood fill
    visited = np.zeros((height, width), dtype=bool)
    regions = []

    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                region = _flood_fill_region(label_image, x, y, visited, width, height)
                if region:
                    regions.append(region)

    # Merge small regions to create larger paintable areas
    regions = _merge_small_regions(regions, label_image, width, height, min_region_size)

    # Create border image
    border = _create_border_image(regions, width, height)

    # Create output image (white background)
    output = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(output)

    # Draw borders in black
    for y in range(height):
        for x in range(width):
            if border[y, x] == 1:
                draw.point((x, y), fill='black')

    # Try to load a font with appropriate size based on image dimensions
    font_size = max(8, min(width, height) // 50)  # Dynamic font size based on image size
    try:
        # Try different font paths
        font = None
        for font_name in ['arial.ttf', 'Arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
                        '/System/Library/Fonts/Supplemental/Arial.ttf',
                        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except:
                continue
        if not font:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # Place labels in regions
    for region in regions:
        if not region:
            continue

        # Get label for this region
        sample_x, sample_y = next(iter(region))
        label = label_image[sample_y, sample_x]

        # Find good position for label (centered in the region)
        label_x, label_y = _find_label_center(region, border, width, height)

        # Draw the label centered on the position
        text = str(label)
        # Get text bounding box for better centering
        bbox = draw.textbbox((label_x, label_y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text on the calculated position
        final_x = label_x - text_width // 2
        final_y = label_y - text_height // 2

        draw.text((final_x, final_y), text, font=font, fill='black')

    return output
