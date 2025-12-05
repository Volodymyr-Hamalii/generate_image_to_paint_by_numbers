"""
Module for converting color-reduced images to paint-by-numbers templates.
"""

import random
from typing import Tuple, Set, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parameters_reader import ParametersBorder, ParametersNumbers
from generate_image.utils import flood_fill_region


# Font size constraints in mm (2 pixels per mm to match the color conversion)
PIXELS_PER_MM = 2


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


def generate_image_to_paint_by_numbers(
    image: Image.Image,
    border_params: ParametersBorder,
    numbers_params: ParametersNumbers,
) -> Image.Image:
    """
    Convert a color-reduced image to a B&W paint-by-numbers template.

    The function:
    1. Segments the image into regions of the same color (regions are already merged from color processing)
    2. Creates a color-to-label mapping
    3. Draws borders between regions
    4. Places numeric labels in each region

    Args:
        image: Input PIL Image with reduced color palette (regions already merged)
        border_params: Border configuration (width in mm and color)
        numbers_params: Numbers configuration (color)

    Returns:
        PIL Image in RGB mode with borders and color numbers on white background
    """
    width, height = image.size
    pixels = image.load()

    if pixels is None:
        raise ValueError("Image pixels are None")


    min_font_size_pixels = int(numbers_params.font_size_in_mm.min * PIXELS_PER_MM)
    max_font_size_pixels = int(numbers_params.font_size_in_mm.max * PIXELS_PER_MM)

    # Create label image mapping each unique color to a number
    print(f"  Creating color label map...")
    unique_colors = {}
    label_image = np.zeros((height, width), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            color = pixels[x, y]
            if color not in unique_colors:
                unique_colors[color] = len(unique_colors)
            label_image[y, x] = unique_colors[color]

    print(f"  Found {len(unique_colors)} unique colors")

    # Segment image into regions using flood fill (regions already merged)
    print(f"  Segmenting image into regions using flood fill...")
    visited = np.zeros((height, width), dtype=bool)
    regions = []

    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                region = flood_fill_region(label_image, x, y, visited, width, height)
                if region:
                    regions.append(region)

    print(f"  Found {len(regions)} regions")

    # Create border image
    print(f"  Creating borders between regions...")
    border_image = _create_border_image(regions, width, height)

    # Create output image (white background)
    output = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(output)

    # Draw single-pixel borders for cleaner lines
    print(f"  Drawing borders (single pixel, color: {border_params.color})...")
    for y in range(height):
        for x in range(width):
            if border_image[y, x] == 1:
                draw.point((x, y), fill=border_params.color)

    print(f"  Placing numeric labels in {len(regions)} regions "
    f"(font: {numbers_params.font_size_in_mm.min}-{numbers_params.font_size_in_mm.max}mm)...")

    # Cache fonts of different sizes
    font_cache = {}

    def get_font(size: int):
        if size not in font_cache:
            try:
                # Try different font paths
                font = None
                for font_name in ['arial.ttf', 'Arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
                                '/System/Library/Fonts/Supplemental/Arial.ttf',
                                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
                    try:
                        font = ImageFont.truetype(font_name, size)
                        break
                    except:
                        continue
                if not font:
                    font = ImageFont.load_default()
                font_cache[size] = font
            except:
                font_cache[size] = ImageFont.load_default()
        return font_cache[size]

    # Place labels in regions
    for region in regions:
        if not region:
            continue

        # Get label for this region
        sample_x, sample_y = next(iter(region))
        label = label_image[sample_y, sample_x]

        # Calculate font size based on region area
        region_area = len(region)
        # Scale font size based on square root of area (to match region diameter)
        region_diameter = (region_area ** 0.5)
        font_size = int(min_font_size_pixels + (max_font_size_pixels - min_font_size_pixels) *
                       min(1.0, region_diameter / 500))  # 500 pixels diameter for max font size
        font_size = max(min_font_size_pixels, min(max_font_size_pixels, font_size))

        # Get font for this size
        font = get_font(font_size)

        # Find good position for label (centered in the region)
        label_x, label_y = _find_label_center(region, border_image, width, height)

        # Draw the label centered on the position
        text = str(label)
        # Get text bounding box for better centering
        bbox = draw.textbbox((label_x, label_y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text on the calculated position
        final_x = label_x - text_width // 2
        final_y = label_y - text_height // 2

        draw.text((final_x, final_y), text, font=font, fill=numbers_params.color)

    print(f"  Paint-by-numbers template complete!")
    return output
