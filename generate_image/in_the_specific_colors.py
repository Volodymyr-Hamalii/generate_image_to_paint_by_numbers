"""
Module for converting images to use only specific allowed colors.
"""

from typing import Tuple

from PIL import Image, ImageFilter

from parameters_reader import ParametersImageSizeInMm


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
    1. Converts hex color strings to RGB tuples
    2. Maps each pixel to the nearest allowed color
    3. Applies median filtering to smooth the result

    Args:
        image: Input PIL Image in RGB mode
        allowed_colors: List of hex color strings (e.g., ["#FF0000", "#00FF00"])

    Returns:
        PIL Image with colors mapped to the allowed palette
    """
    # Convert hex colors to RGB
    palette = [_hex_to_rgb(color) for color in allowed_colors]

    # Get image size and pixels
    width, height = image.size
    pixels = image.load()

    # Map each pixel to nearest palette color
    all_coords = [(x, y) for x in range(width) for y in range(height)]
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
