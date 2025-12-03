"""
This script generates the image to paint by the numbers.

It iterates over all the images in the images folder and generates the image to paint by the numbers.
Into the outputs it puts for each image 2 files:
1. The image converter to the new colours (if TO_USE_ONLY_ALLOWED_COLORS is True - only the allowed colours are used)
2. The B&W image to paint by the numbers (with sections and numbers).
"""

from pathlib import Path
import json
from typing import Any

import numpy as np
from PIL import Image

from generate_image.to_paint_by_numbers import generate_image_to_paint_by_numbers
from generate_image.in_the_specific_colors import generate_image_in_the_specific_colors
from generate_image.in_optimal_colors import generate_image_in_optimal_colors

TO_USE_ONLY_ALLOWED_COLORS: bool = True
MAX_NUMBER_OF_COLORS: int = 24


def read_image(image_path: Path) -> Image.Image:
    """
    Read an image from the given path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object in RGB mode
    """
    image = Image.open(image_path)
    return image.convert('RGB')

def save_image(image: Image.Image, output_dir: Path, image_name: str) -> None:
    """
    Save an image to the outputs directory.

    Args:
        image: PIL Image object to save
        output_dir: Directory to save the image
        image_name: Name of the file (should include extension)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / image_name
    image.save(path)
    print(f"Saved image to {path}")


def main() -> None:
    allowed_colors: list[str] = []
    if TO_USE_ONLY_ALLOWED_COLORS:
        allowed_colors = json.load(open('allowed_colors.json'))

    images_folder = Path('images')
    for image_path in list(images_folder.glob('*.jpg')) + list(images_folder.glob('*.png')):
        file_name: str = image_path.stem
        output_dir = Path('outputs') / file_name

        image = read_image(image_path)

        if TO_USE_ONLY_ALLOWED_COLORS:
            image_in_specific_colors = generate_image_in_the_specific_colors(image, allowed_colors)
        else:
            image_in_specific_colors = generate_image_in_optimal_colors(image, MAX_NUMBER_OF_COLORS)
        save_image(image_in_specific_colors, output_dir, file_name + '_in_colors.png')

        image_to_paint_by_numbers = generate_image_to_paint_by_numbers(image_in_specific_colors)
        save_image(image_to_paint_by_numbers, output_dir, file_name + '_by_numbers.png')

if __name__ == '__main__':
    main()
