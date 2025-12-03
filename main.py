"""
This script generates the image to paint by the numbers.

It iterates over all the images in the images folder and generates the image to paint by the numbers.
Into the outputs it puts for each image 2 files:
1. The image converter to the new colours (if TO_USE_ONLY_ALLOWED_COLORS is True - only the allowed colours are used)
2. The B&W image to paint by the numbers (with sections and numbers).
"""

from pathlib import Path
import json

from generate_image_to_paint_by_numbers import generate_image_to_paint_by_numbers
from generate_image_in_the_specific_colors import generate_image_in_the_specific_colors
from generate_image_in_optimal_colors import generate_image_in_optimal_colors

TO_USE_ONLY_ALLOWED_COLORS: bool = True
MAX_NUMBER_OF_COLORS: int = 24


def read_image(image_path: Path):
    ...

def save_image(image, image_name: str) -> None:
    path = Path('outputs') / image_name


def main() -> None:
    allowed_colors: list[str] = []
    if TO_USE_ONLY_ALLOWED_COLORS:
        allowed_colors = json.load(open('allowed_colors.json'))

    images_folder = Path('images')
    for image_path in images_folder.glob('*.jpg'):
        file_name: str = image_path.stem

        image = read_image(image_path)

        if TO_USE_ONLY_ALLOWED_COLORS:
            image_in_specific_colors = generate_image_in_the_specific_colors(image, allowed_colors)
        else:
            image_in_specific_colors = generate_image_in_optimal_colors(image, MAX_NUMBER_OF_COLORS)

        save_image(image_in_specific_colors, file_name + '_in_colors.png')

        image_to_paint_by_numbers = generate_image_to_paint_by_numbers(image_in_specific_colors)
        save_image(image_to_paint_by_numbers, file_name + '_by_numbers.png')

if __name__ == '__main__':
    main()
