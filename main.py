"""
This script generates the image to paint by the numbers.

It iterates over all the images in the images folder and generates the image to paint by the numbers.
Into the outputs it puts for each image 2 files:
1. The image converter to the new colours (if TO_USE_ONLY_ALLOWED_COLORS is True - only the allowed colours are used)
2. The B&W image to paint by the numbers (with sections and numbers).
"""

from pathlib import Path
from PIL import Image

from parameters_reader import read_parameters, Parameters
from generate_image.to_paint_by_numbers import generate_image_to_paint_by_numbers
from generate_image.in_the_specific_colors import generate_image_in_the_specific_colors
from generate_image.in_optimal_colors import generate_image_in_optimal_colors


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
    parameters: Parameters = read_parameters()

    for image_path in parameters.get_images_to_process_paths():
        file_name: str = image_path.stem
        output_dir = Path("outputs") / file_name

        image = read_image(image_path)

        # Generate the image in the necessary colors
        image_in_specific_colors: Image.Image
        if parameters.to_use_only_allowed_colors.value:
            image_in_specific_colors = generate_image_in_the_specific_colors(
                image,
                parameters.to_use_only_allowed_colors.allowed_colors,
                parameters.image_size_in_mm,
                parameters.min_region_size_in_mm)
        else:
            image_in_specific_colors = generate_image_in_optimal_colors(
                image,
                parameters.max_number_of_colors,
                parameters.image_size_in_mm,
                parameters.min_region_size_in_mm)
        
        # Save the image in the colors
        save_image(image_in_specific_colors, output_dir, file_name + "_in_colors.png")

        # Generate the image to paint by the numbers
        image_to_paint_by_numbers: Image.Image = generate_image_to_paint_by_numbers(
            image_in_specific_colors,
            parameters.min_region_size_in_mm,
            parameters.border)

        # Save the image to paint by the numbers
        save_image(image_to_paint_by_numbers, output_dir, file_name + "_by_numbers.png")

if __name__ == '__main__':
    main()
