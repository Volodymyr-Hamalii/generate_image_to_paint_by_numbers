"""
This script generates the image to paint by the numbers.

It iterates over all the images in the images folder and generates the image to paint by the numbers.
Into the outputs it puts for each image 2 files:
1. The image converter to the new colours (if TO_USE_ONLY_ALLOWED_COLORS is True - only the allowed colours are used)
2. The B&W image to paint by the numbers (with sections and numbers).
"""

from pathlib import Path
from PIL import Image
import time

from utils.logger import logger
from utils.image_operations import read_image, center_crop_image, save_image
from parameters_reader import read_parameters, Parameters

from generate_image.to_paint_by_numbers import generate_image_to_paint_by_numbers
from generate_image.in_the_specific_colors import generate_image_in_the_specific_colors
from generate_image.in_optimal_colors import generate_image_in_optimal_colors


def get_output_dir(file_name: str) -> Path:
    """
    Get the output directory for the given file name.

    Args:
        file_name: Name of the file

    Returns:
        Output directory
    """
    max_attempts = 999
    for i in range(max_attempts):
        dir_name = f"{file_name}-{i}" if i else file_name
        output_dir = Path(__file__).parent / "outputs" / dir_name
        if not output_dir.exists():
            return output_dir
    raise ValueError(
        f"Output directory for {file_name} not found (after {max_attempts} attempts)")


def main() -> None:
    parameters: Parameters = read_parameters()

    for image_path in parameters.get_images_to_process_paths():
        start_time = time.time()
        logger.info(f"\nProcessing image: {image_path}")

        file_name: str = image_path.stem
        output_dir = get_output_dir(file_name)

        image = read_image(image_path)
        if image is None:
            logger.error(f"Error reading image {image_path}. Skipping...")
            continue

        # Apply center cropping if enabled and both dimensions are specified
        if (parameters.crop_to_fit.enabled and
            parameters.image_size_in_mm.width is not None and
            parameters.image_size_in_mm.height is not None):
            logger.info("Applying center crop to match target aspect ratio...")
            image = center_crop_image(
                image,
                parameters.image_size_in_mm.width,
                parameters.image_size_in_mm.height
            )

        # Generate the image in the necessary colors
        image_in_specific_colors: Image.Image
        if parameters.to_use_only_allowed_colors.value:
            logger.info("Generating image in the specific colors...")
            image_in_specific_colors = generate_image_in_the_specific_colors(
                image,
                parameters.to_use_only_allowed_colors.allowed_colors,
                parameters.image_size_in_mm,
                parameters.min_region_size_in_mm,
                parameters.compactness_passes,
            )
        else:
            logger.info("Generating image in the optimal colors...")
            image_in_specific_colors = generate_image_in_optimal_colors(
                image,
                parameters.max_number_of_colors,
                parameters.image_size_in_mm,
                parameters.min_region_size_in_mm,
                parameters.compactness_passes,
            )

        logger.info("Image in the colors generated. Saving...")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(image_in_specific_colors, output_dir, file_name + "_in_colors.png")

        logger.info("Generating image to paint by the numbers...")
        image_to_paint_by_numbers: Image.Image = generate_image_to_paint_by_numbers(
            image_in_specific_colors,
            parameters.border,
            parameters.numbers,
        )

        # Save the image to paint by the numbers
        logger.info("Image to paint by the numbers generated. Saving...")
        save_image(image_to_paint_by_numbers, output_dir, file_name + "_by_numbers.png")

        end_time = time.time()
        processing_time_seconds = end_time - start_time
        processing_time_minutes = processing_time_seconds / 60

        msg = f"# Image {image_path.name} processing completed in {processing_time_minutes:.2f} minutes"
        if processing_time_minutes > 60:
            processing_time_hours = processing_time_minutes / 60
            msg += f" ({processing_time_hours:.2f} hours)"

        logger.info(msg)


if __name__ == '__main__':
    main()
