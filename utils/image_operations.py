
from pathlib import Path
from PIL import Image

from .logger import logger

# Increase PIL image size limit for large images (default is ~178 megapixels)
# Set to 500 megapixels to handle very large images safely
Image.MAX_IMAGE_PIXELS = 500_000_000


def read_image(image_path: Path) -> Image.Image | None:
    """
    Read an image from the given path.

    Args:
        image_path: Path to the image file

    Returns:
        PIL Image object in RGB mode
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {e}")
        return None
    return image.convert("RGB")


def center_crop_image(
    image: Image.Image,
    target_width_mm: int,
    target_height_mm: int
) -> Image.Image:
    """
    Crop the image symmetrically to match target aspect ratio.

    Crops equal amounts from top/bottom or left/right to achieve the target
    aspect ratio without distortion.

    Args:
        image: PIL Image object to crop
        target_width_mm: Target width in millimeters
        target_height_mm: Target height in millimeters

    Returns:
        Cropped PIL Image object
    """
    # Calculate aspect ratios
    target_aspect = target_width_mm / target_height_mm
    current_aspect = image.width / image.height

    # If aspect ratios match (within tolerance), no cropping needed
    if abs(target_aspect - current_aspect) < 0.001:
        return image

    # Determine crop dimensions
    if current_aspect > target_aspect:
        # Image is wider than target - crop left and right
        new_width = int(image.height * target_aspect)
        new_height = image.height
        left = (image.width - new_width) // 2
        top = 0
        right = left + new_width
        bottom = image.height
    else:
        # Image is taller than target - crop top and bottom
        new_width = image.width
        new_height = int(image.width / target_aspect)
        left = 0
        top = (image.height - new_height) // 2
        right = image.width
        bottom = top + new_height

    logger.info(f"  Cropping from {image.width}x{image.height} to {new_width}x{new_height} "
                f"(removed {image.width - new_width}px horizontally, {image.height - new_height}px vertically)")

    return image.crop((left, top, right, bottom))


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
    logger.info(f"> Saved image to {path}")
