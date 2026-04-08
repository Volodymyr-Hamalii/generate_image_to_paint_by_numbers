"""
Module for generating color palette images from paint-by-numbers images.
"""

from PIL import Image, ImageDraw, ImageFont

from utils.logger import logger


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Get a font at the specified size, trying various system fonts.

    Args:
        size: Font size in pixels

    Returns:
        ImageFont instance
    """
    for font_name in ['arial.ttf', 'Arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
                      '/System/Library/Fonts/Supplemental/Arial.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try:
            return ImageFont.truetype(font_name, size)
        except:
            continue
    return ImageFont.load_default()


def generate_color_palette_image(
    image_in_colors: Image.Image,
    swatch_size: int = 60,
    columns: int = 5,
    padding: int = 10,
    font_size: int = 16,
) -> Image.Image:
    """
    Generate a color palette image showing all colors used in the image with their numbers.

    The colors are numbered in the same order as they appear in the paint-by-numbers template
    (sequential as encountered scanning top-left to bottom-right).

    Args:
        image_in_colors: The processed image with reduced color palette
        swatch_size: Size of each color swatch in pixels
        columns: Number of columns in the palette grid
        padding: Padding between swatches and around the image
        font_size: Font size for the color numbers

    Returns:
        PIL Image showing the color palette with numbered swatches
    """
    # Extract unique colors in the same order as to_paint_by_numbers.py
    width, height = image_in_colors.size
    pixels = image_in_colors.load()

    if pixels is None:
        raise ValueError("Image pixels are None")

    unique_colors: dict[tuple[int, int, int], int] = {}
    for y in range(height):
        for x in range(width):
            color = pixels[x, y]
            if color not in unique_colors:
                unique_colors[color] = len(unique_colors)

    # Sort colors by their assigned number
    colors_list = sorted(unique_colors.items(), key=lambda x: x[1])
    num_colors = len(colors_list)

    logger.info(f"  Generating color palette with {num_colors} colors...")

    # Calculate image dimensions
    rows = (num_colors + columns - 1) // columns
    img_width = columns * swatch_size + (columns + 1) * padding
    img_height = rows * swatch_size + (rows + 1) * padding

    # Create output image (white background)
    output = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(output)

    # Get font
    font = _get_font(font_size)

    # Draw each color swatch with its number
    for color, number in colors_list:
        row = number // columns
        col = number % columns

        # Calculate swatch position
        x = padding + col * (swatch_size + padding)
        y = padding + row * (swatch_size + padding)

        # Draw color swatch
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=color, outline='black')

        # Draw number centered on the swatch
        text = str(number)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = x + (swatch_size - text_width) // 2
        text_y = y + (swatch_size - text_height) // 2

        # Choose text color based on brightness of background
        brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        text_color = 'black' if brightness > 128 else 'white'

        draw.text((text_x, text_y), text, font=font, fill=text_color)

    logger.info(f"  Color palette generated!")
    return output
