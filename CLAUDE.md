# Project Context for Claude

## Project Overview

This is a Python-based tool that converts images into paint-by-numbers templates. The script takes input images and generates two outputs:
1. A color-optimized version of the image with a limited palette
2. A black & white paint-by-numbers template with numbered regions and borders

## Architecture

### Main Components

1. **`main.py`**: Entry point that orchestrates the image processing workflow
   - Reads parameters from `parameters.json`
   - Processes each image through color optimization and template generation
   - Saves outputs to the `outputs/` folder

2. **`parameters_reader.py`**: Configuration management
   - Defines dataclasses for all parameters
   - Reads and validates `parameters.json`
   - Provides type-safe parameter access

3. **`generate_image/`**: Core image processing modules
   - `in_optimal_colors.py`: K-means clustering for optimal color selection
   - `in_the_specific_colors.py`: Maps image to predefined color palette
   - `to_paint_by_numbers.py`: Generates the final paint-by-numbers template
   - `utils.py`: Shared utilities for region segmentation and merging

4. **`utils/`**: Common utilities
   - `logger.py`: Logging configuration

### Key Algorithms

1. **Color Reduction**:
   - K-means clustering to find optimal representative colors
   - LAB color space conversion for perceptually accurate color matching (deltaE distance)
   - Batch processing for memory efficiency on large images

2. **Region Processing**:
   - Flood fill algorithm to identify connected color regions
   - Two-phase region merging:
     - Size-based merging (small regions merged with largest neighbor)
     - Compactness-based merging (thin/elongated regions merged for better paintability)
   - Configurable multi-pass compactness checks

3. **Template Generation**:
   - Border drawing between regions
   - Automatic number placement in region centers
   - Dynamic font sizing based on region size

### Data Flow

```
Input Image
    ↓
Center Crop (if enabled and both dimensions specified)
    ↓
Color Reduction (k-means or palette mapping)
    ↓
Region Segmentation (flood fill)
    ↓
Region Merging (size + compactness)
    ↓
Color-Optimized Image Output
    ↓
Template Generation (borders + numbers)
    ↓
Paint-by-Numbers Template Output
```

## Configuration System

All processing parameters are defined in `parameters.json` and loaded through dataclasses in `parameters_reader.py`.

### Key Parameters

- **Image Size**: Width/height in mm (one can be null for auto-calculation based on aspect ratio)
- **Color Mode**: Either optimal (k-means) or specific (predefined palette)
- **Region Size**: Minimum region size in mm for merging
- **Compactness Passes**: Multi-pass system for merging elongated regions
- **Border & Numbers**: Visual styling for the template

## Recent Changes

### December 2024

1. **Center Cropping Feature** (`ParametersCropBehavior` and `main.py`):
   - Added `crop_to_fit` parameter to control cropping behavior
   - Implemented `center_crop_image()` function in `main.py`
   - When both width and height are specified and `enabled: true`, images are cropped symmetrically before resizing
   - Prevents distortion when target aspect ratio differs from source image
   - Cropping is symmetric: equal amounts removed from opposite edges
   - Applied in main.py before color generation functions are called

2. **Optional Dimension Support** (`ParametersImageSizeInMm`):
   - Made width/height optional (one required)
   - Added `get_dimensions()` method to calculate missing dimension from image aspect ratio
   - Maintains proportions when only one dimension is specified

2. **Configurable Compactness Passes**:
   - Moved hardcoded compactness passes from `utils.py` to `parameters.json`
   - Added `ParametersCompactnessPass` dataclass
   - Allows multiple passes with different thresholds for progressive region merging

## Code Style & Patterns

- Type hints throughout (Python 3.10+ syntax with `|` for unions)
- Frozen dataclasses for immutable configuration
- Comprehensive docstrings with Args/Returns sections
- Numpy arrays for efficient image processing
- PIL/Pillow for image I/O and manipulation

## Dependencies

- **Pillow**: Image processing and I/O
- **NumPy**: Array operations for efficient pixel manipulation
- Python 3.10+: Required for modern type hint syntax

## Performance Considerations

- Images are resized to 2 pixels/mm (configurable via `PIXELS_PER_MM` constant)
- Batch processing for color mapping to avoid memory issues
- Color sampling for k-means (max 10,000 samples) to speed up clustering
- Progress logging for long operations

## Testing & Validation

- Example images provided in `images/` folder
- Output examples in `outputs/markiza/` for reference
- Parameters validated during loading (e.g., width or height must be specified)

## Common Tasks

### Adding a New Parameter

1. Add field to relevant dataclass in `parameters_reader.py`
2. Update `read_parameters()` to parse from JSON
3. Add to `parameters.json` with default value
4. Pass through function signatures where needed
5. Update README.md documentation

### Modifying Color Algorithm

- Edit `generate_image/in_optimal_colors.py` for k-means approach
- Edit `generate_image/in_the_specific_colors.py` for palette mapping
- Both use LAB color space for perceptual accuracy

### Adjusting Region Merging

- Edit `merge_small_regions()` in `generate_image/utils.py`
- Configure via `compactness_passes` in `parameters.json`
- Consider both size and compactness thresholds

### Modifying Crop Behavior

- Edit `center_crop_image()` in `main.py` for cropping algorithm
- Current implementation uses symmetric center crop
- Controlled via `crop_to_fit.enabled` in `parameters.json`

## Known Limitations

- Processing time scales with image size and color complexity
- Very complex images may need parameter tuning for good results
- Font rendering for numbers may need adjustment for very small regions

## Future Enhancements (Ideas)

- Parallel processing for multiple images
- Interactive parameter tuning UI
- Color palette export (mapping numbers to paint colors)
- SVG output for scalable templates
- Advanced region merging strategies (e.g., semantic awareness)
