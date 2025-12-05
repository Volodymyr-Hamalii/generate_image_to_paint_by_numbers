from pathlib import Path
from dataclasses import dataclass, field
import json

### Data classes ###

@dataclass(frozen=True)
class ParametersUseOnlyAllowedColors:
    _description: str
    value: bool
    allowed_colors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParametersImageSizeInMm:
    width: int
    height: int

@dataclass(frozen=True)
class ParametersBorder:
    width_in_mm: int
    color: str


@dataclass(frozen=True)
class ParametersNumbersFontSizeInMm:
    min: int
    max: int

@dataclass(frozen=True)
class ParametersNumbers:
    color: str
    font_size_in_mm: ParametersNumbersFontSizeInMm

@dataclass(frozen=True)
class Parameters:
    images_to_process: list[str]
    to_use_only_allowed_colors: ParametersUseOnlyAllowedColors
    max_number_of_colors: int
    image_size_in_mm: ParametersImageSizeInMm
    border: ParametersBorder
    numbers: ParametersNumbers
    min_region_size_in_mm: int

    def get_images_to_process_paths(self) -> list[Path]:
        if self.images_to_process:
            return [Path(__file__).parent / image_path for image_path in self.images_to_process]

        # If no images are specified, get all images from the images folder
        images_folder = Path(__file__).parent / "images"
        return [image_path for image_path in images_folder.iterdir()]


### Reader ###

def read_parameters() -> Parameters:
    parameters_file = Path(__file__).parent / "parameters.json"

    with open(parameters_file, "r") as f:
        parameters = json.load(f)
        return Parameters(
            images_to_process=parameters["images_to_process"],
            image_size_in_mm=ParametersImageSizeInMm(**parameters["image_size_in_mm"]),
            min_region_size_in_mm=parameters["min_region_size_in_mm"],
            border=ParametersBorder(**parameters["border"]),
            to_use_only_allowed_colors=ParametersUseOnlyAllowedColors(
                **parameters["to_use_only_allowed_colors"]),
            max_number_of_colors=parameters["max_number_of_colors"],
            numbers=ParametersNumbers(
                color=parameters["numbers"]["color"],
                font_size_in_mm=ParametersNumbersFontSizeInMm(
                    **parameters["numbers"]["font_size_in_mm"]),
            ),
        )
