from pathlib import Path
from dataclasses import dataclass, field
import json

### Data classes ###

@dataclass(frozen=True)
class ParametersUseOnlyAllowedColors:
    value: bool
    description: str
    allowed_colors: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class Parameters:    
    to_use_only_allowed_colors: ParametersUseOnlyAllowedColors
    max_number_of_colors: int
    image_size_in_mm: dict[str, int]
    min_region_size_in_mm: int


### Reader ###

def read_parameters() -> Parameters:
    parameters_file = Path(__file__).parent / 'parameters.json'

    with open(parameters_file, 'r') as f:
        parameters = json.load(f)
        return Parameters(
            to_use_only_allowed_colors=ParametersUseOnlyAllowedColors(**parameters['to_use_only_allowed_colors']),
            max_number_of_colors=parameters['max_number_of_colors'],
            image_size_in_mm=parameters['image_size_in_mm'],
            min_region_size_in_mm=parameters['min_region_size_in_mm']
        )
