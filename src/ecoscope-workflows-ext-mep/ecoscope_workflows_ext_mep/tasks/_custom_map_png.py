import re
import tempfile
from pathlib import Path
from typing import Union, Optional
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_ext_custom.tasks.results._map import ViewState
from ecoscope_workflows_ext_custom.tasks.io._html_to_png import ScreenshotConfig, html_to_png
import logging

logger = logging.getLogger(__name__)


@task
def zoom_map_and_screenshot(
    input_file: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = None,
    view_state: Optional[ViewState] = None,
    screenshot_config: Optional[ScreenshotConfig] = None,
) -> Optional[str]:
    """
    Adjust the view state in a pydeck HTML map file, take a screenshot, and clean up.

    This function:
    1. Reads the HTML file
    2. Adjusts the view state (longitude, latitude, zoom, pitch, bearing)
    3. Saves to a temporary file
    4. Takes a screenshot using html_to_png with the same filename as input
    5. Removes the temporary file
    6. Returns the path to the PNG image

    Args:
        input_file: Path to the input HTML file (can be None, will return None)
        output_dir: Directory to save the PNG screenshot
        view_state: ViewState object containing longitude, latitude, zoom, pitch, bearing
        screenshot_config: Optional ScreenshotConfig for html_to_png

    Returns:
        Path to the generated PNG file, or None if input_file is None
    """
    if input_file is None:
        logger.info("Input_file is None - skipping...")
        return None

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}\n" f"Check that the HTML generation task completed successfully."
        )

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    logger.info(f"Input file: {input_path}")
    logger.info(f"Output dir: {output_dir}")

    # Read the HTML file
    with open(input_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # If view_state is provided, replace all view parameters
    if view_state is not None:
        logger.info("Applying ViewState:")
        logger.info(f"  Longitude: {view_state.longitude}")
        logger.info(f"  Latitude: {view_state.latitude}")
        logger.info(f"  Zoom: {view_state.zoom}")
        logger.info(f"  Pitch: {view_state.pitch}")
        logger.info(f"  Bearing: {view_state.bearing}")

        # Replace each view state parameter
        view_params = {
            "bearing": view_state.bearing,
            "latitude": view_state.latitude,
            "longitude": view_state.longitude,
            "pitch": view_state.pitch,
            "zoom": view_state.zoom,
        }

        new_html_content = html_content
        params_found = 0

        for param_name, param_value in view_params.items():
            pattern = rf'"{param_name}":\s*(-?\d+\.?\d*)'
            match = re.search(pattern, new_html_content)
            if match:
                logger.info(f"Found {param_name} value: {match.group(1)}")
                new_html_content = re.sub(pattern, f'"{param_name}": {param_value}', new_html_content)
                params_found += 1
            else:
                logger.warning(f"Could not find {param_name} in HTML")

        if params_found == 0:
            raise ValueError(
                f"Could not find any view state parameters in the HTML file: {input_path}\n"
                f"The file may not be a valid pydeck HTML map."
            )
    else:
        logger.info("No ViewState provided - using original view parameters")
        new_html_content = html_content

    # Extract the base filename (without extension) from input file
    original_filename = input_path.stem
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", prefix=f"{original_filename}_", delete=False, encoding="utf-8"
    ) as temp_file:
        temp_file.write(new_html_content)
        temp_html_path = temp_file.name

    logger.info(f"Created temporary file: {temp_html_path}")

    try:
        # Use the existing html_to_png function to take screenshot
        if screenshot_config is None:
            screenshot_config = ScreenshotConfig()

        # Call html_to_png which will generate the PNG
        png_path = html_to_png(html_path=temp_html_path, output_dir=str(output_dir), config=screenshot_config)

        # Rename the PNG to match the original HTML filename
        png_path_obj = Path(png_path)
        desired_png_name = f"{original_filename}.png"
        desired_png_path = png_path_obj.parent / desired_png_name

        # If the generated PNG has a different name, rename it
        if png_path_obj.name != desired_png_name:
            png_path_obj.rename(desired_png_path)
            png_path = str(desired_png_path)
            logger.info(f"Renamed PNG to: {desired_png_name}")

        logger.info(f"\n ßßSuccessfully saved screenshot to: {png_path}")

        return png_path

    finally:
        # Clean up temporary file
        temp_path = Path(temp_html_path)
        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"Temporary file removed: {temp_html_path}")
