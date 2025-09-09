from typing import Annotated
from ecoscope_workflows_core.decorators import task
from pydantic import Field
import os


@task
def create_directory(
    path_name: Annotated[
        str,
        Field(
            description="Path to the directory that should be created",
            default=os.path.join(os.path.dirname(__file__), "output"),
        ),
    ],
) -> str:
    """
    Creates a directory at the specified path if it doesn't already exist.

    Args:
        path_name (str): Directory path to create. Defaults to ./output in the script directory.

    Returns:
        str: The path to the created (or existing) directory.
    """
    print("Creating directory to store our data...")
    os.makedirs(path_name, exist_ok=True)
    print(f"Successfully created directory {path_name}")
    return path_name
