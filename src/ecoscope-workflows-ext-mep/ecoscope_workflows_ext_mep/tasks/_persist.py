import hashlib
import io
from typing import Annotated
from ecoscope_workflows_core.annotations import AnyDataFrame
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.serde import _persist_text
from pydantic import Field


@task
def gdf_to_geojson(
    df: Annotated[AnyDataFrame, Field(description="Dataframe to persist")],
    root_path: Annotated[str, Field(description="Root path to persist text to")],
    filename: Annotated[
        str | None,
        Field(
            description="""\
            Optional filename to persist text to within the `root_path`.
            If not provided, a filename will be generated based on a hash of the df content.
            """,
        ),
    ] = None,
) -> Annotated[str, Field(description="Path to persisted GeoJSON file")]:
    """Persist dataframe to a GeoJSON file or cloud storage object."""
    import geopandas as gpd  # type: ignore[import-untyped]
    import pandas as pd

    if not filename:
        try:
            hash_values = pd.util.hash_pandas_object(df).values
            hash_input = bytes(hash_values)
        except (TypeError, ValueError):
            content = f"{df.shape}{df.head(5).to_dict()}"
            hash_input = content.encode()
        filename = hashlib.sha256(hash_input).hexdigest()[:7]

    buffer = io.StringIO()
    gdf = gpd.GeoDataFrame(df)
    gdf.to_file(buffer, driver="GeoJSON")
    return _persist_text(buffer.getvalue(), root_path, f"{filename}.geojson")
