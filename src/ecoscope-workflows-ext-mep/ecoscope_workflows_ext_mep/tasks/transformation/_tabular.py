from pydantic import Field
from typing import Annotated,cast
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame

ColumnName = Annotated[str, Field(description="Column to aggregate")]


@register()
def dataframe_column_unique(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[list, Field(description="The number of unique values in the column")]:
    return df[column_name].unique()

@register()
def reset_dataframe_index(
    df: AnyDataFrame,
    drop: bool = False,
) -> AnyDataFrame:
    """Reset the index, moving it into a column.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input frame.
    drop : bool, default False
        If False, the old index becomes a column (commonly `id` or
        `index`). If True, the old index is discarded.

    Returns
    -------
    GeoDataFrame
        A copy with the index reset.
    """
    result = df.reset_index(drop=drop)
    return result

@register()
def add_index_column(
    df: AnyDataFrame,
    column: str = "id",
) -> AnyDataFrame:
    """Add a column holding each row's index value.

    Parameters
    ----------
    gdf : GeoDataFrame
        The input frame.
    column : str, default "id"
        Name of the column to create from the index.

    Returns
    -------
    GeoDataFrame
        A copy with the new column added.

    Raises
    ------
    ValueError
        If `column` already exists in `gdf`.
    """
    df[column] = df.index
    return cast(AnyDataFrame, df)