from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import AnyDataFrame

ColumnName = Annotated[str, Field(description="Column to aggregate")]

@register()
def dataframe_column_unique(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[list, Field(description="The number of unique values in the column")]:
    return df[column_name].unique()