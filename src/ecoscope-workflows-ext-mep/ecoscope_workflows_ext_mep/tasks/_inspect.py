from pydantic import Field
from typing import Annotated
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame

from typing import TypeVar

T = TypeVar('T')

@task
def print_output(value: T) -> None:
    print("\n--- Print Output Task ---")
    print(f"Output value: {value}")
    
@task
def view_df(gdf: Annotated[AnyDataFrame, Field(description="A GeoDataFrame to inspect")], name: str) -> AnyDataFrame:
    print(f"\nDisplaying data for {name}")
    print("\n--- GeoDataFrame Summary ---")

    if gdf.empty:
        print("The GeoDataFrame is empty.")
        return gdf

    print(f"Rows: {gdf.shape[0]}")
    print(f"Columns: {gdf.shape[1]}")
    try:
        print(f"CRS: {gdf.crs}")
    except Exception:
        pass

    print("\n--- Column Details ---")
    print(f"Column names: {gdf.columns.tolist()}")
    for col in gdf.columns:
        print(f"Column '{col}':")
        print(f"  Non-null count: {gdf[col].notnull().sum()}")
        print(f"  Null count: {gdf[col].isnull().sum()}")
        print(f"  Unique values: {gdf[col].nunique()}")
        print(f"Data type: {gdf[col].dtype}")
    print(f"  Data type: {gdf.dtypes}")
    print("\n--- First 5 Rows ---")
    print(gdf.head(10))

    return gdf


# @task
# def view_data(rand_inf:List[ViewState,LayerDefinition])->str:
#    print("Data: {rand_inf}")