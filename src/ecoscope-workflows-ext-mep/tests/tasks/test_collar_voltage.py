import pandas as pd
import pytz
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_ext_mep.tasks import calculate_collar_voltage


def test_calculate_collar_voltage():
    calculate_collar_voltage(
        relocs=None,  # Replace with a mock or test GeoDataFrame
        time_range=TimeRange(
            since=pd.to_datetime("2025-01-01").replace(tzinfo=pytz.UTC),
            until=pd.to_datetime("2025-01-07").replace(tzinfo=pytz.UTC),
        ),  # Replace with a mock TimeRange object
    )
