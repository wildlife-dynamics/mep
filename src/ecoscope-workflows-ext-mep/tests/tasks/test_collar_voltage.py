from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytz
from ecoscope_workflows_core.tasks.filter._filter import TimeRange, TimezoneInfo, UTC_TIMEZONEINFO, DEFAULT_TIME_FORMAT
from ecoscope_workflows_ext_mep.tasks import calculate_collar_voltage


def test_calculate_collar_voltage():
    relocs_path = Path(__file__).parent.parent / "data" / "relocs.parquet"
    relocs = gpd.read_parquet(relocs_path)
    df = calculate_collar_voltage(
        relocs=relocs,
        time_range=TimeRange(
            since=pd.to_datetime("2025-01-01").replace(tzinfo=pytz.UTC),
            until=pd.to_datetime("2025-01-07").replace(tzinfo=pytz.UTC),
            timezone= UTC_TIMEZONEINFO,
            time_format= DEFAULT_TIME_FORMAT
        ),
    )

    assert len(df) > 0
