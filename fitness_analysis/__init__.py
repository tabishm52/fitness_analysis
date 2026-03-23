"""Pre-processing of MyNetDiary and Strava fitness data.

Provides functions for loading and processing MyNetDiary and Strava fitness
data into Pandas DataFrames that are convenient for analysis and plotting.

The module assumes Strava data are an unzipped directory of a downloaded user
account archive from www.strava.com. For MyNetDiary, the module assumes a
directory of separate MyNetDiary_Year_XXXX.xls user data files downloaded from
www.mynetdiary.com.

Usage:

>>> import fitness_analysis as fa

>>> weight, calories = fa.load_mnd_data('path/to/MyNetDiary/files/', ...)
>>> activities = fa.load_strava_activities('path/to/Strava/archive/', ...)
"""

from .commute import load_commute_activities
from .mynetdiary import eer_female, eer_male, load_mnd_data
from .strava import load_strava_activities
from .utils import (
    time_series_constant_regression,
    time_series_piecewise_regression,
    time_series_piecewise_regression_with_breaks,
)

__all__ = [
    "eer_female",
    "eer_male",
    "load_commute_activities",
    "load_mnd_data",
    "load_strava_activities",
    "time_series_constant_regression",
    "time_series_piecewise_regression",
    "time_series_piecewise_regression_with_breaks",
]
