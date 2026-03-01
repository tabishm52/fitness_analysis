"""Pre-processing of MyNetDiary and Strava fitness data.

Provides functions for loading and processing of MyNetDiary and Strava fitness
data into Pandas DataFrames. These DataFrames would typically be loaded into an
interactive Jupyter notebook for analysis and plotting.

The module assumes Strava data are an unzipped directory of a downloaded user
account archive from www.strava.com. For MyNetDiary, the module assumes a
directory of separate MyNetDiary_Year_XXXX.xls user data files downloaded from
www.mynetdiary.com.

Usage:

>>> import fitness_analysis as fa

>>> weight, calories = fa.load_mnd_data('path/to/MyNetDiary/files/', ...)
>>> activities = fa.load_strava_activities('path/to/Strava/archive/', ...)
"""

from .utils import time_series_piecewise_regression
from .utils import time_series_piecewise_regression_with_breaks
from .utils import time_series_constant_regression

from .mynetdiary import eer_male
from .mynetdiary import eer_female
from .mynetdiary import load_mnd_data

from .strava import load_strava_activities

from .commute import load_commute_activities
