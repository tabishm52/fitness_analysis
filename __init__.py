"""Pre-processing of MyNetDiary and Strava fitness data.

Provides functions for loading and processing of MyNetDiary and Strava fitness
data into Pandas DataFrames. These functions would typically be called into an
interactive Jupyter notebook.

The module assumes Strava data are an unzipped directory of a downloaded user
account archive from www.strava.com. For MyNetDiary, the module assumes a
directory of separate MyNetDiary_Year_XXXX.xls user data files downloaded from
www.mynetdiary.com.

Usage:

>>> import fitness_analyis as fa

>>> weight, calories = fa.load_mnd_data('path/to/MyNetDiary/files/', ...)
>>> activities = fa.load_strava_activities('path/to/Strava/archive/')
"""

from .utils import time_series_linear_regression
from .utils import time_series_constant_regression
from .utils import eer_male
from .utils import eer_female

from .parse_fit_file import parse_fit
from .parse_tcx_gpx import parse_tcx
from .parse_tcx_gpx import parse_gpx
from .parse_activity import ActivityParser

from .main import load_mnd_data
from .main import load_strava_activities
