# fitness-analysis

Analysis tools for fitness activity data from Strava and MyNetDiary.

Provides functions for loading and processing MyNetDiary and Strava fitness
data into Pandas DataFrames that are convenient for analysis and plotting.

The module assumes Strava data are an unzipped directory of a downloaded user
account archive from www.strava.com. For MyNetDiary, the module assumes a
directory of separate `MyNetDiary_Year_XXXX.xls` user data files downloaded
from www.mynetdiary.com.

## Installation

First install [activity-parser](https://github.com/GITHUB_USER/activity-parser):

```bash
pip install git+https://github.com/GITHUB_USER/activity-parser.git
```

Then install fitness-analysis:

```bash
pip install git+https://github.com/GITHUB_USER/fitness-analysis.git
```

## Usage

```python
import fitness_analysis as fa

weight, calories = fa.load_mnd_data('path/to/MyNetDiary/files/', ...)
activities, weekly_sums = fa.load_strava_activities('path/to/Strava/archive/', ...)
```

## License

MIT
