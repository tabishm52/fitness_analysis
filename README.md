# fitness-analysis

Analysis tools for fitness activity data from Strava and MyNetDiary.

Provides functions for loading and processing MyNetDiary and Strava fitness data into Pandas DataFrames that are convenient for analysis and plotting.

The module assumes Strava data are an unzipped directory of a downloaded user account archive from www.strava.com.
For MyNetDiary, the module assumes a directory of separate `MyNetDiary_Year_XXXX.xls` user data files downloaded from www.mynetdiary.com.

## Installation

First install [activity-parser](https://github.com/tabishm52/activity_parser):

```bash
pip install activity-parser
```

Then install fitness-analysis:

```bash
pip install git+https://github.com/tabishm52/fitness_analysis.git
```

## Usage

```python
import fitness_analysis as fa

weight, calories = fa.load_mnd_data('path/to/MyNetDiary/files/', ...)
activities, weekly_sums = fa.load_strava_activities('path/to/Strava/archive/', ...)
power_curves = fa.load_power_curves('path/to/Strava/archive/', ...)
```

Other capabilities:

- `load_commute_activities` — summary metrics (distance, time, elevation) for recurring bike-commute activities.
  Distinct from `load_strava_activities` because it supports automatic splitting of round-trip commutes recorded as one activity.
- `load_activity_records` / `load_activity_coords` — load parsed FIT/TCX/GPX records or trimmed lat/lon data for a set of activity files.
- `cluster_routes` — cluster bicycle activities by GPS route similarity (Fréchet distance) or activity name.
- `geocode_positions` / `seed_geocode_cache` — reverse/forward geocode GPS positions into addresses.
- `piecewise_fit` / `piecewise_fit_auto` — piecewise linear regression on a time series, with automatic breakpoint/segment-count selection.

Most loaders cache their results to disk (Parquet or SQLite) and are paired with an `invalidate_*_cache` function to force a refresh.

## License

MIT
