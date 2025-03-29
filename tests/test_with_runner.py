"""Tests a multi-stage pre-processing pipeline via the Runner utility. Only checks final outputs.

In this test, the following stages are run:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization

The stage configuration arguments will be as given in the yaml block below:
"""

import re
from functools import partial

import polars as pl
from meds import code_metadata_filepath, subject_id_field, subject_splits_filepath

from tests.utils import MEDS_transforms_pipeline_tester, parse_shards_yaml


def add_params(templ_str: str, **kwargs):
    return templ_str.format(**kwargs)


def exact_str_regex(s: str) -> str:
    return f"^{re.escape(s)}$"


RUNNER_SCRIPT = "MEDS_transform-pipeline"
AGGREGATE_CODE_METADATA_SCRIPT = (
    "MEDS_transform-stage pkg://MEDS_transforms.configs._preprocess.yaml aggregate_code_metadata"
)

SPLITS_DF = pl.DataFrame(
    {
        subject_id_field: [239684, 1195293, 68729, 814703, 754281, 1500733],
        "split": ["train", "train", "train", "train", "tuning", "held_out"],
    }
)

MEDS_SHARDS = parse_shards_yaml(
    """
train/0: |-2
  subject_id,time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00",DOB,
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
  239684,"05/11/2010, 17:41:51",HR,102.6
  239684,"05/11/2010, 17:41:51",TEMP,96.0
  239684,"05/11/2010, 17:48:48",HR,105.1
  239684,"05/11/2010, 17:48:48",TEMP,96.2
  239684,"05/11/2010, 18:25:35",HR,113.4
  239684,"05/11/2010, 18:25:35",TEMP,95.8
  239684,"05/11/2010, 18:57:18",HR,112.6
  239684,"05/11/2010, 18:57:18",TEMP,95.5
  239684,"05/11/2010, 19:27:19",DISCHARGE,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00",DOB,
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

train/1: |-2
  subject_id,time,code,numeric_value
  68729,,EYE_COLOR//HAZEL,
  68729,,HEIGHT,160.3953106166676
  68729,"03/09/1978, 00:00:00",DOB,
  68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
  68729,"05/26/2010, 02:30:56",HR,86.0
  68729,"05/26/2010, 02:30:56",TEMP,97.8
  68729,"05/26/2010, 04:51:52",DISCHARGE,
  814703,,EYE_COLOR//HAZEL,
  814703,,HEIGHT,156.48559093209357
  814703,"03/28/1976, 00:00:00",DOB,
  814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
  814703,"02/05/2010, 05:55:39",HR,170.2
  814703,"02/05/2010, 05:55:39",TEMP,100.1
  814703,"02/05/2010, 07:02:30",DISCHARGE,

tuning/0: |-2
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

held_out/0: |-2
  subject_id,time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00",DOB,
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
    """
)

MEDS_CODE_METADATA = pl.DataFrame(
    {
        "code": ["EYE_COLOR//BLUE", "EYE_COLOR//BROWN", "EYE_COLOR//HAZEL", "HR", "TEMP"],
        "description": [
            "Blue Eyes. Less common than brown.",
            "Brown Eyes. The most common eye color.",
            "Hazel eyes. These are uncommon",
            "Heart Rate",
            "Body Temperature",
        ],
        "parent_codes": [None, None, None, ["LOINC/8867-4"], ["LOINC/8310-5"]],
    },
    schema={"code": pl.String, "description": pl.String, "parent_codes": pl.List(pl.String)},
)

# After filtering out subjects with fewer than 5 events:
WANT_FILTER = parse_shards_yaml(
    f"""
"filter_subjects/train/0": |-2
  {subject_id_field},time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00",DOB,
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
  239684,"05/11/2010, 17:41:51",HR,102.6
  239684,"05/11/2010, 17:41:51",TEMP,96.0
  239684,"05/11/2010, 17:48:48",HR,105.1
  239684,"05/11/2010, 17:48:48",TEMP,96.2
  239684,"05/11/2010, 18:25:35",HR,113.4
  239684,"05/11/2010, 18:25:35",TEMP,95.8
  239684,"05/11/2010, 18:57:18",HR,112.6
  239684,"05/11/2010, 18:57:18",TEMP,95.5
  239684,"05/11/2010, 19:27:19",DISCHARGE,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00",DOB,
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

"filter_subjects/train/1": |-2
  {subject_id_field},time,code,numeric_value

"filter_subjects/tuning/0": |-2
  {subject_id_field},time,code,numeric_value

"filter_subjects/held_out/0": |-2
  {subject_id_field},time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00",DOB,
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""
)

WANT_TIME_DERIVED = parse_shards_yaml(
    f"""
"add_time_derived_measurements/train/0": |-2
  {subject_id_field},time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00","TIME_OF_DAY//[00,06)",
  239684,"12/28/1980, 00:00:00",DOB,
  239684,"05/11/2010, 17:41:51","TIME_OF_DAY//[12,18)",
  239684,"05/11/2010, 17:41:51",AGE,29.36883360091833
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
  239684,"05/11/2010, 17:41:51",HR,102.6
  239684,"05/11/2010, 17:41:51",TEMP,96.0
  239684,"05/11/2010, 17:48:48","TIME_OF_DAY//[12,18)",
  239684,"05/11/2010, 17:48:48",AGE,29.36884681513314
  239684,"05/11/2010, 17:48:48",HR,105.1
  239684,"05/11/2010, 17:48:48",TEMP,96.2
  239684,"05/11/2010, 18:25:35","TIME_OF_DAY//[18,24)",
  239684,"05/11/2010, 18:25:35",AGE,29.36891675223647
  239684,"05/11/2010, 18:25:35",HR,113.4
  239684,"05/11/2010, 18:25:35",TEMP,95.8
  239684,"05/11/2010, 18:57:18","TIME_OF_DAY//[18,24)",
  239684,"05/11/2010, 18:57:18",AGE,29.36897705595538
  239684,"05/11/2010, 18:57:18",HR,112.6
  239684,"05/11/2010, 18:57:18",TEMP,95.5
  239684,"05/11/2010, 19:27:19","TIME_OF_DAY//[18,24)",
  239684,"05/11/2010, 19:27:19",AGE,29.369034127420306
  239684,"05/11/2010, 19:27:19",DISCHARGE,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00","TIME_OF_DAY//[00,06)",
  1195293,"06/20/1978, 00:00:00",DOB,
  1195293,"06/20/2010, 19:23:52","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 19:23:52",AGE,32.002896271955265
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:25:32","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 19:25:32",AGE,32.00289944083172
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 19:45:19","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 19:45:19",AGE,32.00293705539522
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:12:31","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 20:12:31",AGE,32.002988771458945
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:24:44","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 20:24:44",AGE,32.00301199932335
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 20:41:33",AGE,32.003043973286765
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04","TIME_OF_DAY//[18,24)",
  1195293,"06/20/2010, 20:50:04",AGE,32.00306016624544
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

"add_time_derived_measurements/train/1": |-2
  {subject_id_field},time,code,numeric_value

"add_time_derived_measurements/tuning/0": |-2
  {subject_id_field},time,code,numeric_value

"add_time_derived_measurements/held_out/0": |-2
  {subject_id_field},time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00","TIME_OF_DAY//[00,06)",
  1500733,"07/20/1986, 00:00:00",DOB,
  1500733,"06/03/2010, 14:54:38","TIME_OF_DAY//[12,18)",
  1500733,"06/03/2010, 14:54:38",AGE,23.873531791091356
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49","TIME_OF_DAY//[12,18)",
  1500733,"06/03/2010, 15:39:49",AGE,23.873617699332012
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:20:49","TIME_OF_DAY//[12,18)",
  1500733,"06/03/2010, 16:20:49",AGE,23.873695653692767
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26","TIME_OF_DAY//[12,18)",
  1500733,"06/03/2010, 16:44:26",AGE,23.873740556672114
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""
)

# Fit outliers python code
FIT_OUTLIERS_CODE = """
```python
>>> from tests.test_test_with_runner import WANT_TIME_DERIVED
>>> import polars as pl
>>> VALS = pl.col("numeric_value").drop_nulls().drop_nans()
>>> post_outliers = (
...     WANT_TIME_DERIVED['add_time_derived_measurements/train/0']
...     .group_by("code")
...     .agg(
...         VALS.len().alias("values/n_occurrences"),
...         VALS.sum().alias("values/sum"),
...         (VALS**2).sum().alias("values/sum_sqd")
...     )
...     .filter(pl.col("values/n_occurrences") > 0)
... )
>>> post_outliers
shape: (4, 4)
┌────────┬──────────────────────┬─────────────┬────────────────┐
│ code   ┆ values/n_occurrences ┆ values/sum  ┆ values/sum_sqd │
│ ---    ┆ ---                  ┆ ---         ┆ ---            │
│ str    ┆ u32                  ┆ f32         ┆ f32            │
╞════════╪══════════════════════╪═════════════╪════════════════╡
│ HR     ┆ 10                   ┆ 1104.300049 ┆ 122174.726562  │
│ AGE    ┆ 12                   ┆ 370.865448  ┆ 11482.001953   │
│ TEMP   ┆ 10                   ┆ 983.600037  ┆ 96788.53125    │
│ HEIGHT ┆ 2                    ┆ 339.958008  ┆ 57841.734375   │
└────────┴──────────────────────┴─────────────┴────────────────┘
>>> print(post_outliers.to_dict(as_series=False))
{'code': ['HR', 'AGE', 'TEMP', 'HEIGHT'],
 'values/n_occurrences': [10, 12, 10, 2],
 'values/sum': [1104.300048828125, 370.8654479980469, 983.6000366210938, 339.9580078125],
 'values/sum_sqd': [122174.7265625, 11482.001953125, 96788.53125, 57841.734375]}


```
"""

# Input:
# code,description,parent_codes
# EYE_COLOR//BLUE,"Blue Eyes. Less common than brown.",
# EYE_COLOR//BROWN,"Brown Eyes. The most common eye color.",
# EYE_COLOR//HAZEL,"Hazel eyes. These are uncommon",
# HR,"Heart Rate",LOINC/8867-4
# TEMP,"Body Temperature",LOINC/8310-5

WANT_FIT_OUTLIERS = {
    "fit_outlier_detection/codes.parquet": pl.DataFrame(
        {
            "code": [
                "EYE_COLOR//BLUE",
                "EYE_COLOR//BROWN",
                "HR",
                "TEMP",
                "AGE",
                "HEIGHT",
                "TIME_OF_DAY//[18,24)",
                "TIME_OF_DAY//[12,18)",
                "TIME_OF_DAY//[00,06)",
                "ADMISSION//CARDIAC",
                "DISCHARGE",
                "DOB",
            ],
            "values/n_occurrences": [0, 0, 10, 10, 12, 2, 0, 0, 0, 0, 0, 0],
            "values/sum": [
                0.0,
                0.0,
                1104.300048828125,
                983.6000366210938,
                370.8654479980469,
                339.9580078125,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "values/sum_sqd": [
                0.0,
                0.0,
                122174.7265625,
                96788.53125,
                11482.001953125,
                57841.734375,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "description": [
                "Blue Eyes. Less common than brown.",
                "Brown Eyes. The most common eye color.",
                "Heart Rate",
                "Body Temperature",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            "parent_codes": [
                None,
                None,
                ["LOINC/8867-4"],
                ["LOINC/8310-5"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        },
        schema={
            "code": pl.String,
            "description": pl.String,
            "parent_codes": pl.List(pl.String),
            "values/n_occurrences": pl.UInt8,  # In the real stage, this is shrunk, so it differs from the ex.
            "values/sum": pl.Float32,
            "values/sum_sqd": pl.Float32,
        },
    ).sort(by="code")
}

# For occluding outliers
OCCLUDE_OUTLIERS_CODE = """
```python
# This implies the following means and standard deviations
>>> from tests.test_test_with_runner import WANT_FIT_OUTLIERS as metadata_df
>>> mean_col = pl.col("values/sum") / pl.col("values/n_occurrences")
>>> stddev_col = (pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean_col**2) ** 0.5
>>> metadata_df.select(
...     "code",
...     (mean_col - stddev_col).alias("values/inlier_lower_bound"),
...     (mean_col + stddev_col).alias("values/inlier_upper_bound")
... )
shape: (4, 3)
┌────────┬───────────────────────────┬───────────────────────────┐
│ code   ┆ values/inlier_lower_bound ┆ values/inlier_upper_bound │
│ ---    ┆ ---                       ┆ ---                       │
│ str    ┆ f64                       ┆ f64                       │
╞════════╪═══════════════════════════╪═══════════════════════════╡
│ HR     ┆ 105.666951                ┆ 115.193058                │
│ AGE    ┆ 29.606836                 ┆ 32.204072                 │
│ TEMP   ┆ 96.319708                 ┆ 100.400299                │
│ HEIGHT ┆ 164.686989                ┆ 175.271019                │
└────────┴───────────────────────────┴───────────────────────────┘

```
"""

WANT_OCCLUDE_OUTLIERS = parse_shards_yaml(
    f"""
"occlude_outliers/train/0": |-2
  {subject_id_field},time,code,numeric_value,numeric_value/is_inlier
  239684,,EYE_COLOR//BROWN,,
  239684,,HEIGHT,,false
  239684,"12/28/1980, 00:00:00","TIME_OF_DAY//[00,06)",,
  239684,"12/28/1980, 00:00:00",DOB,,
  239684,"05/11/2010, 17:41:51","TIME_OF_DAY//[12,18)",,
  239684,"05/11/2010, 17:41:51",AGE,,false
  239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,,
  239684,"05/11/2010, 17:41:51",HR,,false
  239684,"05/11/2010, 17:41:51",TEMP,,false
  239684,"05/11/2010, 17:48:48","TIME_OF_DAY//[12,18)",,
  239684,"05/11/2010, 17:48:48",AGE,,false
  239684,"05/11/2010, 17:48:48",HR,,false
  239684,"05/11/2010, 17:48:48",TEMP,,false
  239684,"05/11/2010, 18:25:35","TIME_OF_DAY//[18,24)",,
  239684,"05/11/2010, 18:25:35",AGE,,false
  239684,"05/11/2010, 18:25:35",HR,113.4,true
  239684,"05/11/2010, 18:25:35",TEMP,,false
  239684,"05/11/2010, 18:57:18","TIME_OF_DAY//[18,24)",,
  239684,"05/11/2010, 18:57:18",AGE,,false
  239684,"05/11/2010, 18:57:18",HR,112.6,true
  239684,"05/11/2010, 18:57:18",TEMP,,false
  239684,"05/11/2010, 19:27:19","TIME_OF_DAY//[18,24)",,
  239684,"05/11/2010, 19:27:19",AGE,,false
  239684,"05/11/2010, 19:27:19",DISCHARGE,,
  1195293,,EYE_COLOR//BLUE,,
  1195293,,HEIGHT,,false
  1195293,"06/20/1978, 00:00:00","TIME_OF_DAY//[00,06)",,
  1195293,"06/20/1978, 00:00:00",DOB,,
  1195293,"06/20/2010, 19:23:52","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 19:23:52",AGE,32.002896271955265,true
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,,
  1195293,"06/20/2010, 19:23:52",HR,109.0,true
  1195293,"06/20/2010, 19:23:52",TEMP,100.0,true
  1195293,"06/20/2010, 19:25:32","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 19:25:32",AGE,32.00289944083172,true
  1195293,"06/20/2010, 19:25:32",HR,114.1,true
  1195293,"06/20/2010, 19:25:32",TEMP,100.0,true
  1195293,"06/20/2010, 19:45:19","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 19:45:19",AGE,32.00293705539522,true
  1195293,"06/20/2010, 19:45:19",HR,,false
  1195293,"06/20/2010, 19:45:19",TEMP,99.9,true
  1195293,"06/20/2010, 20:12:31","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 20:12:31",AGE,32.002988771458945,true
  1195293,"06/20/2010, 20:12:31",HR,112.5,true
  1195293,"06/20/2010, 20:12:31",TEMP,99.8,true
  1195293,"06/20/2010, 20:24:44","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 20:24:44",AGE,32.00301199932335,true
  1195293,"06/20/2010, 20:24:44",HR,107.7,true
  1195293,"06/20/2010, 20:24:44",TEMP,100.0,true
  1195293,"06/20/2010, 20:41:33","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 20:41:33",AGE,32.003043973286765,true
  1195293,"06/20/2010, 20:41:33",HR,107.5,true
  1195293,"06/20/2010, 20:41:33",TEMP,100.4,true
  1195293,"06/20/2010, 20:50:04","TIME_OF_DAY//[18,24)",,
  1195293,"06/20/2010, 20:50:04",AGE,32.00306016624544,true
  1195293,"06/20/2010, 20:50:04",DISCHARGE,,

"occlude_outliers/train/1": |-2
  {subject_id_field},time,code,numeric_value,numeric_value/is_inlier

"occlude_outliers/tuning/0": |-2
  {subject_id_field},time,code,numeric_value,numeric_value/is_inlier

"occlude_outliers/held_out/0": |-2
  {subject_id_field},time,code,numeric_value,numeric_value/is_inlier
  1500733,,EYE_COLOR//BROWN,,
  1500733,,HEIGHT,,false
  1500733,"07/20/1986, 00:00:00","TIME_OF_DAY//[00,06)",,
  1500733,"07/20/1986, 00:00:00",DOB,,
  1500733,"06/03/2010, 14:54:38","TIME_OF_DAY//[12,18)",,
  1500733,"06/03/2010, 14:54:38",AGE,,false
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,,
  1500733,"06/03/2010, 14:54:38",HR,,false
  1500733,"06/03/2010, 14:54:38",TEMP,100.0,true
  1500733,"06/03/2010, 15:39:49","TIME_OF_DAY//[12,18)",,
  1500733,"06/03/2010, 15:39:49",AGE,,false
  1500733,"06/03/2010, 15:39:49",HR,,false
  1500733,"06/03/2010, 15:39:49",TEMP,100.3,true
  1500733,"06/03/2010, 16:20:49","TIME_OF_DAY//[12,18)",,
  1500733,"06/03/2010, 16:20:49",AGE,,false
  1500733,"06/03/2010, 16:20:49",HR,,false
  1500733,"06/03/2010, 16:20:49",TEMP,100.1,true
  1500733,"06/03/2010, 16:44:26","TIME_OF_DAY//[12,18)",,
  1500733,"06/03/2010, 16:44:26",AGE,,false
  1500733,"06/03/2010, 16:44:26",DISCHARGE,,
"""
)

FIT_NORMALIZATION_CODE = """
```python
>>> from tests.test_test_with_runner import WANT_OCCLUDE_OUTLIERS as dfs
>>> import polars as pl
>>> VALS = pl.col("numeric_value").drop_nulls().drop_nans()
>>> post_transform = (
...     dfs[next(k for k in dfs.keys() if k.endswith("/train/0"))]
...     .group_by("code")
...     .agg(
...         pl.len().alias("code/n_occurrences"),
...         pl.col("subject_id").n_unique().alias("code/n_subjects"),
...         VALS.len().alias("values/n_occurrences"),
...         VALS.sum().alias("values/sum"),
...         (VALS**2).sum().alias("values/sum_sqd")
...     )
... )
>>> post_transform.filter(pl.col("values/n_occurrences") > 0)
shape: (3, 6)
┌──────┬────────────────────┬─────────────────┬──────────────────────┬────────────┬────────────────┐
│ code ┆ code/n_occurrences ┆ code/n_subjects ┆ values/n_occurrences ┆ values/sum ┆ values/sum_sqd │
│ ---  ┆ ---                ┆ ---             ┆ ---                  ┆ ---        ┆ ---            │
│ str  ┆ u32                ┆ u32             ┆ u32                  ┆ f32        ┆ f32            │
╞══════╪════════════════════╪═════════════════╪══════════════════════╪════════════╪════════════════╡
│ HR   ┆ 10                 ┆ 2               ┆ 7                    ┆ 776.799988 ┆ 86249.921875   │
│ TEMP ┆ 10                 ┆ 2               ┆ 6                    ┆ 600.100037 ┆ 60020.214844   │
│ AGE  ┆ 12                 ┆ 2               ┆ 7                    ┆ 224.020844 ┆ 7169.333496    │
└──────┴────────────────────┴─────────────────┴──────────────────────┴────────────┴────────────────┘
>>> print(post_transform.filter(pl.col("values/n_occurrences") > 0).to_dict(as_series=False))
{'code': ['HR', 'TEMP', 'AGE'],
 'code/n_occurrences': [10, 10, 12],
 'code/n_subjects': [2, 2, 2],
 'values/n_occurrences': [7, 6, 7],
 'values/sum': [776.7999877929688, 600.1000366210938, 224.02084350585938],
 'values/sum_sqd': [86249.921875, 60020.21484375, 7169.33349609375]}

"""

WANT_FIT_NORMALIZATION = {
    "fit_normalization/codes.parquet": pl.DataFrame(
        {
            "code": [
                "EYE_COLOR//BLUE",
                "EYE_COLOR//BROWN",
                "HR",
                "TEMP",
                "AGE",
                "HEIGHT",
                "TIME_OF_DAY//[18,24)",
                "TIME_OF_DAY//[12,18)",
                "TIME_OF_DAY//[00,06)",
                "ADMISSION//CARDIAC",
                "DISCHARGE",
                "DOB",
            ],
            "code/n_occurrences": [1, 1, 10, 10, 12, 2, 10, 2, 2, 2, 2, 2],
            "code/n_subjects": [1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2],
            "values/n_occurrences": [0, 0, 7, 6, 7, 0, 0, 0, 0, 0, 0, 0],
            "values/sum": [
                0.0,
                0.0,
                776.7999877929688,
                600.1000366210938,
                224.0208376784967,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "values/sum_sqd": [
                0.0,
                0.0,
                86249.921875,
                60020.21484375,
                7169.33349609375,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "description": [
                "Blue Eyes. Less common than brown.",
                "Brown Eyes. The most common eye color.",
                "Heart Rate",
                "Body Temperature",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            "parent_codes": [
                None,
                None,
                ["LOINC/8867-4"],
                ["LOINC/8310-5"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        },
        schema={
            "code": pl.String,
            "description": pl.String,
            "parent_codes": pl.List(pl.String),
            "code/n_occurrences": pl.UInt8,
            "code/n_subjects": pl.UInt8,
            "values/n_occurrences": pl.UInt8,  # In the real stage, this is shrunk, so it differs from the ex.
            "values/sum": pl.Float32,
            "values/sum_sqd": pl.Float32,
        },
    ).sort(by="code")
}

# As the last metadata stage, this gets a special directory.
WANT_FIT_VOCABULARY_INDICES = {
    "metadata/codes.parquet": pl.DataFrame(
        {
            "code": [
                "EYE_COLOR//BLUE",
                "EYE_COLOR//BROWN",
                "HR",
                "TEMP",
                "AGE",
                "HEIGHT",
                "TIME_OF_DAY//[18,24)",
                "TIME_OF_DAY//[12,18)",
                "TIME_OF_DAY//[00,06)",
                "ADMISSION//CARDIAC",
                "DISCHARGE",
                "DOB",
            ],
            "code/vocab_index": [5, 6, 8, 9, 2, 7, 12, 11, 10, 1, 3, 4],
            "code/n_occurrences": [1, 1, 10, 10, 12, 2, 10, 2, 2, 2, 2, 2],
            "code/n_subjects": [1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2],
            "values/n_occurrences": [0, 0, 7, 6, 7, 0, 0, 0, 0, 0, 0, 0],
            "values/sum": [
                0.0,
                0.0,
                776.7999877929688,
                600.1000366210938,
                224.0208376784967,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "values/sum_sqd": [
                0.0,
                0.0,
                86249.921875,
                60020.21484375,
                7169.33349609375,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "description": [
                "Blue Eyes. Less common than brown.",
                "Brown Eyes. The most common eye color.",
                "Heart Rate",
                "Body Temperature",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            "parent_codes": [
                None,
                None,
                ["LOINC/8867-4"],
                ["LOINC/8310-5"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        },
        schema={
            "code": pl.String,
            "description": pl.String,
            "parent_codes": pl.List(pl.String),
            "code/n_occurrences": pl.UInt8,
            "code/n_subjects": pl.UInt8,
            "code/vocab_index": pl.UInt8,
            "values/n_occurrences": pl.UInt8,
            "values/sum": pl.Float32,
            "values/sum_sqd": pl.Float32,
        },
    ).sort(by="code")
}


NORMALIZATION_CODE = """
```python
# This implies the following means and standard deviations
>>> import polars as pl
>>> from tests.test_test_with_runner import WANT_FIT_VOCABULARY_INDICES as metadata_df
>>> metadata_df = list(metadata_df.values())[0]
>>> from tests.test_test_with_runner import WANT_OCCLUDE_OUTLIERS as dfs
>>> mean_col = pl.col("values/sum") / pl.col("values/n_occurrences")
>>> stddev_col = (pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean_col**2) ** 0.5
>>> metadata_df = metadata_df.select(
...     "code",
...     "code/vocab_index",
...     mean_col.alias("values/mean"),
...     stddev_col.alias("values/stddev"),
... )
>>> metadata_df
shape: (12, 4)
┌──────────────────────┬──────────────────┬─────────────┬───────────────┐
│ code                 ┆ code/vocab_index ┆ values/mean ┆ values/stddev │
│ ---                  ┆ ---              ┆ ---         ┆ ---           │
│ str                  ┆ u8               ┆ f32         ┆ f32           │
╞══════════════════════╪══════════════════╪═════════════╪═══════════════╡
│ ADMISSION//CARDIAC   ┆ 1                ┆ NaN         ┆ NaN           │
│ AGE                  ┆ 2                ┆ 32.002979   ┆ NaN           │
│ DISCHARGE            ┆ 3                ┆ NaN         ┆ NaN           │
│ DOB                  ┆ 4                ┆ NaN         ┆ NaN           │
│ EYE_COLOR//BLUE      ┆ 5                ┆ NaN         ┆ NaN           │
│ …                    ┆ …                ┆ …           ┆ …             │
│ HR                   ┆ 8                ┆ 110.971428  ┆ 2.599767      │
│ TEMP                 ┆ 9                ┆ 100.01667   ┆ 0.1875        │
│ TIME_OF_DAY//[00,06) ┆ 10               ┆ NaN         ┆ NaN           │
│ TIME_OF_DAY//[12,18) ┆ 11               ┆ NaN         ┆ NaN           │
│ TIME_OF_DAY//[18,24) ┆ 12               ┆ NaN         ┆ NaN           │
└──────────────────────┴──────────────────┴─────────────┴───────────────┘
>>> import pprint
>>> pp = pprint.PrettyPrinter(width=80, compact=True)
>>> for k, df in dfs.items():
...    df = df.join(metadata_df, on="code").select(
...        "code/vocab_index",
...        (pl.col("numeric_value") - pl.col("values/mean")) / pl.col("values/stddev")
...    )
...    print("/".join(k.split("/")[1:]))
...    pp.pprint(df.to_dict(as_series=False))
train/0
{'code/vocab_index': [6, 7, 10, 4, 11, 2, 1, 8, 9, 11, 2, 8, 9, 12, 2, 8, 9, 12,
                      2, 8, 9, 12, 2, 3, 5, 7, 10, 4, 12, 2, 1, 8, 9, 12, 2, 8,
                      9, 12, 2, 8, 9, 12, 2, 8, 9, 12, 2, 8, 9, 12, 2, 8, 9, 12,
                      2, 3],
 'numeric_value': [None, None, None, None, None, None, None, None, None, None,
                   None, None, None, None, None, 0.9341503977775574, None, None,
                   None, 0.6264293789863586, None, None, None, None, None, None,
                   None, None, None, nan, None, -0.7583094239234924,
                   -0.0889078751206398, None, nan, 1.2034040689468384,
                   -0.0889078751206398, None, nan, None, -0.6222330927848816,
                   None, nan, 0.5879650115966797, -1.1555582284927368, None,
                   nan, -1.2583553791046143, -0.0889078751206398, None, nan,
                   -1.3352841138839722, 2.04443359375, None, nan, None]}
train/1
{'code/vocab_index': [], 'numeric_value': []}
tuning/0
{'code/vocab_index': [], 'numeric_value': []}
held_out/0
{'code/vocab_index': [6, 7, 10, 4, 11, 2, 8, 9, 11, 2, 8, 9, 11, 2, 8, 9, 11, 2,
                      3],
 'numeric_value': [None, None, None, None, None, None, None,
                   -0.0889078751206398, None, None, None, 1.5111083984375, None,
                   None, None, 0.4444173276424408, None, None, None]}

```
"""

# Note we have dropped the row in the held out shard that doesn't have a code in the vocabulary!
WANT_NORMALIZATION = parse_shards_yaml(
    f"""
"data/train/0": |-2
  {subject_id_field},time,code,numeric_value
  239684,,6,
  239684,,7,
  239684,"12/28/1980, 00:00:00",10,
  239684,"12/28/1980, 00:00:00",4,
  239684,"05/11/2010, 17:41:51",11,
  239684,"05/11/2010, 17:41:51",2,
  239684,"05/11/2010, 17:41:51",1,
  239684,"05/11/2010, 17:41:51",8,
  239684,"05/11/2010, 17:41:51",9,
  239684,"05/11/2010, 17:48:48",11,
  239684,"05/11/2010, 17:48:48",2,
  239684,"05/11/2010, 17:48:48",8,
  239684,"05/11/2010, 17:48:48",9,
  239684,"05/11/2010, 18:25:35",12,
  239684,"05/11/2010, 18:25:35",2,
  239684,"05/11/2010, 18:25:35",8,0.9341503977775574
  239684,"05/11/2010, 18:25:35",9,
  239684,"05/11/2010, 18:57:18",12,
  239684,"05/11/2010, 18:57:18",2,
  239684,"05/11/2010, 18:57:18",8,0.6264293789863586
  239684,"05/11/2010, 18:57:18",9,
  239684,"05/11/2010, 19:27:19",12,
  239684,"05/11/2010, 19:27:19",2,
  239684,"05/11/2010, 19:27:19",3,
  1195293,,5,
  1195293,,7,
  1195293,"06/20/1978, 00:00:00",10,
  1195293,"06/20/1978, 00:00:00",4,
  1195293,"06/20/2010, 19:23:52",12,
  1195293,"06/20/2010, 19:23:52",2,nan
  1195293,"06/20/2010, 19:23:52",1,
  1195293,"06/20/2010, 19:23:52",8,-0.7583094239234924
  1195293,"06/20/2010, 19:23:52",9,-0.0889078751206398
  1195293,"06/20/2010, 19:25:32",12,
  1195293,"06/20/2010, 19:25:32",2,nan
  1195293,"06/20/2010, 19:25:32",8,1.2034040689468384
  1195293,"06/20/2010, 19:25:32",9,-0.0889078751206398
  1195293,"06/20/2010, 19:45:19",12,
  1195293,"06/20/2010, 19:45:19",2,nan
  1195293,"06/20/2010, 19:45:19",8,
  1195293,"06/20/2010, 19:45:19",9,-0.6222330927848816
  1195293,"06/20/2010, 20:12:31",12,
  1195293,"06/20/2010, 20:12:31",2,nan
  1195293,"06/20/2010, 20:12:31",8,0.5879650115966797
  1195293,"06/20/2010, 20:12:31",9,-1.1555582284927368
  1195293,"06/20/2010, 20:24:44",12,
  1195293,"06/20/2010, 20:24:44",2,nan
  1195293,"06/20/2010, 20:24:44",8,-1.2583553791046143
  1195293,"06/20/2010, 20:24:44",9,-0.0889078751206398
  1195293,"06/20/2010, 20:41:33",12,
  1195293,"06/20/2010, 20:41:33",2,nan
  1195293,"06/20/2010, 20:41:33",8,-1.3352841138839722
  1195293,"06/20/2010, 20:41:33",9,2.04443359375
  1195293,"06/20/2010, 20:50:04",12,
  1195293,"06/20/2010, 20:50:04",2,nan
  1195293,"06/20/2010, 20:50:04",3,

"data/train/1": |-2
  {subject_id_field},time,code,numeric_value

"data/tuning/0": |-2
  {subject_id_field},time,code,numeric_value

"data/held_out/0": |-2
  {subject_id_field},time,code,numeric_value
  1500733,,6,
  1500733,,7,
  1500733,"07/20/1986, 00:00:00",10,
  1500733,"07/20/1986, 00:00:00",4,
  1500733,"06/03/2010, 14:54:38",11,
  1500733,"06/03/2010, 14:54:38",2,
  1500733,"06/03/2010, 14:54:38",8,
  1500733,"06/03/2010, 14:54:38",9,-0.0889078751206398
  1500733,"06/03/2010, 15:39:49",11,
  1500733,"06/03/2010, 15:39:49",2,
  1500733,"06/03/2010, 15:39:49",8,
  1500733,"06/03/2010, 15:39:49",9,1.5111083984375
  1500733,"06/03/2010, 16:20:49",11,
  1500733,"06/03/2010, 16:20:49",2,
  1500733,"06/03/2010, 16:20:49",8,
  1500733,"06/03/2010, 16:20:49",9,0.4444173276424408
  1500733,"06/03/2010, 16:44:26",11,
  1500733,"06/03/2010, 16:44:26",2,
  1500733,"06/03/2010, 16:44:26",3,
  """,
    code=pl.UInt8,
)

# Normally, you wouldn't need to specify all of these scripts, but in testing with local scripts we need to
# specify them all as they need to point to their python paths.
STAGE_RUNNER_YAML = f"""
fit_normalization:
  script: {AGGREGATE_CODE_METADATA_SCRIPT}
"""

PARALLEL_STAGE_RUNNER_YAML = f"""
parallelize:
  n_workers: 2
  launcher: "joblib"

{STAGE_RUNNER_YAML}
"""

PIPELINE_NO_STAGES_YAML = """
defaults:
  - _preprocess
  - _self_

input_dir: {input_dir}
cohort_dir: {cohort_dir}
"""

PIPELINE_YAML = f"""
defaults:
  - _preprocess
  - _self_

input_dir: {{input_dir}}
cohort_dir: {{cohort_dir}}

description: "A test pipeline for the MEDS-transforms pipeline runner."

stages:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization

stage_configs:
  filter_subjects:
    min_events_per_subject: 5
  add_time_derived_measurements:
    age:
      DOB_code: "DOB" # This is the MEDS official code for BIRTH
      age_code: "AGE"
      age_unit: "years"
    time_of_day:
      time_of_day_code: "TIME_OF_DAY"
      endpoints: [6, 12, 18, 24]
  fit_outlier_detection:
    _script: {str(AGGREGATE_CODE_METADATA_SCRIPT)}
    aggregations:
      - "values/n_occurrences"
      - "values/sum"
      - "values/sum_sqd"
  occlude_outliers:
    stddev_cutoff: 1
  fit_normalization:
    aggregations:
      - "code/n_occurrences"
      - "code/n_subjects"
      - "values/n_occurrences"
      - "values/sum"
      - "values/sum_sqd"
"""

NO_ARGS_HELP_STR = """
== MEDS-Transforms Pipeline Runner ==
MEDS-Transforms Pipeline Runner is a command line tool for running entire MEDS-transform pipelines in a single
command.

**MEDS-transforms Pipeline description:**

No description provided.
"""

WITH_CONFIG_HELP_STR = """
== MEDS-Transforms Pipeline Runner ==
MEDS-Transforms Pipeline Runner is a command line tool for running entire MEDS-transform pipelines in a single
command.

**MEDS-transforms Pipeline description:**

A test pipeline for the MEDS-transforms pipeline runner.
"""


def test_pipeline():
    shared_kwargs = {
        "config_name": "runner",
        "stage_name": None,
        "stage_kwargs": None,
        "do_pass_stage_name": False,
        "do_use_config_yaml": False,
        "do_include_dirs": False,
        "hydra_verbose": False,
    }

    MEDS_transforms_pipeline_tester(
        script=str(RUNNER_SCRIPT) + " -h",
        input_files={},
        want_outputs={},
        assert_no_other_outputs=True,
        should_error=False,
        test_name="Runner Help Test",
        stdout_regex=exact_str_regex(NO_ARGS_HELP_STR.strip()),
        **shared_kwargs,
    )

    MEDS_transforms_pipeline_tester(
        script=str(RUNNER_SCRIPT) + " -h",
        input_files={"pipeline.yaml": partial(add_params, PIPELINE_YAML)},
        want_outputs={},
        assert_no_other_outputs=True,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        test_name="Runner Help Test",
        stdout_regex=exact_str_regex(WITH_CONFIG_HELP_STR.strip()),
        **shared_kwargs,
    )

    shared_kwargs["script"] = RUNNER_SCRIPT

    MEDS_transforms_pipeline_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_YAML),
            "stage_runner.yaml": STAGE_RUNNER_YAML,
        },
        want_outputs={
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_VOCABULARY_INDICES,
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
        },
        assert_no_other_outputs=False,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        stage_runner_fp="{input_dir}/stage_runner.yaml",
        test_name="Runner Test",
        df_check_kwargs={"check_column_order": False},
        **shared_kwargs,
    )

    MEDS_transforms_pipeline_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_YAML),
            "stage_runner.yaml": PARALLEL_STAGE_RUNNER_YAML,
        },
        want_outputs={
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_VOCABULARY_INDICES,
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
        },
        assert_no_other_outputs=False,
        should_error=False,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        stage_runner_fp="{input_dir}/stage_runner.yaml",
        test_name="Runner Test with parallelism",
        df_check_kwargs={"check_column_order": False},
        **shared_kwargs,
    )

    MEDS_transforms_pipeline_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "_preprocess.yaml": partial(add_params, PIPELINE_YAML),
        },
        should_error=True,
        pipeline_config_fp="{input_dir}/_preprocess.yaml",
        test_name="Runner should fail if the pipeline config has an invalid name",
        **shared_kwargs,
    )

    MEDS_transforms_pipeline_tester(
        input_files={
            **{f"data/{k}": v for k, v in MEDS_SHARDS.items()},
            code_metadata_filepath: MEDS_CODE_METADATA,
            subject_splits_filepath: SPLITS_DF,
            "pipeline.yaml": partial(add_params, PIPELINE_NO_STAGES_YAML),
        },
        should_error=True,
        pipeline_config_fp="{input_dir}/pipeline.yaml",
        test_name="Runner should fail if the pipeline has no stages",
        **shared_kwargs,
    )
