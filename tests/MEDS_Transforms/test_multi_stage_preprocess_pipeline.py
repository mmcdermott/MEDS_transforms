"""Tests a multi-stage pre-processing pipeline. Only checks the end result, not the intermediate files.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.

In this test, the following stages are run:
  - filter_subjects
  - add_time_derived_measurements
  - fit_outlier_detection
  - occlude_outliers
  - fit_normalization
  - fit_vocabulary_indices
  - normalization
  - tokenization
  - tensorization

The stage configuration arguments will be as given in the yaml block below:
"""


from datetime import datetime

import polars as pl
from meds import subject_id_field
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from tests.MEDS_Transforms import (
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
    AGGREGATE_CODE_METADATA_SCRIPT,
    FILTER_SUBJECTS_SCRIPT,
    FIT_VOCABULARY_INDICES_SCRIPT,
    NORMALIZATION_SCRIPT,
    OCCLUDE_OUTLIERS_SCRIPT,
    TENSORIZATION_SCRIPT,
    TOKENIZATION_SCRIPT,
)
from tests.MEDS_Transforms.transform_tester_base import multi_stage_transform_tester, parse_shards_yaml

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

STAGE_CONFIG_YAML = """
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
>>> from tests.test_multi_stage_preprocess_pipeline import WANT_TIME_DERIVED
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
>>> from tests.test_multi_stage_preprocess_pipeline import WANT_FIT_OUTLIERS as metadata_df
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
  1195293,"06/20/2010, 20:24:44","TIME_OF_DAY//[18,24)",
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
>>> from tests.test_multi_stage_preprocess_pipeline import WANT_OCCLUDE_OUTLIERS as dfs
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
>>> from tests.test_multi_stage_preprocess_pipeline import WANT_FIT_VOCABULARY_INDICES as metadata_df
>>> metadata_df = list(metadata_df.values())[0]
>>> from tests.test_multi_stage_preprocess_pipeline import WANT_OCCLUDE_OUTLIERS as dfs
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
"normalization/train/0": |-2
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
  1195293,"06/20/2010, 20:24:44",12
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

"normalization/train/1": |-2
  {subject_id_field},time,code,numeric_value

"normalization/tuning/0": |-2
  {subject_id_field},time,code,numeric_value

"normalization/held_out/0": |-2
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

TOKENIZATION_SCHEMA_DF_SCHEMA = {
    subject_id_field: pl.Int64,
    "code": pl.List(pl.UInt8),
    "numeric_value": pl.List(pl.Float32),
    "start_time": pl.Datetime("us"),
    "time": pl.List(pl.Datetime("us")),
}
WANT_TOKENIZATION_SCHEMAS = {
    "tokenization/schemas/train/0": pl.DataFrame(
        {
            subject_id_field: [239684, 1195293],
            "code": [[6, 7], [5, 7]],
            "numeric_value": [[None, None], [None, None]],
            "start_time": [datetime(1980, 12, 28), datetime(1978, 6, 20)],
            "time": [
                [
                    datetime(1980, 12, 28, 0, 0, 0),
                    datetime(2010, 5, 11, 17, 41, 51),
                    datetime(2010, 5, 11, 17, 48, 48),
                    datetime(2010, 5, 11, 18, 25, 35),
                    datetime(2010, 5, 11, 18, 57, 18),
                    datetime(2010, 5, 11, 19, 27, 19),
                ],
                [
                    datetime(1978, 6, 20, 0, 0, 0),
                    datetime(2010, 6, 20, 19, 23, 52),
                    datetime(2010, 6, 20, 19, 25, 32),
                    datetime(2010, 6, 20, 19, 45, 19),
                    datetime(2010, 6, 20, 20, 12, 31),
                    datetime(2010, 6, 20, 20, 24, 44),
                    datetime(2010, 6, 20, 20, 41, 33),
                    datetime(2010, 6, 20, 20, 50, 4),
                ],
            ],
        },
        schema=TOKENIZATION_SCHEMA_DF_SCHEMA,
    ),
    "tokenization/schemas/train/1": pl.DataFrame(
        {k: [] for k in [subject_id_field, "code", "numeric_value", "start_time", "time"]},
        schema=TOKENIZATION_SCHEMA_DF_SCHEMA,
    ),
    "tokenization/schemas/tuning/0": pl.DataFrame(
        {k: [] for k in [subject_id_field, "code", "numeric_value", "start_time", "time"]},
        schema=TOKENIZATION_SCHEMA_DF_SCHEMA,
    ),
    "tokenization/schemas/held_out/0": pl.DataFrame(
        {
            subject_id_field: [1500733],
            "code": [[6, 7]],
            "numeric_value": [[None, None]],
            "start_time": [datetime(1986, 7, 20)],
            "time": [
                [
                    datetime(1986, 7, 20, 0, 0, 0),
                    datetime(2010, 6, 3, 14, 54, 38),
                    datetime(2010, 6, 3, 15, 39, 49),
                    datetime(2010, 6, 3, 16, 20, 49),
                    datetime(2010, 6, 3, 16, 44, 26),
                ]
            ],
        },
        schema=TOKENIZATION_SCHEMA_DF_SCHEMA,
    ),
}


TOKENIZATION_EVENT_SEQS_DF_SCHEMA = {
    subject_id_field: pl.Int64,
    "code": pl.List(pl.List(pl.UInt8)),
    "numeric_value": pl.List(pl.List(pl.Float32)),
    "time_delta_days": pl.List(pl.Float32),
}

WANT_TOKENIZATION_EVENT_SEQS = {
    "tokenization/event_seqs/train/0": pl.DataFrame(
        {
            subject_id_field: [239684, 1195293],
            "code": [
                [[10, 4], [11, 2, 1, 8, 9], [11, 2, 8, 9], [12, 2, 8, 9], [12, 2, 8, 9], [12, 2, 3]],
                [
                    [10, 4],
                    [12, 2, 1, 8, 9],
                    [12, 2, 8, 9],
                    [12, 2, 8, 9],
                    [12, 2, 8, 9],
                    [12, 2, 8, 9],
                    [12, 2, 8, 9],
                    [12, 2, 3],
                ],
            ],
            "numeric_value": [
                [
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
                    [float("nan"), float("nan"), float("nan"), float("nan")],
                    [float("nan"), float("nan"), 0.9341503977775574, float("nan")],
                    [float("nan"), float("nan"), 0.6264293789863586, float("nan")],
                    [float("nan"), float("nan"), float("nan")],
                ],
                [
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan"), float("nan"), -0.7583094239234924, -0.0889078751206398],
                    [float("nan"), float("nan"), 1.2034040689468384, -0.0889078751206398],
                    [float("nan"), float("nan"), float("nan"), -0.6222330927848816],
                    [float("nan"), float("nan"), 0.5879650115966797, -1.1555582284927368],
                    [float("nan"), float("nan"), -1.2583553791046143, -0.0889078751206398],
                    [float("nan"), float("nan"), -1.3352841138839722, 2.04443359375],
                    [float("nan"), float("nan"), float("nan")],
                ],
            ],
            "time_delta_days": (
                WANT_TOKENIZATION_SCHEMAS["tokenization/schemas/train/0"]
                .select(
                    pl.col("time")
                    .list.diff()
                    .list.eval((pl.element().dt.total_seconds() / 86400).fill_null(float("nan")))
                )["time"]
                .to_list()
            ),
        },
        schema=TOKENIZATION_EVENT_SEQS_DF_SCHEMA,
    ),
    "tokenization/event_seqs/train/1": pl.DataFrame(
        {k: [] for k in [subject_id_field, "code", "numeric_value", "time_delta_days"]},
        schema=TOKENIZATION_EVENT_SEQS_DF_SCHEMA,
    ),
    "tokenization/event_seqs/tuning/0": pl.DataFrame(
        {k: [] for k in [subject_id_field, "code", "numeric_value", "time_delta_days"]},
        schema=TOKENIZATION_EVENT_SEQS_DF_SCHEMA,
    ),
    "tokenization/event_seqs/held_out/0": pl.DataFrame(
        {
            subject_id_field: [1500733],
            "code": [
                [
                    [10, 4],
                    [11, 2, 8, 9],
                    [11, 2, 8, 9],
                    [11, 2, 8, 9],
                    [11, 2, 3],
                ]
            ],
            "numeric_value": [
                [
                    [float("nan"), float("nan")],
                    [float("nan"), float("nan"), float("nan"), -0.0889078751206398],
                    [float("nan"), float("nan"), float("nan"), 1.5111083984375],
                    [float("nan"), float("nan"), float("nan"), 0.4444173276424408],
                    [float("nan"), float("nan"), float("nan")],
                ]
            ],
            "time_delta_days": (
                WANT_TOKENIZATION_SCHEMAS["tokenization/schemas/held_out/0"]
                .select(
                    pl.col("time")
                    .list.diff()
                    .list.eval((pl.element().dt.total_seconds() / 86400).fill_null(float("nan")))
                )["time"]
                .to_list()
            ),
        },
        schema=TOKENIZATION_EVENT_SEQS_DF_SCHEMA,
    ),
}


WANT_NRTs = {
    "data/train/0.nrt": JointNestedRaggedTensorDict(
        WANT_TOKENIZATION_EVENT_SEQS["tokenization/event_seqs/train/0"]
        .select("time_delta_days", "code", "numeric_value")
        .to_dict(as_series=False)
    ),
    "data/train/1.nrt": JointNestedRaggedTensorDict({}),  # this shard was fully filtered out.
    "data/tuning/0.nrt": JointNestedRaggedTensorDict({}),  # this shard was fully filtered out.
    "data/held_out/0.nrt": JointNestedRaggedTensorDict(
        WANT_TOKENIZATION_EVENT_SEQS["tokenization/event_seqs/held_out/0"]
        .select("time_delta_days", "code", "numeric_value")
        .to_dict(as_series=False)
    ),
}


def test_pipeline():
    multi_stage_transform_tester(
        transform_scripts=[
            FILTER_SUBJECTS_SCRIPT,
            ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
            AGGREGATE_CODE_METADATA_SCRIPT,
            OCCLUDE_OUTLIERS_SCRIPT,
            AGGREGATE_CODE_METADATA_SCRIPT,
            FIT_VOCABULARY_INDICES_SCRIPT,
            NORMALIZATION_SCRIPT,
            TOKENIZATION_SCRIPT,
            TENSORIZATION_SCRIPT,
        ],
        stage_names=[
            "filter_subjects",
            "add_time_derived_measurements",
            "fit_outlier_detection",
            "occlude_outliers",
            "fit_normalization",
            "fit_vocabulary_indices",
            "normalization",
            "tokenization",
            "tensorization",
        ],
        stage_configs=STAGE_CONFIG_YAML,
        want_metadata={
            **WANT_FIT_OUTLIERS,
            **WANT_FIT_NORMALIZATION,
            **WANT_FIT_VOCABULARY_INDICES,
        },
        want_data={
            **WANT_FILTER,
            **WANT_TIME_DERIVED,
            **WANT_OCCLUDE_OUTLIERS,
            **WANT_NORMALIZATION,
            **WANT_TOKENIZATION_SCHEMAS,
            **WANT_TOKENIZATION_EVENT_SEQS,
            **WANT_NRTs,
        },
        input_code_metadata=MEDS_CODE_METADATA,
    )
