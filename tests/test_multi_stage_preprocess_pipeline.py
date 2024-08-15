"""Tests a multi-stage pre-processing pipeline. Only checks the end result, not the intermediate files.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.

In this test, the following stages are run:
  - filter_patients
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

import polars as pl
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .transform_tester_base import (
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT,
    AGGREGATE_CODE_METADATA_SCRIPT,
    FILTER_PATIENTS_SCRIPT,
    FIT_VOCABULARY_INDICES_SCRIPT,
    NORMALIZATION_SCRIPT,
    OCCLUDE_OUTLIERS_SCRIPT,
    TENSORIZATION_SCRIPT,
    TOKENIZATION_SCRIPT,
    multi_stage_transform_tester,
    parse_shards_yaml,
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

STAGE_CONFIG_YAML = """
filter_patients:
  min_events_per_patient: 5
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
    - "code/n_patients"
    - "values/n_occurrences"
    - "values/sum"
    - "values/sum_sqd"
"""

# After filtering out patients with fewer than 5 events:
WANT_FILTER = parse_shards_yaml(
    """
  "filter_patients/train/0": |-2
    patient_id,time,code,numeric_value
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

  "filter_patients/train/1": |-2
    patient_id,time,code,numeric_value

  "filter_patients/tuning/0": |-2
    patient_id,time,code,numeric_value

  "filter_patients/held_out/0": |-2
    patient_id,time,code,numeric_value
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
    """
  "add_time_derived_measurements/train/0": |-2
    patient_id,time,code,numeric_value
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
    patient_id,time,code,numeric_value

  "add_time_derived_measurements/tuning/0": |-2
    patient_id,time,code,numeric_value

  "add_time_derived_measurements/held_out/0": |-2
    patient_id,time,code,numeric_value
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
>>> mean_col = pl.col("values/sum") / pl.col("values/n_occurrences")
>>> stddev_col = (pl.col("values/sum_sqd") / pl.col("values/n_occurrences") - mean_col**2) ** 0.5
>>> post_outliers.select(
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
    """
  "occlude_outliers/train/0": |-2
    patient_id,time,code,numeric_value,numeric_value/is_inlier
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
    patient_id,time,code,numeric_value,numeric_value/is_inlier

  "occlude_outliers/tuning/0": |-2
    patient_id,time,code,numeric_value,numeric_value/is_inlier

  "occlude_outliers/held_out/0": |-2
    patient_id,time,code,numeric_value,numeric_value/is_inlier
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
...         pl.col("patient_id").n_unique().alias("code/n_patients"),
...         VALS.len().alias("values/n_occurrences"),
...         VALS.sum().alias("values/sum"),
...         (VALS**2).sum().alias("values/sum_sqd")
...     )
... )
>>> post_transform.filter(pl.col("values/n_occurrences") > 0)
shape: (3, 6)
┌──────┬────────────────────┬─────────────────┬──────────────────────┬────────────┬────────────────┐
│ code ┆ code/n_occurrences ┆ code/n_patients ┆ values/n_occurrences ┆ values/sum ┆ values/sum_sqd │
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
 'code/n_patients': [2, 2, 2],
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
            "code/n_patients": [1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2],
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
            "code/n_patients": pl.UInt8,
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
            "code/n_patients": [1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2],
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
            "code/n_patients": pl.UInt8,
            "code/vocab_index": pl.UInt8,
            "values/n_occurrences": pl.UInt8,  # In the real stage, this is shrunk, so it differs from the ex.
            "values/sum": pl.Float32,
            "values/sum_sqd": pl.Float32,
        },
    ).sort(by="code")
}


WANT_NRTs = {
    "data/train/1.nrt": JointNestedRaggedTensorDict({}),  # this shard was fully filtered out.
    "data/tuning/0.nrt": JointNestedRaggedTensorDict({}),  # this shard was fully filtered out.
}


def test_pipeline():
    multi_stage_transform_tester(
        transform_scripts=[
            FILTER_PATIENTS_SCRIPT,
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
            "filter_patients",
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
            **WANT_NRTs,
        },
        outputs_from_cohort_dir=True,
        input_code_metadata=MEDS_CODE_METADATA,
    )
