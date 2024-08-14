"""Tests the occlude outliers script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""


import polars as pl

from .transform_tester_base import OCCLUDE_OUTLIERS_SCRIPT, single_stage_transform_tester
from .utils import MEDS_PL_SCHEMA, parse_meds_csvs

# This is the code metadata
# MEDS_CODE_METADATA_CSV = """
# code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd,description,parent_code
# ,44,4,28,3198.8389005974336,382968.28937288234,,
# ADMISSION//CARDIAC,2,2,0,,,,
# ADMISSION//ORTHOPEDIC,1,1,0,,,,
# ADMISSION//PULMONARY,1,1,0,,,,
# DISCHARGE,4,4,0,,,,
# DOB,4,4,0,,,,
# EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
# EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
# EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
# HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
# HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
# TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
# """
#
# The below string contains python code to use these numbers to compute the means and standard deviations
# of the codes, and to compute the cutoffs that dictate if something is an outlier or not:
NORMALIZED_VALS_CALC_STR = """
```python
import numpy as np

# We'll set stddev_cutoff to 1 in this test.
CUTOFF=1

# These are the values/n_occurrences, values/sum, and values/sum_sqd for each of the codes with values:
stats_by_code = {
    "HEIGHT": (4, 656.8389005974336, 108056.12937288235),
    "HR": (12, 1360.5000000000002, 158538.77),
    "TEMP": (12, 1181.4999999999998, 116373.38999999998),
}

means_stds_by_code = {}
for code, (n_occurrences, sum_, sum_sqd) in stats_by_code.items():
    # These types are to match the input schema for the code metadata applied in these tests.
    n_occurrences = np.uint8(n_occurrences)
    sum_ = np.float32(sum_)
    sum_sqd = np.float32(sum_sqd)
    mean = sum_ / n_occurrences
    std = ((sum_sqd / n_occurrences) - mean**2)**0.5
    means_stds_by_code[code] = (mean, std)
    print(f"Code: {code}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Cut-off: {mean - CUTOFF * std}, {mean + CUTOFF * std}")
```
This returns:
```
Code: HEIGHT
Mean: 164.20973205566406
Std: 7.014064537200555
Cut-off: 157.1956675184635, 171.22379659286463
Code: HR
Mean: 113.375
Std: 18.912240786392818
Cut-off: 94.46275921360719, 132.28724078639283
Code: TEMP
Mean: 98.45833587646484
Std: 1.9334743338704625
Cut-off: 96.52486154259438, 100.39181021033531
```
"""  # noqa: E501

WANT_TRAIN_0 = """
patient_id,time,code,numeric_value,numeric_value/is_inlier
239684,,EYE_COLOR//BROWN,,
239684,,HEIGHT,,false
239684,"12/28/1980, 00:00:00",DOB,,
239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,,
239684,"05/11/2010, 17:41:51",HR,102.6,true
239684,"05/11/2010, 17:41:51",TEMP,,false
239684,"05/11/2010, 17:48:48",HR,105.1,true
239684,"05/11/2010, 17:48:48",TEMP,,false
239684,"05/11/2010, 18:25:35",HR,113.4,true
239684,"05/11/2010, 18:25:35",TEMP,,false
239684,"05/11/2010, 18:57:18",HR,112.6,true
239684,"05/11/2010, 18:57:18",TEMP,,false
239684,"05/11/2010, 19:27:19",DISCHARGE,,
1195293,,EYE_COLOR//BLUE,,
1195293,,HEIGHT,164.6868838269085,true
1195293,"06/20/1978, 00:00:00",DOB,,
1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,,
1195293,"06/20/2010, 19:23:52",HR,109.0,true
1195293,"06/20/2010, 19:23:52",TEMP,100.0,true
1195293,"06/20/2010, 19:25:32",HR,114.1,true
1195293,"06/20/2010, 19:25:32",TEMP,100.0,true
1195293,"06/20/2010, 19:45:19",HR,119.8,true
1195293,"06/20/2010, 19:45:19",TEMP,99.9,true
1195293,"06/20/2010, 20:12:31",HR,112.5,true
1195293,"06/20/2010, 20:12:31",TEMP,99.8,true
1195293,"06/20/2010, 20:24:44",HR,107.7,true
1195293,"06/20/2010, 20:24:44",TEMP,100.0,true
1195293,"06/20/2010, 20:41:33",HR,107.5,true
1195293,"06/20/2010, 20:41:33",TEMP,,false
1195293,"06/20/2010, 20:50:04",DISCHARGE,,
"""

WANT_TRAIN_1 = """
patient_id,time,code,numeric_value,numeric_value/is_inlier
68729,,EYE_COLOR//HAZEL,,
68729,,HEIGHT,160.3953106166676,true
68729,"03/09/1978, 00:00:00",DOB,,
68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,,
68729,"05/26/2010, 02:30:56",HR,,false
68729,"05/26/2010, 02:30:56",TEMP,97.8,true
68729,"05/26/2010, 04:51:52",DISCHARGE,,
814703,,EYE_COLOR//HAZEL,,
814703,,HEIGHT,,false
814703,"03/28/1976, 00:00:00",DOB,,
814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,,
814703,"02/05/2010, 05:55:39",HR,,false
814703,"02/05/2010, 05:55:39",TEMP,100.1,true
814703,"02/05/2010, 07:02:30",DISCHARGE,,
"""

WANT_TUNING_0 = """
patient_id,time,code,numeric_value,numeric_value/is_inlier
754281,,EYE_COLOR//BROWN,,
754281,,HEIGHT,166.22261567137025,true
754281,"12/19/1988, 00:00:00",DOB,,
754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,,
754281,"01/03/2010, 06:27:59",HR,,false
754281,"01/03/2010, 06:27:59",TEMP,99.8,true
754281,"01/03/2010, 08:22:13",DISCHARGE,,
"""

WANT_HELD_OUT_0 = """
patient_id,time,code,numeric_value,numeric_value/is_inlier
1500733,,EYE_COLOR//BROWN,,
1500733,,HEIGHT,158.60131573580904,true
1500733,"07/20/1986, 00:00:00",DOB,,
1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,,
1500733,"06/03/2010, 14:54:38",HR,,false
1500733,"06/03/2010, 14:54:38",TEMP,100.0,true
1500733,"06/03/2010, 15:39:49",HR,,false
1500733,"06/03/2010, 15:39:49",TEMP,100.3,true
1500733,"06/03/2010, 16:20:49",HR,,false
1500733,"06/03/2010, 16:20:49",TEMP,100.1,true
1500733,"06/03/2010, 16:44:26",DISCHARGE,,
"""

WANT_SHARDS = parse_meds_csvs(
    {
        "train/0": WANT_TRAIN_0,
        "train/1": WANT_TRAIN_1,
        "tuning/0": WANT_TUNING_0,
        "held_out/0": WANT_HELD_OUT_0,
    },
    schema={
        **MEDS_PL_SCHEMA,
        "numeric_value/is_inlier": pl.Boolean,
    },
)


def test_occlude_outliers():
    single_stage_transform_tester(
        transform_script=OCCLUDE_OUTLIERS_SCRIPT,
        stage_name="occlude_outliers",
        transform_stage_kwargs={"stddev_cutoff": 1},
        want_data=WANT_SHARDS,
    )
