"""Tests the normalization script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

import polars as pl

from .transform_tester_base import NORMALIZATION_SCRIPT, single_stage_transform_tester
from .utils import MEDS_PL_SCHEMA, parse_meds_csvs

# This is the code metadata file we'll use in this transform test. It is different than the default as we need
# a code/vocab_index
MEDS_CODE_METADATA_CSV = """
code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/sum,values/sum_sqd,code/vocab_index
ADMISSION//CARDIAC,2,2,0,,,1
ADMISSION//ORTHOPEDIC,1,1,0,,,2
ADMISSION//PULMONARY,1,1,0,,,3
DISCHARGE,4,4,0,,,4
DOB,4,4,0,,,5
EYE_COLOR//BLUE,1,1,0,,,6
EYE_COLOR//BROWN,1,1,0,,,7
EYE_COLOR//HAZEL,2,2,0,,,8
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,9
HR,12,4,12,1360.5000000000002,158538.77,10
TEMP,12,4,12,1181.4999999999998,116373.38999999998,11
"""

#
# The below string contains python code to use these numbers to compute the means and standard deviations
# of the codes, and to compute the normalized values that are observed:
NORMALIZED_VALS_CALC_STR = """
```python
import numpy as np

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

vals_by_code_and_subj = {
    "HR": [
        [102.6, 105.1, 113.4, 112.6],
        [109.0, 114.1, 119.8, 112.5, 107.7, 107.5],
        [86.0],
        [170.2],
        [142.0],
        [91.4,84.4,90.1],
    ],
    "TEMP": [
        [96.0, 96.2, 95.8, 95.5],
        [100.0, 100.0, 99.9, 99.8, 100.0, 100.4],
        [97.8],
        [100.1],
        [99.8],
        [100.0,100.3,100.1],
    ],
    "HEIGHT": [
        [175.271115221764],
        [164.6868838269085],
        [160.3953106166676],
        [156.48559093209357],
        [166.22261567137025],
        [158.60131573580904],
    ],
}

normalized_vals_by_code_and_subj = {}
for code, vals in vals_by_code_and_subj.items():
    mean, std = means_stds_by_code[code]
    normalized_vals_by_code_and_subj[code] = [
        [(np.float32(val) - mean) / std for val in subj_vals] for subj_vals in vals
    ]

for code, normalized_vals in normalized_vals_by_code_and_subj.items():
    print(f"Code: {code}")
    for subj_vals in normalized_vals:
        print([float(x) for x in subj_vals])
```
This returns:
```
Code: HR
[-0.569736897945404, -0.43754738569259644, 0.001321975840255618, -0.04097883030772209]
[-0.23133166134357452, 0.03833488002419472, 0.3397272229194641, -0.046266332268714905, -0.3000703752040863, -0.31064537167549133]
[-1.4474751949310303]
[3.0046675205230713]
[1.513569951057434]
[-1.1619458198547363, -1.5320764780044556, -1.230684518814087]
Code: TEMP
[-1.2714673280715942, -1.168027639389038, -1.37490713596344, -1.5300706624984741]
[0.7973587512969971, 0.7973587512969971, 0.745638906955719, 0.6939190626144409, 0.7973587512969971, 1.004242181777954]
[-0.3404940366744995]
[0.8490786552429199]
[0.6939190626144409]
[0.7973587512969971, 0.9525222778320312, 0.8490786552429199]
Code: HEIGHT
[1.5770268440246582]
[0.06802856922149658]
[-0.5438239574432373]
[-1.1012336015701294]
[0.28697699308395386]
[-0.7995940446853638]
```
"""  # noqa: E501

# In addition to the ages, the code/vocab_index by code is:
#     ADMISSION//CARDIAC: 1
#     ADMISSION//ORTHOPEDIC: 2
#     ADMISSION//PULMONARY: 3
#     DISCHARGE: 4
#     DOB: 5
#     EYE_COLOR//BLUE: 6
#     EYE_COLOR//BROWN: 7
#     EYE_COLOR//HAZEL: 8
#     HEIGHT: 9
#     HR: 10
#     TEMP: 11

WANT_TRAIN_0 = """
subject_id,time,code,numeric_value
239684,,7,
239684,,9,1.5770268440246582
239684,"12/28/1980, 00:00:00",5,
239684,"05/11/2010, 17:41:51",1,
239684,"05/11/2010, 17:41:51",10,-0.569736897945404
239684,"05/11/2010, 17:41:51",11,-1.2714673280715942
239684,"05/11/2010, 17:48:48",10,-0.43754738569259644
239684,"05/11/2010, 17:48:48",11,-1.168027639389038
239684,"05/11/2010, 18:25:35",10,0.001321975840255618
239684,"05/11/2010, 18:25:35",11,-1.37490713596344
239684,"05/11/2010, 18:57:18",10,-0.04097883030772209
239684,"05/11/2010, 18:57:18",11,-1.5300706624984741
239684,"05/11/2010, 19:27:19",4,
1195293,,6,
1195293,,9,0.06802856922149658
1195293,"06/20/1978, 00:00:00",5,
1195293,"06/20/2010, 19:23:52",1,
1195293,"06/20/2010, 19:23:52",10,-0.23133166134357452
1195293,"06/20/2010, 19:23:52",11,0.7973587512969971
1195293,"06/20/2010, 19:25:32",10,0.03833488002419472
1195293,"06/20/2010, 19:25:32",11,0.7973587512969971
1195293,"06/20/2010, 19:45:19",10,0.3397272229194641
1195293,"06/20/2010, 19:45:19",11,0.745638906955719
1195293,"06/20/2010, 20:12:31",10,-0.046266332268714905
1195293,"06/20/2010, 20:12:31",11,0.6939190626144409
1195293,"06/20/2010, 20:24:44",10,-0.3000703752040863
1195293,"06/20/2010, 20:24:44",11,0.7973587512969971
1195293,"06/20/2010, 20:41:33",10,-0.31064537167549133
1195293,"06/20/2010, 20:41:33",11,1.004242181777954
1195293,"06/20/2010, 20:50:04",4,
"""

WANT_TRAIN_1 = """
subject_id,time,code,numeric_value
68729,,8,
68729,,9,-0.5438239574432373
68729,"03/09/1978, 00:00:00",5,
68729,"05/26/2010, 02:30:56",3,
68729,"05/26/2010, 02:30:56",10,-1.4474751949310303
68729,"05/26/2010, 02:30:56",11,-0.3404940366744995
68729,"05/26/2010, 04:51:52",4,
814703,,8,
814703,,9,-1.1012336015701294
814703,"03/28/1976, 00:00:00",5,
814703,"02/05/2010, 05:55:39",2,
814703,"02/05/2010, 05:55:39",10,3.0046675205230713
814703,"02/05/2010, 05:55:39",11,0.8490786552429199
814703,"02/05/2010, 07:02:30",4,
"""

WANT_TUNING_0 = """
subject_id,time,code,numeric_value
754281,,7,
754281,,9,0.28697699308395386
754281,"12/19/1988, 00:00:00",5,
754281,"01/03/2010, 06:27:59",3,
754281,"01/03/2010, 06:27:59",10,1.513569951057434
754281,"01/03/2010, 06:27:59",11,0.6939190626144409
754281,"01/03/2010, 08:22:13",4,
"""

WANT_HELD_OUT_0 = """
subject_id,time,code,numeric_value
1500733,,7,
1500733,,9,-0.7995940446853638
1500733,"07/20/1986, 00:00:00",5,
1500733,"06/03/2010, 14:54:38",2,
1500733,"06/03/2010, 14:54:38",10,-1.1619458198547363
1500733,"06/03/2010, 14:54:38",11,0.7973587512969971
1500733,"06/03/2010, 15:39:49",10,-1.5320764780044556
1500733,"06/03/2010, 15:39:49",11,0.9525222778320312
1500733,"06/03/2010, 16:20:49",10,-1.230684518814087
1500733,"06/03/2010, 16:20:49",11,0.8490786552429199
1500733,"06/03/2010, 16:44:26",4,
"""

NORMALIZED_MEDS_SCHEMA = {**MEDS_PL_SCHEMA, "code": pl.UInt8}

WANT_SHARDS = parse_meds_csvs(
    {
        "train/0": WANT_TRAIN_0,
        "train/1": WANT_TRAIN_1,
        "tuning/0": WANT_TUNING_0,
        "held_out/0": WANT_HELD_OUT_0,
    },
    schema=NORMALIZED_MEDS_SCHEMA,
)


def test_normalization():
    single_stage_transform_tester(
        transform_script=NORMALIZATION_SCRIPT,
        stage_name="normalization",
        transform_stage_kwargs=None,
        input_code_metadata=MEDS_CODE_METADATA_CSV,
        want_data=WANT_SHARDS,
    )
