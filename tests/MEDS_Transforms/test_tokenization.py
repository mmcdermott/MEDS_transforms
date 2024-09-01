"""Tests the tokenization script.

Note that this test relies on the normalized shards from the normalization test.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from datetime import datetime

import polars as pl

from tests.MEDS_Transforms import TOKENIZATION_SCRIPT

from ..utils import parse_meds_csvs
from .test_normalization import NORMALIZED_MEDS_SCHEMA
from .test_normalization import WANT_HELD_OUT_0 as NORMALIZED_HELD_OUT_0
from .test_normalization import WANT_SHARDS as NORMALIZED_SHARDS
from .test_normalization import WANT_TRAIN_1 as NORMALIZED_TRAIN_1
from .test_normalization import WANT_TUNING_0 as NORMALIZED_TUNING_0
from .transform_tester_base import single_stage_transform_tester

SECONDS_PER_DAY = 60 * 60 * 24


def ts_to_time_delta_days(ts: list[list[datetime]]) -> list[list[float]]:
    """TODO: Doctests"""
    out = []
    for subject_ts in ts:
        out.append([float("nan")])
        for i in range(1, len(subject_ts)):
            out[-1].append((subject_ts[i] - subject_ts[i - 1]).total_seconds() / SECONDS_PER_DAY)
    return out


# TODO: Make these schemas exportable, maybe???
# TODO: Why is the code getting converted to a float?
SCHEMAS_SCHEMA = {
    "subject_id": NORMALIZED_MEDS_SCHEMA["subject_id"],
    "code": pl.List(NORMALIZED_MEDS_SCHEMA["code"]),
    "numeric_value": pl.List(NORMALIZED_MEDS_SCHEMA["numeric_value"]),
    "start_time": NORMALIZED_MEDS_SCHEMA["time"],
    "time": pl.List(NORMALIZED_MEDS_SCHEMA["time"]),
}

SEQ_SCHEMA = {
    "subject_id": NORMALIZED_MEDS_SCHEMA["subject_id"],
    "code": pl.List(pl.List(pl.UInt8)),
    "numeric_value": pl.List(pl.List(NORMALIZED_MEDS_SCHEMA["numeric_value"])),
    "time_delta_days": pl.List(pl.Float32),
}

TRAIN_0_TIMES = [
    [
        datetime(1980, 12, 28),
        datetime(2010, 5, 11, 17, 41, 51),
        datetime(2010, 5, 11, 17, 48, 48),
        datetime(2010, 5, 11, 18, 25, 35),
        datetime(2010, 5, 11, 18, 57, 18),
        datetime(2010, 5, 11, 19, 27, 19),
    ],
    [
        datetime(1978, 6, 20),
        datetime(2010, 6, 20, 19, 23, 52),
        datetime(2010, 6, 20, 19, 25, 32),
        datetime(2010, 6, 20, 19, 45, 19),
        datetime(2010, 6, 20, 20, 12, 31),
        datetime(2010, 6, 20, 20, 24, 44),
        datetime(2010, 6, 20, 20, 41, 33),
        datetime(2010, 6, 20, 20, 50, 4),
    ],
]
WANT_SCHEMAS_TRAIN_0 = pl.DataFrame(
    {
        "subject_id": [239684, 1195293],
        "code": [[7, 9], [6, 9]],
        "numeric_value": [[None, 1.5770268440246582], [None, 0.06802856922149658]],
        "start_time": [ts[0] for ts in TRAIN_0_TIMES],
        "time": TRAIN_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_SCHEMAS_TRAIN_0_MISSING_STATIC = pl.DataFrame(
    {
        "subject_id": [239684, 1195293],
        "code": [None, [6, 9]],
        "numeric_value": [None, [None, 0.06802856922149658]],
        "start_time": [ts[0] for ts in TRAIN_0_TIMES],
        "time": TRAIN_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_TRAIN_0 = pl.DataFrame(
    {
        "subject_id": [239684, 1195293],
        "time_delta_days": ts_to_time_delta_days(TRAIN_0_TIMES),
        "code": [
            [[5], [1, 10, 11], [10, 11], [10, 11], [10, 11], [4]],
            [[5], [1, 10, 11], [10, 11], [10, 11], [10, 11], [10, 11], [10, 11], [4]],
        ],
        "numeric_value": [
            [
                [float("nan")],
                [float("nan"), -0.569736897945404, -1.2714673280715942],
                [-0.43754738569259644, -1.168027639389038],
                [0.001321975840255618, -1.37490713596344],
                [-0.04097883030772209, -1.5300706624984741],
                [float("nan")],
            ],
            [
                [float("nan")],
                [float("nan"), -0.23133166134357452, 0.7973543255932579],
                [0.03833488002419472, 0.7973543255932579],
                [0.3397272229194641, 0.745638906955719],
                [-0.046266332268714905, 0.6939135937995033],
                [-0.3000703752040863, 0.7973543255932579],
                [-0.31064537167549133, 1.004242181777954],
                [float("nan")],
            ],
        ],
    },
    schema=SEQ_SCHEMA,
)

TRAIN_1_TIMES = [
    [datetime(1978, 3, 9), datetime(2010, 5, 26, 2, 30, 56), datetime(2010, 5, 26, 4, 51, 52)],
    [datetime(1976, 3, 28), datetime(2010, 2, 5, 5, 55, 39), datetime(2010, 2, 5, 7, 2, 30)],
]

WANT_SCHEMAS_TRAIN_1 = pl.DataFrame(
    {
        "subject_id": [68729, 814703],
        "code": [[8, 9], [8, 9]],
        "numeric_value": [[None, -0.5438239574432373], [None, -1.1012336015701294]],
        "start_time": [ts[0] for ts in TRAIN_1_TIMES],
        "time": TRAIN_1_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_TRAIN_1 = pl.DataFrame(
    {
        "subject_id": [68729, 814703],
        "time_delta_days": ts_to_time_delta_days(TRAIN_1_TIMES),
        "code": [[[5], [3, 10, 11], [4]], [[5], [2, 10, 11], [4]]],
        "numeric_value": [
            [[float("nan")], [float("nan"), -1.4474751949310303, -0.3404940366744995], [float("nan")]],
            [[float("nan")], [float("nan"), 3.0046675205230713, 0.8490786552429199], [float("nan")]],
        ],
    },
    schema=SEQ_SCHEMA,
)

TUNING_0_TIMES = [[datetime(1988, 12, 19), datetime(2010, 1, 3, 6, 27, 59), datetime(2010, 1, 3, 8, 22, 13)]]

WANT_SCHEMAS_TUNING_0 = pl.DataFrame(
    {
        "subject_id": [754281],
        "code": [[7, 9]],
        "numeric_value": [[None, 0.28697699308395386]],
        "start_time": [ts[0] for ts in TUNING_0_TIMES],
        "time": TUNING_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_TUNING_0 = pl.DataFrame(
    {
        "subject_id": [754281],
        "time_delta_days": ts_to_time_delta_days(TUNING_0_TIMES),
        "code": [[[5], [3, 10, 11], [4]]],
        "numeric_value": [
            [[float("nan")], [float("nan"), 1.513569951057434, 0.6939190626144409], [float("nan")]],
        ],
    },
    schema=SEQ_SCHEMA,
)


HELD_OUT_0_TIMES = [
    [
        datetime(1986, 7, 20),
        datetime(2010, 6, 3, 14, 54, 38),
        datetime(2010, 6, 3, 15, 39, 49),
        datetime(2010, 6, 3, 16, 20, 49),
        datetime(2010, 6, 3, 16, 44, 26),
    ]
]

WANT_SCHEMAS_HELD_OUT_0 = pl.DataFrame(
    {
        "subject_id": [1500733],
        "code": [[7, 9]],
        "numeric_value": [[None, -0.7995940446853638]],
        "start_time": [ts[0] for ts in HELD_OUT_0_TIMES],
        "time": HELD_OUT_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_HELD_OUT_0 = pl.DataFrame(
    {
        "subject_id": [1500733],
        "time_delta_days": ts_to_time_delta_days(HELD_OUT_0_TIMES),
        "code": [[[5], [2, 10, 11], [10, 11], [10, 11], [4]]],
        "numeric_value": [
            [
                [float("nan")],
                [float("nan"), -1.1619458198547363, 0.7973587512969971],
                [-1.5320764780044556, 0.9525222778320312],
                [-1.230684518814087, 0.8490786552429199],
                [float("nan")],
            ]
        ],
    },
    schema=SEQ_SCHEMA,
)


WANT_SCHEMAS = {
    "schemas/train/0": WANT_SCHEMAS_TRAIN_0,
    "schemas/train/1": WANT_SCHEMAS_TRAIN_1,
    "schemas/tuning/0": WANT_SCHEMAS_TUNING_0,
    "schemas/held_out/0": WANT_SCHEMAS_HELD_OUT_0,
}

WANT_SCHEMAS_MISSING_STATIC = {
    "schemas/train/0": WANT_SCHEMAS_TRAIN_0_MISSING_STATIC,
    "schemas/train/1": WANT_SCHEMAS_TRAIN_1,
    "schemas/tuning/0": WANT_SCHEMAS_TUNING_0,
    "schemas/held_out/0": WANT_SCHEMAS_HELD_OUT_0,
}

WANT_EVENT_SEQS = {
    "event_seqs/train/0": WANT_EVENT_SEQ_TRAIN_0,
    "event_seqs/train/1": WANT_EVENT_SEQ_TRAIN_1,
    "event_seqs/tuning/0": WANT_EVENT_SEQ_TUNING_0,
    "event_seqs/held_out/0": WANT_EVENT_SEQ_HELD_OUT_0,
}

NORMALIZED_TRAIN_0 = """
subject_id,time,code,numeric_value
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

NORMALIZED_SHARDS_MISSING_STATIC = parse_meds_csvs(
    {
        "train/0": NORMALIZED_TRAIN_0,
        "train/1": NORMALIZED_TRAIN_1,
        "tuning/0": NORMALIZED_TUNING_0,
        "held_out/0": NORMALIZED_HELD_OUT_0,
    },
    schema=NORMALIZED_MEDS_SCHEMA,
)


def test_tokenization():
    single_stage_transform_tester(
        transform_script=TOKENIZATION_SCRIPT,
        stage_name="tokenization",
        transform_stage_kwargs=None,
        input_shards=NORMALIZED_SHARDS,
        want_data={**WANT_SCHEMAS, **WANT_EVENT_SEQS},
        df_check_kwargs={"check_column_order": False},
    )

    single_stage_transform_tester(
        transform_script=TOKENIZATION_SCRIPT,
        stage_name="tokenization",
        transform_stage_kwargs={"train_only": True},
        input_shards=NORMALIZED_SHARDS,
        want_data={**WANT_SCHEMAS, **WANT_EVENT_SEQS},
        should_error=True,
    )

    single_stage_transform_tester(
        transform_script=TOKENIZATION_SCRIPT,
        stage_name="tokenization",
        transform_stage_kwargs=None,
        input_shards=NORMALIZED_SHARDS_MISSING_STATIC,
        want_data={**WANT_SCHEMAS_MISSING_STATIC, **WANT_EVENT_SEQS},
        df_check_kwargs={"check_column_order": False},
    )
