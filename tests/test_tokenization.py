"""Tests the tokenization script.

Note that this test relies on the normalized shards from the normalization test.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from datetime import datetime

import polars as pl

from .test_normalization import NORMALIZED_MEDS_SCHEMA
from .test_normalization import WANT_SHARDS as NORMALIZED_SHARDS
from .transform_tester_base import TOKENIZATION_SCRIPT, single_stage_transform_tester

SECONDS_PER_DAY = 60 * 60 * 24


def ts_to_time_delta_days(ts: list[list[datetime]]) -> list[list[float]]:
    """TODO: Doctests"""
    out = []
    for patient_ts in ts:
        out.append([float("nan")])
        for i in range(1, len(patient_ts)):
            out[-1].append((patient_ts[i] - patient_ts[i - 1]).total_seconds() / SECONDS_PER_DAY)
    return out


# TODO: Make these schemas exportable, maybe???
# TODO: Why is the code getting converted to a float?
SCHEMAS_SCHEMA = {
    "patient_id": NORMALIZED_MEDS_SCHEMA["patient_id"],
    "code": pl.List(NORMALIZED_MEDS_SCHEMA["code"]),
    "numeric_value": pl.List(NORMALIZED_MEDS_SCHEMA["numeric_value"]),
    "start_time": NORMALIZED_MEDS_SCHEMA["time"],
    "time": pl.List(NORMALIZED_MEDS_SCHEMA["time"]),
}

SEQ_SCHEMA = {
    "patient_id": NORMALIZED_MEDS_SCHEMA["patient_id"],
    "code": pl.List(pl.List(pl.Float64)),
    "numeric_value": pl.List(pl.List(NORMALIZED_MEDS_SCHEMA["numeric_value"])),
    "time_delta_days": pl.List(pl.Float64),
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
        "patient_id": [239684, 1195293],
        "code": [[7, 9], [6, 9]],
        "numeric_value": [[None, 1.5770289975852931], [None, 0.0680278558478863]],
        "start_time": [ts[0] for ts in TRAIN_0_TIMES],
        "time": TRAIN_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_TRAIN_0 = pl.DataFrame(
    {
        "patient_id": [239684, 1195293],
        "time_delta_days": ts_to_time_delta_days(TRAIN_0_TIMES),
        "code": [
            [[5], [1, 10, 11], [10, 11], [10, 11], [10, 11], [4]],
            [[5], [1, 10, 11], [10, 11], [10, 11], [10, 11], [10, 11], [10, 11], [4]],
        ],
        "numeric_value": [
            [
                [float("nan")],
                [float("nan"), -0.5697368239808219, -1.2714603102818045],
                [-0.4375473056558053, -1.16801957848805],
                [0.0013218951832504667, -1.3749010420755592],
                [-0.04097875068075545, -1.5300621397661873],
                [float("nan")],
            ],
            [
                [float("nan")],
                [float("nan"), -0.23133165706877906, 0.7973543255932579],
                [0.03833496031425452, 0.7973543255932579],
                [0.3397270620952925, 0.7456339596963844],
                [-0.046266331413755815, 0.6939135937995033],
                [-0.30007020659778755, 0.7973543255932579],
                [-0.31064536806378906, 1.0042357891807672],
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
        "patient_id": [68729, 814703],
        "code": [[8, 9], [8, 9]],
        "numeric_value": [[None, -0.543824685211534], [None, -1.101236106768607]],
        "start_time": [ts[0] for ts in TRAIN_1_TIMES],
        "time": TRAIN_1_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_TRAIN_1 = pl.DataFrame(
    {
        "patient_id": [68729, 814703],
        "time_delta_days": ts_to_time_delta_days(TRAIN_1_TIMES),
        "code": [[[5], [3, 10, 11], [4]], [[5], [2, 10, 11], [4]]],
        "numeric_value": [
            [[float("nan")], [float("nan"), -1.4474752256589318, -0.3404937241380279], [float("nan")]],
            [[float("nan")], [float("nan"), 3.0046677515276268, 0.8490746914901316], [float("nan")]],
        ],
    },
    schema=SEQ_SCHEMA,
)

TUNING_0_TIMES = [[datetime(1988, 12, 19), datetime(2010, 1, 3, 6, 27, 59), datetime(2010, 1, 3, 8, 22, 13)]]

WANT_SCHEMAS_TUNING_0 = pl.DataFrame(
    {
        "patient_id": [754281],
        "code": [[7, 9]],
        "numeric_value": [[None, 0.28697820001946645]],
        "start_time": [ts[0] for ts in TUNING_0_TIMES],
        "time": TUNING_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_TUNING_0 = pl.DataFrame(
    {
        "patient_id": [754281],
        "time_delta_days": ts_to_time_delta_days(TUNING_0_TIMES),
        "code": [[[5], [3, 10, 11], [4]]],
        "numeric_value": [
            [[float("nan")], [float("nan"), 1.5135699848214401, 0.6939135937995033], [float("nan")]]
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
        "patient_id": [1500733],
        "code": [[7, 9]],
        "numeric_value": [[None, -0.7995957679188177]],
        "start_time": [ts[0] for ts in HELD_OUT_0_TIMES],
        "time": HELD_OUT_0_TIMES,
    },
    schema=SCHEMAS_SCHEMA,
)

WANT_EVENT_SEQ_HELD_OUT_0 = pl.DataFrame(
    {
        "patient_id": [1500733],
        "time_delta_days": ts_to_time_delta_days(HELD_OUT_0_TIMES),
        "code": [[[5], [2, 10, 11], [10, 11], [10, 11], [4]]],
        "numeric_value": [
            [
                [float("nan")],
                [float("nan"), -1.1619458660768958, 0.7973543255932579],
                [-1.5320765173869422, 0.9525154232838862],
                [-1.230684415605905, 0.8490746914901316],
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

WANT_EVENT_SEQS = {
    "event_seqs/train/0": WANT_EVENT_SEQ_TRAIN_0,
    "event_seqs/train/1": WANT_EVENT_SEQ_TRAIN_1,
    "event_seqs/tuning/0": WANT_EVENT_SEQ_TUNING_0,
    "event_seqs/held_out/0": WANT_EVENT_SEQ_HELD_OUT_0,
}


def test_tokenization():
    single_stage_transform_tester(
        transform_script=TOKENIZATION_SCRIPT,
        stage_name="tokenization",
        transform_stage_kwargs=None,
        input_shards=NORMALIZED_SHARDS,
        want_outputs={**WANT_SCHEMAS, **WANT_EVENT_SEQS},
    )
