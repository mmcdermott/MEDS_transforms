"""Tests the extract values script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from .transform_tester_base import EXTRACT_VALUES_SCRIPT, parse_shards_yaml, single_stage_transform_tester

INPUT_SHARDS = parse_shards_yaml(
    """
    train/0: |-2
      patient_id,time,code,numeric_value,text_value
      239684,,EYE_COLOR//BROWN,,
      239684,"12/28/1980, 00:00:00",DOB,,
      239684,"05/11/2010, 17:41:51",BP,,"120/80"
      1195293,,EYE_COLOR//BLUE,,
      1195293,"06/20/1978, 00:00:00",DOB,,
      1195293,"06/20/2010, 19:23:52",BP,,"144/96"
      1195293,"06/20/2010, 19:23:52",HR,80,
      1195293,"06/20/2010, 19:23:52",TEMP,,"100F"
    train/1: |-2
      patient_id,time,code,numeric_value,text_value
      68729,,EYE_COLOR//HAZEL,,
      68729,"03/09/1978, 00:00:00",DOB,,
      814703,"02/05/2010, 05:55:39",HR,170.2,
      1195293,"06/20/2010, 19:23:52",TEMP,,"37C"
      814703,,EYE_COLOR//HAZEL,,
      814703,"03/28/1976, 00:00:00",DOB,,
      814703,"02/05/2010, 05:55:39",HR,170.2,
    tuning/0: |-2
      patient_id,time,code,numeric_value,text_value
      754281,,EYE_COLOR//BROWN,,
      754281,"12/19/1988, 00:00:00",DOB,,
      754281,"01/03/2010, 06:27:59",HR,142.0,
      754281,"06/20/2010, 20:23:50",BP,,"134/76"
      754281,"06/20/2010, 21:00:02",TEMP,,"36.2C"
    held_out/0: |-2
      patient_id,time,code,numeric_value,text_value
      1500733,,EYE_COLOR//BROWN,,
      1500733,"07/20/1986, 00:00:00",DOB,,
      1500733,"06/03/2010, 14:54:38",HR,91.4
      1500733,"06/03/2010, 14:54:38",BP,,"123/82"
    """
)

WANT_SHARDS = parse_shards_yaml(
    """
    train/0: |-2
      patient_id,time,code,numeric_value,text_value
      239684,,EYE_COLOR//BROWN,,
      239684,"12/28/1980, 00:00:00",DOB,,
      239684,"05/11/2010, 17:41:51",BP//SYSTOLIC,120,
      239684,"05/11/2010, 17:41:51",BP//DIASTOLIC,80,
      1195293,,EYE_COLOR//BLUE,,
      1195293,"06/20/1978, 00:00:00",DOB,,
      1195293,"06/20/2010, 19:23:52",BP//SYSTOLIC,144,
      1195293,"06/20/2010, 19:23:52",BP//DIASTOLIC,96,
      1195293,"06/20/2010, 19:23:52",HR,80,
      1195293,"06/20/2010, 19:23:52",TEMP//F,100,
    train/1: |-2
      patient_id,time,code,numeric_value,text_value
      68729,,EYE_COLOR//HAZEL,,
      68729,"03/09/1978, 00:00:00",DOB,,
      814703,"02/05/2010, 05:55:39",HR,170.2,
      1195293,"06/20/2010, 19:23:52",TEMP//C,37,
      814703,,EYE_COLOR//HAZEL,,
      814703,"03/28/1976, 00:00:00",DOB,,
      814703,"02/05/2010, 05:55:39",HR,170.2,
    tuning/0: |-2
      patient_id,time,code,numeric_value,text_value
      754281,,EYE_COLOR//BROWN,,
      754281,"12/19/1988, 00:00:00",DOB,,
      754281,"01/03/2010, 06:27:59",HR,142.0,
      754281,"06/20/2010, 20:23:50",BP//SYSTOLIC,134,
      754281,"06/20/2010, 20:23:50",BP//DIASTOLIC,76,
      754281,"06/20/2010, 21:00:02",TEMP//C,36.2,
    held_out/0: |-2
      patient_id,time,code,numeric_value,text_value
      1500733,,EYE_COLOR//BROWN,,
      1500733,"07/20/1986, 00:00:00",DOB,,
      1500733,"06/03/2010, 14:54:38",HR,91.4,
      1500733,"06/03/2010, 14:54:38",BP//SYSTOLIC,123,
      1500733,"06/03/2010, 14:54:38",BP//DIASTOLIC,82,
    """
)


def test_extract_values():
    single_stage_transform_tester(
        transform_script=EXTRACT_VALUES_SCRIPT,
        stage_name="extract_values",
        transform_stage_kwargs={
            "_match_revise_mode": "multi_match_and_revise",
            "_match_revise": [
                {
                    "_matcher": {"code": "BP"},
                    "numeric_value": {"extract": {"from": "text_value", "regex": r"(\d+)/.*"}},
                    "code": "{code}//SYSTOLIC",
                    "text_value": {"literal": None},
                },
                {
                    "_matcher": {"code": "BP"},
                    "numeric_value": {"extract": {"from": "text_value", "regex": r".*/(\d+)"}},
                    "code": "{code}//DIASTOLIC",
                    "text_value": {"literal": None},
                },
            ],
        },
        input_shards=INPUT_SHARDS,
        want_data=WANT_SHARDS,
        do_use_config_yaml=True,
    )
