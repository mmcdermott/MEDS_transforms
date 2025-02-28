"""Tests the merge to MEDS events process.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

from tests.MEDS_Extract import MERGE_TO_MEDS_COHORT_SCRIPT
from tests.utils import MEDS_transforms_pipeline_tester, parse_shards_yaml

EVENT_CFGS_YAML = """
subjects:
  subject_id_col: MRN
  eye_color:
    code:
      - EYE_COLOR
      - col(eye_color)
    time: null
    _metadata:
      demo_metadata:
        description: description
  height:
    code: HEIGHT
    time: null
    numeric_value: height
  dob:
    code: DOB
    time: col(dob)
    time_format: "%m/%d/%Y"
admit_vitals:
  admissions:
    code:
      - ADMISSION
      - col(department)
    time: col(admit_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  discharge:
    code: DISCHARGE
    time: col(disch_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  HR:
    code: HR
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: HR
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "HR"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "HR"}}
  temp:
    code: TEMP
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: temp
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "temp"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "temp"}}
"""

SHARDS_JSON = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}

INPUT_SHARDS = parse_shards_yaml(
    """
data/train/0/subjects/[0-6): |-2
  subject_id,time,code,numeric_value
  239684,,EYE_COLOR//BROWN,
  239684,,HEIGHT,175.271115221764
  239684,"12/28/1980, 00:00:00",DOB,
  1195293,,EYE_COLOR//BLUE,
  1195293,,HEIGHT,164.6868838269085
  1195293,"06/20/1978, 00:00:00",DOB,

data/train/1/subjects/[0-6): |-2
  subject_id,time,code,numeric_value
  68729,,EYE_COLOR//HAZEL,
  68729,,HEIGHT,160.3953106166676
  68729,"03/09/1978, 00:00:00",DOB,
  814703,,EYE_COLOR//HAZEL,
  814703,,HEIGHT,156.48559093209357
  814703,"03/28/1976, 00:00:00",DOB,

data/tuning/0/subjects/[0-6): |-2
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,

data/held_out/0/subjects/[0-6): |-2
  subject_id,time,code,numeric_value
  1500733,,EYE_COLOR//BROWN,
  1500733,,HEIGHT,158.60131573580904
  1500733,"07/20/1986, 00:00:00",DOB,

data/train/0/admit_vitals/[0-10): |-2
  subject_id,time,code,numeric_value
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
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:25:32",HR,114.1
  1195293,"06/20/2010, 19:25:32",TEMP,100.0
  1195293,"06/20/2010, 20:12:31",HR,112.5
  1195293,"06/20/2010, 20:12:31",TEMP,99.8
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

data/train/0/admit_vitals/[10-16): |-2
  subject_id,time,code,numeric_value
  1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
  1195293,"06/20/2010, 19:23:52",HR,109.0
  1195293,"06/20/2010, 19:23:52",TEMP,100.0
  1195293,"06/20/2010, 19:45:19",HR,119.8
  1195293,"06/20/2010, 19:45:19",TEMP,99.9
  1195293,"06/20/2010, 20:24:44",HR,107.7
  1195293,"06/20/2010, 20:24:44",TEMP,100.0
  1195293,"06/20/2010, 20:41:33",HR,107.5
  1195293,"06/20/2010, 20:41:33",TEMP,100.4
  1195293,"06/20/2010, 20:50:04",DISCHARGE,

data/train/1/admit_vitals/[0-10): |-2
  subject_id,time,code,numeric_value
  68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
  68729,"05/26/2010, 02:30:56",HR,86.0
  68729,"05/26/2010, 02:30:56",TEMP,97.8
  68729,"05/26/2010, 04:51:52",DISCHARGE,
  814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
  814703,"02/05/2010, 05:55:39",HR,170.2
  814703,"02/05/2010, 05:55:39",TEMP,100.1
  814703,"02/05/2010, 07:02:30",DISCHARGE,

data/train/1/admit_vitals/[10-16): |-2
  subject_id,time,code,numeric_value

data/tuning/0/admit_vitals/[0-10): |-2
  subject_id,time,code,numeric_value
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

data/tuning/0/admit_vitals/[10-16): |-2
  subject_id,time,code,numeric_value

data/held_out/0/admit_vitals/[0-10): |-2
  subject_id,time,code,numeric_value
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 16:20:49",HR,90.1
  1500733,"06/03/2010, 16:20:49",TEMP,100.1
  1500733,"06/03/2010, 16:44:26",DISCHARGE,

data/held_out/0/admit_vitals/[10-16): |-2
  subject_id,time,code,numeric_value
  1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
  1500733,"06/03/2010, 14:54:38",HR,91.4
  1500733,"06/03/2010, 14:54:38",TEMP,100.0
  1500733,"06/03/2010, 15:39:49",HR,84.4
  1500733,"06/03/2010, 15:39:49",TEMP,100.3
  1500733,"06/03/2010, 16:44:26",DISCHARGE,
    """
)

WANT_OUTPUTS = parse_shards_yaml(
    """
data/train/0: |-2
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


data/train/1: |-2
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

data/tuning/0: |-2
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

data/held_out/0: |-2
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


def test_merge_to_MEDS_cohort():
    MEDS_transforms_pipeline_tester(
        script=MERGE_TO_MEDS_COHORT_SCRIPT,
        stage_name="merge_to_MEDS_cohort",
        stage_kwargs=None,
        config_name="extract",
        input_files={
            **INPUT_SHARDS,
            "event_cfgs.yaml": EVENT_CFGS_YAML,
            "metadata/.shards.json": SHARDS_JSON,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        shards_map_fp="{input_dir}/metadata/.shards.json",
        want_outputs=WANT_OUTPUTS,
        df_check_kwargs={"check_column_order": False},
    )

    # Should error without event conversion file
    MEDS_transforms_pipeline_tester(
        script=MERGE_TO_MEDS_COHORT_SCRIPT,
        stage_name="merge_to_MEDS_cohort",
        stage_kwargs=None,
        config_name="extract",
        input_files={
            **INPUT_SHARDS,
            "metadata/.shards.json": SHARDS_JSON,
        },
        event_conversion_config_fp="{input_dir}/event_cfgs.yaml",
        shards_map_fp="{input_dir}/metadata/.shards.json",
        should_error=True,
    )
