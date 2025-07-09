# Simple MEDS-Transform Example Package

> [!NOTE]
> Note that the code blocks provided here will act as though they are run from the \_repository root
> directory\_ (i.e., one directory above this one).

In this directory, we show an example of using MEDS-Transform with an external package which defines both
custom stages and uses built-in stages to run a larger pipeline. This example is split between the following
files:

```python
>>> print_directory("example", PrintConfig(ignore_regex=r"__pycache__|.egg-info"))
├── README.md
├── data
│   ├── data
│   │   ├── held_out
│   │   │   └── 0.parquet
│   │   ├── train
│   │   │   ├── 0.parquet
│   │   │   └── 1.parquet
│   │   └── tuning
│   │       └── 0.parquet
│   ├── metadata
│   │   ├── codes.parquet
│   │   ├── dataset.json
│   │   └── subject_splits.parquet
│   └── source.yaml
└── simple_example_pkg
    ├── pyproject.toml
    └── src
        └── simple_example_pkg
            ├── __init__.py
            ├── pipelines
            │   └── identity_pipeline.yaml
            └── stages
                ├── __init__.py
                └── identity_stage
                    ├── __init__.py
                    ├── examples
                    │   └── out_data.yaml
                    └── identity_stage.py

```

In this directory, we can see we have a sample dataset (in the [`data/` directory](data)), a sample python package (in
the [`simple_example_pkg/` directory](simple_example_pkg)), and a README file (this file).

In the rest of this document, we'll do the following:

1. We'll describe the source of the static data used in this example.
2. We'll describe the sample package and show a workflow of installing that package and running a pipeline
    using the stage it defines and the pipeline configuration file it provides.

## Static MEDS Data

The static data used in this example is stored in the `data/` directory in the MEDS format. It is generated
from a simple static yaml file, the contents of which are shown below.

> [!NOTE]
> This is the same fictional data as is defined in the
> [MEDS-testing-helpers](https://meds-testing-helpers.readthedocs.io/en/latest/) package, just re-defined from
> source here for clarity.

```yaml
data/train/0.parquet: |-
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

data/train/1.parquet: |-
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

data/tuning/0.parquet: |-
  subject_id,time,code,numeric_value
  754281,,EYE_COLOR//BROWN,
  754281,,HEIGHT,166.22261567137025
  754281,"12/19/1988, 00:00:00",DOB,
  754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
  754281,"01/03/2010, 06:27:59",HR,142.0
  754281,"01/03/2010, 06:27:59",TEMP,99.8
  754281,"01/03/2010, 08:22:13",DISCHARGE,

data/held_out/0.parquet: |-
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

metadata/subject_splits.parquet: |-
  subject_id,split
  239684,train
  1195293,train
  68729,train
  814703,train
  754281,tuning
  1500733,held_out

metadata/codes.parquet: |-
  code,description,parent_codes
  EYE_COLOR//BLUE,Blue Eyes. Less common than brown.,
  EYE_COLOR//BROWN,Brown Eyes. The most common eye color.,
  EYE_COLOR//HAZEL,Hazel eyes. These are uncommon,
  HR,Heart Rate,LOINC/8867-4
  TEMP,Body Temperature,LOINC/8310-5
```

## Simple Example Package

TODO
