# Tokenization and Tensorization for MEDS Models

Supporting appropriate tokenization, tensorization, and ultimately data loading strategies for MEDS models is
a critical component of the MEDS framework. This document outlines the tokenization and tensorization
strategies for MEDS models that are currently supported by MEDS-Transforms and how they should be used.

## Definitions:
- **Normalization**: Here, we use _normalization_ to refer to the process of converting the data from its
  MEDS-dtyped format into a numerical format suitable for eventual tensorization. This includes converting
  string codes into numerical vocabulary indicies, and normalizing numerical data to a common scale as needed,
  etc.
  **TODO**: Should this include computing time-deltas and somehow normalizing those as well???
- **Tokenization**: In this, continuous data, event-stream setting, we use _tokenization_ to refer to the
  process of converting the data from a flat, MEDS-adjacent formatting at the level of individual patient
  measurements, to a format organized per-patient and per-sequence-elements. In essence, this identifies the
  "schema" of the data for the learning framework (in the sense that this format alludes to the format batches
  of data will take on in the input of the model). At a conceptual level, rather than a technical one, we will
  use _tokenization strategies_ to refer the different ways data can be organized into sequence elements for
  final modeling. Note this may also include separating static, dynamic, and/or time-derived measurements from
  one another, depending on how they will be used in the model.
- **Tensorization**: In this, continuous data, event-stream setting, we use _tensorization_ to refer to the
  process of converting the tokenized data from a data-frame format into a format suitable for rapid,
  efficient ingestion into deep learning modeling tensors for use with PyTorch datasets. This process does not
  change the conceptual scale of the data, but it may change the format of the data to be more easily
  retrieved by the PyTorch dataset in a scalable manner.
- **Event** vs. **Measurement**/**Observation**: A measurement/observation is a single observation about a
  patient in time -- e.g., a single lab result, a single vital sign, etc. This will be technically realized as
  a combination of a timestamp, code, and a possible `numeric_value`, possibly with some other limited
  columns. An _event_ is the collection of all measurements that take place for a patient at a single point in
  time.

## Conceptual "tokenization strategies" we need to support:
Sample data (in non-normalized form):

`patient_id` | `time`     | `code`  | `numeric_value` | `text_value`
-------------|------------|---------|-----------------|-------------
1            |            | STATIC1 |                 |
1            | 12/1 10:00 | HR      | 88              |
1            | 12/1 10:00 | RR      | 20              |
1            | 12/1 10:00 | Temp    | 37.2            |
1            | 12/1 10:00 | BP      |                 | 120/80
1            | 12/1 10:04 | O2      | 98              |
1            | 12/1 10:04 | RR      | 22              |
1            | 12/5 18:34 | DISCH   |                 |
2            |            | STATIC2 |                 |
2            | 3/17 10:00 | HR      | 90              |
2            | 3/17 10:00 | RR      | 18              |
2            | 3/17 11:28 | Temp    | 37.0            |
2            | 3/17 11:28 | BP      |                 | 130/90
2            | 3/17 11:30 | O2      | 96              |
2            | 3/17 11:30 | RR      | 20              |
2            | 3/17 11:30 | Temp    | 37.1            |
2            | 3/18 01:30 | DISCH   |                 |

### Tokenization Core Strategies:
#### Event-level tokenization:
Here, given a MEDS dataset, we want to perform sequence modeling such that each sequence element corresponds
to an _event_ (unique timepoint) for the patient. Each sequence element thus consists of a (1) unique
timepoint and (2) variable size collection of measurements that occur at that timepoint.

Under this tokenization strategy _without modification_, our sample data would be represented as:
  1. Patient 1:
     - Sequence Element 0: NO TIMESTAMP
       - STATIC1
     - Sequence Element 1: 12/1 10:00
       - HR: 88
       - RR: 20
       - Temp: 37.2
       - BP: 120/80
     - Sequence Element 2: 12/1 10:04
       - O2: 98
       - RR: 22
     - Sequence Element 3: 12/5 18:34
       - DISCH
  2. Patient 2:
    - Sequence Element 0: NO TIMESTAMP
      - STATIC2
    - Sequence Element 1: 3/17 10:00
      - HR: 90
      - RR: 18
    - Sequence Element 2: 3/17 11:28
      - Temp: 37.0
      - BP: 130/90
    - Sequence Element 3: 3/17 11:30
      - O2: 96
      - RR: 20
      - Temp: 37.1
    - Sequence Element 4: 3/18 01:30
      - DISCH

#### Measurement-level tokenization:
Here, given a MEDS dataset, we want to perform sequence modeling such that each sequence element corresponds
to a _measurement_ for the patient. Each sequence element thus consists of a (1) non-unique timepoint, (2) a
code, and (3) a `numeric_value` (which may be null). Note here that we are, by design, excluding `text_value`
from a field in the tokenized view, as it is not in a naively normalizable format.

Under this tokenization strategy _without modification_, our sample data would be represented as:
  1. Patient 1:
     - Sequence Element 0: `{time: null, code: STATIC1}`
     - Sequence Element 1: `{time: 12/1 10:00, code: HR, numeric_value: 88}`
     - Sequence Element 2: `{time: 12/1 10:00, code: RR, numeric_value: 20}`
     - Sequence Element 3: `{time: 12/1 10:00, code: Temp, numeric_value: 37.2}`
     - Sequence Element 4: `{time: 12/1 10:00, code: BP}`
     - Sequence Element 5: `{time: 12/1 10:04, code: O2, numeric_value: 98}`
     - Sequence Element 6: `{time: 12/1 10:04, code: RR, numeric_value: 22}`
     - Sequence Element 7: `{time: 12/5 18:34, code: DISCH}`
  2. Patient 2:
    - Sequence Element 0: `{time: null, code: STATIC2}`
    - Sequence Element 1: `{time: 3/17 10:00, code: HR, numeric_value: 90}`
    - Sequence Element 2: `{time: 3/17 10:00, code: RR, numeric_value: 18}`
    - Sequence Element 3: `{time: 3/17 11:28, code: Temp, numeric_value: 37.0}`
    - Sequence Element 4: `{time: 3/17 11:28, code: BP}`
    - Sequence Element 5: `{time: 3/17 11:30, code: O2, numeric_value: 96}`
    - Sequence Element 6: `{time: 3/17 11:30, code: RR, numeric_value: 20}`
    - Sequence Element 7: `{time: 3/17 11:30, code: Temp, numeric_value: 37.1}`
    - Sequence Element 8: `{time: 3/18 01:30, code: DISCH}`

### Tokenization Modifiers:
Here, we discuss some natural ways that tokenization can be modified to better suit the needs of the model.
Often, these modifications are not things that happen during tokenization itself, technically, but may happen
in advance of tokenization as more traditional data processing steps.

#### Add time-interval tokens:
For some models, it may be useful to add a token to the sequence elements that represents the time interval
between any two sequence elements, or between any two sequence elements that do not occur at the same time.
This is most useful for measurement-level tokenization, as all sequence elements occur at unique times in
event level tokenization and those time deltas can thus be naturally leveraged directly (via, e.g., temporal
position embeddings).

Using time interval tokens with measurement-level tokenization, you may result in a tokenization strategy like
this:
  1. Patient 1:
     - Sequence Element 0: `{time: null, code: STATIC1}`
     - Sequence Element 1: `{time: 12/1 10:00, code: HR, numeric_value: 88}`
     - Sequence Element 2: `{time: 12/1 10:00, code: RR, numeric_value: 20}`
     - Sequence Element 3: `{time: 12/1 10:00, code: Temp, numeric_value: 37.2}`
     - Sequence Element 4: `{time: 12/1 10:00, code: BP}`
     - Sequence Element 5: `{time: 12/1 10:04, code: TIME_INTERVAL//MIN, numeric_value: 4}`
     - Sequence Element 6: `{time: 12/1 10:04, code: O2, numeric_value: 98}`
     - Sequence Element 7: `{time: 12/1 10:04, code: RR, numeric_value: 22}`
     - Sequence Element 8: `{time: 12/5 18:34, code: TIME_INTERVAL//MIN, numeric_value: 514}`
     - Sequence Element 9: `{time: 12/5 18:34, code: DISCH}`
  2. Patient 2:
    - Sequence Element 0: `{time: null, code: STATIC2}`
    - Sequence Element 1: `{time: 3/17 10:00, code: HR, numeric_value: 90}`
    - Sequence Element 2: `{time: 3/17 10:00, code: RR, numeric_value: 18}`
    - Sequence Element 3: `{time: 3/17 10:00, code: TIME_INTERVAL//MIN, numeric_value: 88}`
    - Sequence Element 4: `{time: 3/17 11:28, code: Temp, numeric_value: 37.0}`
    - Sequence Element 5: `{time: 3/17 11:28, code: BP}`
    - Sequence Element 6: `{time: 3/17 11:30, code: TIME_INTERVAL//MIN, numeric_value: 2}`
    - Sequence Element 7: `{time: 3/17 11:30, code: O2, numeric_value: 96}`
    - Sequence Element 8: `{time: 3/17 11:30, code: RR, numeric_value: 20}`
    - Sequence Element 9: `{time: 3/17 11:30, code: Temp, numeric_value: 37.1}`
    - Sequence Element 10: `{time: 3/18 01:30, code: TIME_INTERVAL//MIN, numeric_value: 840}`
    - Sequence Element 11: `{time: 3/18 01:30, code: DISCH}`

This process is an excellent example of why these extra tokens should be added via dedicated data
pre-processing steps rather than as part of the tokenization process itself. A user using these tokens must
consider things like:
  1. Ensuring that the time interval codes are included in the vocabulary.
  2. Ensuring that the time interval numeric values are correctly normalized, which may require aggregation
     alongside other codes on the train data.

Often, when these tokens are added, in the ultimate, tensorized data, elements such as the timestamp of the
sequence elements are dropped (as they are captured via the time interval tokens directly).

#### Separate static, dynamic, and/or time-derived measurements:

#### Aggregate measurements into temporal buckets:

### Text-based tokenization
TODO
