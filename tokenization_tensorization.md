# Tokenization and Tensorization for MEDS Models

Supporting appropriate tokenization, tensorization, and ultimately data loading strategies for MEDS models is
a critical component of the MEDS framework. This document outlines the tokenization and tensorization
strategies for MEDS models that are currently supported by MEDS-Transforms and how they should be used.

## Definitions:

- **Normalization**: Here, we use _normalization_ to refer to the process of converting the data from its
  MEDS-dtyped format into a numerical format suitable for eventual tensorization. This includes converting
  string codes into numerical vocabulary indices, and normalizing numerical data to a common scale as needed,
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

| `patient_id` | `time`     | `code`  | `numeric_value` | `text_value` |
| ------------ | ---------- | ------- | --------------- | ------------ |
| 1            |            | STATIC1 |                 |              |
| 1            | 12/1 10:00 | HR      | 88              |              |
| 1            | 12/1 10:00 | RR      | 20              |              |
| 1            | 12/1 10:00 | Temp    | 37.2            |              |
| 1            | 12/1 10:00 | BP      |                 | 120/80       |
| 1            | 12/1 10:04 | O2      | 98              |              |
| 1            | 12/1 10:04 | RR      | 22              |              |
| 1            | 12/5 18:34 | DISCH   |                 |              |
| 2            |            | STATIC2 |                 |              |
| 2            | 3/17 10:00 | HR      | 90              |              |
| 2            | 3/17 10:00 | RR      | 18              |              |
| 2            | 3/17 11:28 | Temp    | 37.0            |              |
| 2            | 3/17 11:28 | BP      |                 | 130/90       |
| 2            | 3/17 11:30 | O2      | 96              |              |
| 2            | 3/17 11:30 | RR      | 20              |              |
| 2            | 3/17 11:30 | Temp    | 37.1            |              |
| 2            | 3/18 01:30 | DISCH   |                 |              |

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
     - BP
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
     - BP
   - Sequence Element 3: 3/17 11:30
     - O2: 96
     - RR: 20
     - Temp: 37.1
   - Sequence Element 4: 3/18 01:30
     - DISCH

Questions / Issues with this strategy:

1. Do we need to order the measurements within an event at all? If so, how do we decide on the order?
2. What about duplicate codes within an event? Is that an issue at all?

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

#### Grouped measurement tokenization (name TBD):

Here, given a MEDS dataset, we want to perform sequence modeling such that each sequence element corresponds
to a _group of measurements_ that have codes satisfying some criteria within a single _event_ for the patient.
In this schema, each sequence element consists of a (1) non-unique timepoint, (2) a variable size collection
of measurements such that measurements are subdivided into groups (which may be only partially observed) in a
pre-specified manner.

E.g., suppose that a patients HR, RR, and O2 are always recorded with the same medical device and, for some
reason, we therefore wanted to ensure that, if they occur in an event, they are all included in the same
sequence element. We could use this tokenization strategy to do this, configuring something like:

```yaml
event_groups:
  - name: Vital Signs
    codes: [HR, RR, O2]
```

and our resulting tokenized data would look like this:

1. Patient 1:
   - Sequence Element 0: `{time: null, measurements: [{code: STATIC1}]}`
   - Sequence Element 1:
     `{time: 12/1 10:00, measurements: [{code: HR, numeric_value: 88}, {code: RR, numeric_value: 20}]`
   - Sequence Element 2: `{time: 12/1 10:00, measurements: [{code: Temp, numeric_value: 37.2}]}`
   - Sequence Element 3: `{time: 12/1 10:00, measurements: [{code: BP}]}`
   - Sequence Element 4:
     `{time: 12/1 10:04, measurements: [{code: O2, numeric_value: 98}, {code: RR, numeric_value: 22}]`
   - Sequence Element 5: `{time: 12/5 18:34, measurements: [{code: DISCH}`
2. Patient 2:
   - Sequence Element 0: `{time: null, measurements: [{code: STATIC2}]}`
   - Sequence Element 1:
     `{time: 3/17 10:00, measurements: [{code: HR, numeric_value: 90}, {code: RR, numeric_value: 18}]}`
   - Sequence Element 2: `{time: 3/17 11:28, measurements: [{code: Temp, numeric_value: 37.0}]}`
   - Sequence Element 3: `{time: 3/17 11:28, measurements: [{code: BP}]}`
   - Sequence Element 4: `{time: 3/17 11:30, measurements: [{code: O2, numeric_value: 96}]}`
   - Sequence Element 5: `{time: 3/17 11:30, measurements: [{code: RR, numeric_value: 20}]}`
   - Sequence Element 6: `{time: 3/17 11:30, measurements: [{code: Temp, numeric_value: 37.1}]}`
   - Sequence Element 7: `{time: 3/18 01:30, measurements: [{code: DISCH}]}`

Questions / Issues with this strategy:

1. How do we decide on the order of the grouped measurements? Do we order them by the first code within a
   group in the event? Do we order them by the order they appear in the `event_groups` configuration all at
   the front of the sequence element?
2. Do we need to order the measurements within a group at all? If so, how do we decide on the order?
3. What about duplicate codes within a group of measurements?

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

Questions / Issues with this strategy:

1. How do we handle null timestamps in this case? Do we have a special "start of sequence" token or something
   to represent the "time interval" from null to the first timepoint? Or what about the "end of sequence"
   token? Do we have a special time interval for going from the last timepoint to a null timepoint?

#### Separate static, dynamic, and/or time-derived measurements:

For some models, it may be useful to separate static measurements from dynamic measurements, or to separate
time-derived measurements from other measurements, such that a batch of data would understand both what the
patient's static data is (which should be used as an input but never computed or generated for autoregressive
models), what dynamic data is (which should be used as an input and may be computed or generated or used as a
label for autoregressive models), and what time-derived data is (which should be used as an input, won't be
computed or used as a label by the model, but may need to be programmatically generated for autoregressive
models, depending on the formulation).

This modifier for tokenization would result in separate sets of elements for each patient, regardless of
whether the tokenization is event, measurement, or grouped-measurement style. In particular:

1. Static measurements would be included in a separate set of elements, with no time information, in a
   single ragged sequence of static observations (codes and values) per-patient. This forms a 2D ragged
   tensor regardless of the tokenization strategy used.
2. Dynamic measurements would be included in a separate set of elements, with time information, with a
   sequence at the granularity and nesting level defined by the tokenization strategy used. For event or
   grouped-measurement tokenization, this would be a 3D ragged tensor. For measurement-level tokenization, a
   2D ragged tensor.
3. Time-derived measurements would be included in a separate set of elements, with time information, with a
   sequence granularity and nested lengths matching the dynamic sequence. Separating out these measurements
   likely only makes sense for event-level tokenization, as they are derived from the time of the event, and
   using them for each measurement independently when many will have the same timepoint is likely not
   useful. Instead, to approximate this for measurement-level tokenization, one would likely need to employ
   time-interval tokens and use grouped measurement tokenization to group the time-interval tokens with
   other time-derived measurements so that all such measurements occur first followed by the dynamic
   measurements.

#### Aggregate measurements into irregularly sized temporal buckets:

With a separate transformation, users may also want to explore tokenization strategies that aggregate data
into differing levels of temporal granularity (e.g., 1-hour buckets, 1-day buckets, etc.), or even into
dynamically defined boundaries (e.g., hospitalizations, etc.) for more complex models. This is not currently
planned as a high-priority feature for MEDS-Transforms, but if it is a use-case of interest to you, don't
hesitate to let us know.

### Text-based tokenization

TODO
