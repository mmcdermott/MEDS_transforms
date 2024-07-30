# Canonical Definitions for MEDS Terminology Elements

In addition to those terms defined in the
[official MEDS Schema](https://github.com/Medical-Event-Data-Standard/meds), we define the following terms for
use in MEDS-transforms:

#### _vocabulary index_ or _code index_

The integer index (starting from 0, which will always correspond to an `"UNK"` vocabulary element) that
uniquely identifies where in the ordered list of vocabulary elements a given element is located. This will be
used as an integral or positional encoding of the vocabulary element for things like embedding matrices,
output layer logit identification, etc.

#### A _measurement_ or _patient measurement_ or _observation_

A single measurable quantity observed about the patient during their care. These observations can take on many
forms, such as observing a diagnostic code being applied to the patient, observing a patient's admission or
transfer from one unit to another, observing a laboratory test result, but always correspond to a single
measureable unit about a single patient. They are encoded in MEDS datasets as a single row in the main MEDS
schema.

#### An _event_ or _patient event_

All observations about a patient that occur at a unique timestamp (within the level of temporal granularity in
the MEDS dataset).

#### An _event index_

For a given patient, when the set of unique timestamps (including the null timestamp, which corresponds to all
static observations) are sorted in ascending order, the event index is the integer index of the timestamp in
this sorted list.

#### A _static measurement_

A _static_ measurement is one that occurs without a source timestamp being recorded in the raw dataset **and**
that can be interpreted as being applicable to the patient at any point in time during their care.

#### A _time-derived measurement_

Measurements that are time-varying, but can be computed deterministically in advance using only the timestamp
at which a measurement occurs and the patient's static or historical data, such as the patient's age or the
season of the year in which a measurement occurs. These measurements warrant special consideration as they are
often valuable covariates but not targets of prediction in ML use cases.

#### A _dynamic measurement_

Measurements that are time-varying and cannot be computed deterministically in advance. The vast majority of
the data in a MEDS dataset will be dynamic measurements.
