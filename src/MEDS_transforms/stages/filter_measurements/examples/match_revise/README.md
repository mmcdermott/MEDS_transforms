Demonstrates how the **match-revise** framework lets you apply different filter criteria to different
subsets of codes, rather than one blanket threshold.

The configuration defines three rules, evaluated in order:

1. **Static measurements** (no timestamp) are never filtered, so demographic codes like `EYE_COLOR` and
    `HEIGHT` are always retained.
2. **Key event codes** (`MEDS_BIRTH`, `MEDS_DEATH`, `ADMISSION`, `DISCHARGE`) are also exempt from
    filtering, regardless of frequency.
3. **All remaining timed codes** must appear at least 10 times across the dataset to be kept.

Because match-revise rules are evaluated top-to-bottom, a row that matches an earlier rule is excluded
from later rules. This lets you express "keep everything except rare lab values" without enumerating
every code you want to keep.
