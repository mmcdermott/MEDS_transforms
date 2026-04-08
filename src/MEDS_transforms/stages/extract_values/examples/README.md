Uses `multi_match_and_revise` mode to apply four extraction rules:

1. **Blood pressure systolic** -- for rows with `code: BP`, extracts the first number from `text_value`
    (e.g., `"120/80"` -> `120`) into `numeric_value`, and renames the code to `BP//SYSTOLIC`.
2. **Blood pressure diastolic** -- same source rows, extracts the second number (`"120/80"` -> `80`),
    renaming to `BP//DIASTOLIC`.
3. **Temperature in Celsius** -- for `TEMP` codes where `text_value` matches `"37.2C"`, extracts the
    number into `numeric_value` and renames to `TEMP//C`.
4. **Temperature in Fahrenheit** -- same pattern for `"98.6F"`, renaming to `TEMP//F`.

Because this uses `multi_match_and_revise`, a single input row can match multiple rules (e.g., a BP
reading produces both a systolic and a diastolic output row).
