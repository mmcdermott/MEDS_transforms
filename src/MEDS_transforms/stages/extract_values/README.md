Parses structured or semi-structured text in MEDS measurements into typed columns.

Many clinical data sources store values as text (e.g., blood pressure as `"120/80"`, temperature as
`"37.2C"`). This stage uses regex-based extraction rules to pull numeric or categorical values out of
`text_value` (or other columns) and write them into `numeric_value` or new code suffixes.

This stage is typically used with the **match-revise** framework so that different extraction rules
apply to different code types.
