These examples show the time-derived measurement functors:

- **on_raw_static_data** -- enables all three functors (age, time_of_day, timeline_tokens) on the
    full sample data, demonstrating the synthetic events each one produces.
- **in_example_pipeline** -- enables only age computation, as would be typical in a focused pipeline
    that doesn't need time-of-day or timeline tokenization.
