Reorders measurements within each (subject, timestamp) group according to a configurable priority list.

Many downstream models are sensitive to the order of events at the same timepoint (e.g., an attention
model may treat the first token differently). This stage lets you enforce a deterministic, clinically
meaningful ordering -- for example, placing admission codes before vital signs, and vital signs before
discharge codes.

Each entry in `ordered_code_patterns` is a regex. Measurements are sorted by which pattern matches
their code first; codes that match no pattern are placed last.
