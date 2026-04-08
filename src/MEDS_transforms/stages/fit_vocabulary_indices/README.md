Assigns a unique integer index (`code/vocab_index`) to each code in the metadata file.

This is a prerequisite for the normalization stage, which replaces string codes with their integer
indices. The current implementation assigns indices in lexicographic order of the code (and any code
modifiers). The mapping is written back to the `codes.parquet` metadata file as a new column.
