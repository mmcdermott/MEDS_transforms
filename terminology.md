# Canonical Definitions for MEDS Terminology Elements

#### "vocabulary index" or "code index"

The integer index (starting from 0, which will always correspond to an `"UNK"` vocabulary element) that
uniquely identifies where in the ordered list of vocabulary elements a given element is located. This will be
used as an integral or positional encoding of the vocabulary element for things like embedding matrices,
output layer logit identification, etc.
