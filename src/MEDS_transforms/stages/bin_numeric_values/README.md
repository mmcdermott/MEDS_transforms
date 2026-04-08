Discretizes continuous numeric values into categorical bins.

After binning, a measurement like `code: "HR", numeric_value: 85.0` becomes
`code: "HR//[80,90)"` with the original `numeric_value` cleared. This is useful for models that
work with categorical tokens rather than continuous values.

Bin edges can come from two sources:

- **Quantile-based bins** (default) -- computed automatically from `values/quantiles` in the code
    metadata. Run `aggregate_code_metadata` with quantile aggregations first.
- **Custom bins** -- specified via a YAML file with explicit bin boundaries per code, using the
    `custom_bins_filepath` config option.
