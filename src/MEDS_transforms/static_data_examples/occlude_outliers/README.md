# Occlude Outliers Example

If we set the code metadata to the following:

```yaml
metadata/codes.parquet: |-
  code,code/n_occurrences,code/n_subjects,values/n_occurrences,values/sum,values/sum_sqd,description,parent_code
  ,44,4,28,3198.8389005974336,382968.28937288234,,
  ADMISSION//CARDIAC,2,2,0,,,,
  ADMISSION//ORTHOPEDIC,1,1,0,,,,
  ADMISSION//PULMONARY,1,1,0,,,,
  DISCHARGE,4,4,0,,,,
  DOB,4,4,0,,,,
  EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
  EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
  EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
  HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
  HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
  TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
```

Then the below code can compute the means and standard deviations, which will help us determine the cut-offs
for values. If we set the stddev cutoff to 1, this gives the following:

```python
import numpy as np

# We'll set stddev_cutoff to 1 in this test.
CUTOFF = 1

# These are the values/n_occurrences, values/sum, and values/sum_sqd for each of the codes with values:
stats_by_code = {
    "HEIGHT": (4, 656.8389005974336, 108056.12937288235),
    "HR": (12, 1360.5000000000002, 158538.77),
    "TEMP": (12, 1181.4999999999998, 116373.38999999998),
}

means_stds_by_code = {}
for code, (n_occurrences, sum_, sum_sqd) in stats_by_code.items():
    # These types are to match the input schema for the code metadata applied in these tests.
    n_occurrences = np.uint8(n_occurrences)
    sum_ = np.float32(sum_)
    sum_sqd = np.float32(sum_sqd)
    mean = sum_ / n_occurrences
    std = ((sum_sqd / n_occurrences) - mean**2) ** 0.5
    means_stds_by_code[code] = (mean, std)
    print(f"Code: {code}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    print(f"Cut-off: {mean - CUTOFF * std}, {mean + CUTOFF * std}")
```

This returns:

```
Code: HEIGHT
Mean: 164.20973205566406
Std: 7.014064537200555
Cut-off: 157.1956675184635, 171.22379659286463
Code: HR
Mean: 113.375
Std: 18.912240786392818
Cut-off: 94.46275921360719, 132.28724078639283
Code: TEMP
Mean: 98.45833587646484
Std: 1.9334743338704625
Cut-off: 96.52486154259438, 100.39181021033531
```
