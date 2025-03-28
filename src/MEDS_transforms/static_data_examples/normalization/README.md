# Normalization test

This computes the normalization value:

```python
import numpy as np

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

vals_by_code_and_subj = {
    "HR": [
        [102.6, 105.1, 113.4, 112.6],
        [109.0, 114.1, 119.8, 112.5, 107.7, 107.5],
        [86.0],
        [170.2],
        [142.0],
        [91.4, 84.4, 90.1],
    ],
    "TEMP": [
        [96.0, 96.2, 95.8, 95.5],
        [100.0, 100.0, 99.9, 99.8, 100.0, 100.4],
        [97.8],
        [100.1],
        [99.8],
        [100.0, 100.3, 100.1],
    ],
    "HEIGHT": [
        [175.271115221764],
        [164.6868838269085],
        [160.3953106166676],
        [156.48559093209357],
        [166.22261567137025],
        [158.60131573580904],
    ],
}

normalized_vals_by_code_and_subj = {}
for code, vals in vals_by_code_and_subj.items():
    mean, std = means_stds_by_code[code]
    normalized_vals_by_code_and_subj[code] = [
        [(np.float32(val) - mean) / std for val in subj_vals] for subj_vals in vals
    ]

for code, normalized_vals in normalized_vals_by_code_and_subj.items():
    print(f"Code: {code}")
    for subj_vals in normalized_vals:
        print([float(x) for x in subj_vals])
```

This returns:

```python
Code: HR
[-0.569736897945404, -0.43754738569259644, 0.001321975840255618, -0.04097883030772209]
[
    -0.23133166134357452,
    0.03833488002419472,
    0.3397272229194641,
    -0.046266332268714905,
    -0.3000703752040863,
    -0.31064537167549133,
]
[-1.4474751949310303]
[3.0046675205230713]
[1.513569951057434]
[-1.1619458198547363, -1.5320764780044556, -1.230684518814087]
Code: TEMP
[-1.2714673280715942, -1.168027639389038, -1.37490713596344, -1.5300706624984741]
[
    0.7973587512969971,
    0.7973587512969971,
    0.745638906955719,
    0.6939190626144409,
    0.7973587512969971,
    1.004242181777954,
]
[-0.3404940366744995]
[0.8490786552429199]
[0.6939190626144409]
[0.7973587512969971, 0.9525222778320312, 0.8490786552429199]
Code: HEIGHT
[1.5770268440246582]
[0.06802856922149658]
[-0.5438239574432373]
[-1.1012336015701294]
[0.28697699308395386]
[-0.7995940446853638]
```
