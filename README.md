# Car Price Prediction

This project trains a linear regression model to estimate a used car's selling
price from vehicle attributes. It reads
[`car_price_details.csv`](./car_price_details.csv), performs categorical
encoding and a basic missing-value correction, splits the data into training
and test sets, fits a scikit-learn `LinearRegression` model, and prints its
R-squared score.

The repository is a small, console-based Python example rather than a
production prediction service or web application. It has no command-line
arguments, saved model file, REST API, graphical interface, or interactive
prediction prompt.

## Table of contents

- [Project status and output](#project-status-and-output)
- [Dataset](#dataset)
- [Feature and target definitions](#feature-and-target-definitions)
- [Workflow](#workflow)
- [Encoding and preprocessing decisions](#encoding-and-preprocessing-decisions)
- [Model and evaluation](#model-and-evaluation)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Understanding the printed result](#understanding-the-printed-result)
- [Known limitations and implementation details](#known-limitations-and-implementation-details)
- [Suggested improvements](#suggested-improvements)
- [Troubleshooting](#troubleshooting)

## Project status and output

The current implementation is in [`main.py`](./main.py). Running it on the
included dataset produces an R-squared score of approximately `0.5128204109367687`
(about `51.28%`).

This score is a holdout-test result for the dataset and split configured in
the source file. It should not be treated as a final-quality estimate for
unseen real-world car listings.

## Dataset

The data source is [`car_price_details.csv`](./car_price_details.csv). It
contains **4,340 rows and 8 columns**.

| Column | Source type | Description |
|---|---|---|
| `name` | String | Car name or model description |
| `year` | Integer | Year in which the car was manufactured |
| `selling_price` | Integer | Target value; price of the car in the dataset's currency |
| `km_driven` | Integer | Total kilometres reported as driven |
| `fuel` | String | Fuel type |
| `seller_type` | String | Seller category |
| `transmission` | String | Gearbox category |
| `owner` | String | Ownership history category |

### Numeric summary

| Column | Minimum | 25th percentile | Median | 75th percentile | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `year` | 1992 | 2011 | 2014 | 2016 | 2020 | 2013.09 |
| `selling_price` | 20,000 | 208,750 | 350,000 | 600,000 | 8,900,000 | 504,127.25 |
| `km_driven` | 1 | 35,000 | 60,000 | 90,000 | 806,599 | 66,215.78 |

The dataset has no null values in any of its eight source columns. However,
the source can still contain category-to-code mapping gaps. The implementation
also reports 763 exact duplicate rows, including repeated car listings.

### Categorical values

| Column | Values and row counts |
|---|---|
| `fuel` | Petrol: 2,123; Diesel: 2,153; CNG: 40; LPG: 23; Electric: 1 |
| `seller_type` | Individual: 3,244; Dealer: 994; Trustmark Dealer: 102 |
| `transmission` | Manual: 3,892; Automatic: 448 |
| `owner` | First Owner: 2,832; Second Owner: 1,106; Third Owner: 304; Fourth & Above Owner: 81; Test Drive Car: 17 |

There are 1,491 unique values in `name`. The name is descriptive text rather
than a numerical feature, and the current script drops it instead of applying
text extraction or encoding.

## Feature and target definitions

After loading and preprocessing the data, `selling_price` is used as the target
variable `y`.

The model uses these six features:

1. `year`
2. `km_driven`
3. `fuel`
4. `seller_type`
5. `transmission`
6. `owner`

The original `name` column is dropped before model training.

## Workflow

The complete execution flow is:

1. Import Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and metric helpers.
2. Read the CSV from the hard-coded Windows path
   `D:\car_price_prediction\car_price_details.csv`.
3. Encode the `fuel`, `seller_type`, `transmission`, and `owner` columns as
   integers.
4. Fill unmapped fuel values with the fuel-mode code.
5. Drop the free-text `name` column.
6. Set `selling_price` as the target and use all remaining columns as features.
7. Split the data into 90% training data and 10% test data.
8. Fit `LinearRegression` with scikit-learn.
9. Predict the test set and calculate R-squared.
10. Print only the R-squared value to standard output.

## Encoding and preprocessing decisions

### Fuel encoding

| Original value | Encoded value |
|---|---:|
| Petrol | 0 |
| Diesel | 1 |
| CNG | 2 |
| LPG | 3 |
| Electric | 4 |

The source mapping in [`main.py`](./main.py) currently writes
`Electronic`, not `Electric`. The single row containing `Electric` therefore
does not match the mapping and becomes `NaN`. The code then replaces that
missing value with the mode of the encoded fuel column. Diesel is currently
the mode, so the electric listing is effectively treated as Diesel (`1`) in
the present implementation.

### Seller-type encoding

| Original value | Encoded value |
|---|---:|
| Individual | 0 |
| Dealer | 1 |
| Trustmark Dealer | 2 |

### Transmission encoding

| Original value | Encoded value |
|---|---:|
| Manual | 0 |
| Automatic | 1 |

### Owner encoding

| Original value | Encoded value |
|---|---:|
| First Owner | 0 |
| Second Owner | 1 |
| Third Owner | 2 |
| Fourth & Above Owner | 3 |
| Test Drive Car | 4 |

The mappings are ordinal integers, not learned embeddings. Because the model
uses ordinary linear regression, these values impose an ordered numeric
relationship on nominal categories. For example, a seller code of `2` is
interpreted as twice the numeric level of code `1`, even when such an
interpretation has no real-world meaning.

Additional preprocessing choices:

- No missing-value handling is applied to numeric columns.
- No rows are removed as duplicates or outliers.
- No scaling or normalization is applied.
- No one-hot encoding is applied.
- No feature engineering creates variables such as vehicle age.
- The `name` text column is removed completely.
- The split is not stratified.

## Model and evaluation

The model is scikit-learn's ordinary least-squares `LinearRegression`:

```python
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_train, y_train)
training_data_prediction = linear_regression_model.predict(x_test)
error_score = metrics.r2_score(y_test, training_data_prediction)
```

Despite the variable name `training_data_prediction`, it contains predictions
made on `x_test`, not training predictions.

### Split

| Setting | Value |
|---|---|
| Test size | 10% (`test_size=0.1`) |
| Training size | 90% |
| Shuffle | Enabled by the scikit-learn default |
| Random seed | `2` (`random_state=2`) |
| Stratification | Not configured |

With `random_state=2`, the split is deterministic for the included dataset,
provided the input CSV and package behavior remain unchanged.

### R-squared

R-squared is calculated with `sklearn.metrics.r2_score`:

```text
R-squared = 1 - (sum of squared residuals / total sum of squares)
```

The baseline test score is approximately `0.5128204109367687`.
This means the model explains about `51.28%` of the test-set variance in
`selling_price` under the current preprocessing and split. R-squared can be
negative for a poor model, and a value near 1 indicates a closer fit.

The script does **not** print:

- Mean absolute error
- Mean squared error
- Root mean squared error
- Training R-squared
- Cross-validation score
- Feature coefficients
- Predictions
- Residuals
- A saved model or preprocessing configuration

## Project structure

```text
car_price_prediction/
|-- main.py                 # Data loading, preprocessing, training, and scoring
|-- car_price_details.csv   # 4,340-row used-car dataset
|-- requirements.txt        # Pinned Python package versions
|-- .venv/                  # Optional local virtual environment
`-- README.md               # Project documentation
```

The `.venv/` directory is environment-specific and is not required to run the
code. It should normally be excluded from source control.

## Requirements

The repository currently has a Windows virtual environment using Python 3.13,
but Python 3.13 is not explicitly declared by a project metadata file. A
compatible Python 3 version is required.

`requirements.txt` pinning includes:

| Package | Purpose |
|---|---|
| pandas | CSV loading and tabular data operations |
| NumPy | Numerical array support |
| scikit-learn | Train/test split, linear regression, and R-squared |
| Matplotlib | Plotting support |
| Seaborn | Statistical visualization support |
| scipy | Scientific and numerical dependencies |
| joblib | scikit-learn serialization and dependency support |

The exact dependency list and versions are in
[`requirements.txt`](./requirements.txt).

Some imports in `main.py` are currently unused:

- `numpy`
- `Lasso`
- `matplotlib.pyplot`
- `seaborn`

They do not affect the baseline output and are not dependencies required solely
for an alternative model path in the script.

## Installation

Run commands from the repository root.

### Windows PowerShell

1. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, allow local scripts for the current process:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

3. Upgrade pip and install the pinned dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### Alternative Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To leave a virtual environment, run:

```powershell
deactivate
```

## Running the project

From the repository root:

```powershell
python main.py
```

Or, if using the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py
```

Expected console output format:

```text
0.5128204109367687
```

Small numerical differences are possible if the environment uses different
versions of NumPy or scikit-learn.

### Path configuration

The input path is hard-coded:

```python
dataframe = pd.read_csv("D:\car_price_prediction\car_price_details.csv")
```

Before running on a checkout in another folder or non-Windows system, replace
it with a location-independent path. One reliable approach for this repository
layout is:

```python
from pathlib import Path

data_path = Path(__file__).resolve().parent / "car_price_details.csv"
dataframe = pd.read_csv(data_path)
```

This makes `main.py` find the CSV next to the script regardless of the
current working directory.

## Understanding the printed result

The printed number is R-squared on the 434 test rows (10% of 4,340).
It is calculated against the held-out test split. It does not include
probability intervals or individual prediction confidence.

The test result is influenced by every implementation detail described above,
including:

- Integer category mappings
- The fuel mapping typo
- Retention of duplicate rows
- Exclusion of `name`
- The lack of scaling
- The ordinary linear model
- The fixed 90/10 split
- The fixed random seed

Changing any of these inputs can materially change the score.

## Known limitations and implementation details

1. **Fuel category typo:** the source contains `Electric`, while the code maps
   `Electronic`. The row is corrected to the current fuel mode.
2. **Hard-coded absolute path:** the CSV path depends on `D:\car_price_prediction`.
3. **No model persistence:** rerunning the script retrains the model every time.
4. **No prediction interface:** the script cannot accept a new car as input.
5. **No train/validation/test separation:** the script uses only training and
   holdout-test sets.
6. **No feature scaling:** numeric features have very different magnitudes.
7. **Ordinal category codes:** categories are represented by arbitrary integers.
8. **No text feature extraction:** 1,491 unique `name` values are discarded.
9. **Duplicate listings:** 763 exact duplicate rows are retained.
10. **No outlier treatment:** extreme values such as 806,599 km and
    8.9 million selling-price units remain in the data.
11. **No missing-value validation after preprocessing:** only the fuel mapping
    gap is corrected.
12. **No metrics beyond R-squared:** errors and coefficients are not reported.
13. **No automated tests:** current behavior has no test suite.
14. **No license file:** reuse rights are not defined in this repository.

## Suggested improvements

For a more robust and maintainable implementation, consider:

1. Replacing the absolute CSV path with `Path(__file__).resolve().parent`.
2. Fixing `Electronic` to `Electric`.
3. One-hot encoding nominal fields or using a scikit-learn `ColumnTransformer`.
4. Standardizing numeric features for a reproducible pipeline.
5. Using vehicle age instead of raw manufacturing year or in addition to it.
6. Parsing or encoding the `name` column in a controlled, lower-dimensional way.
7. Removing duplicate rows only after confirming they are true duplicates.
8. Inspecting and documenting outliers before deciding whether to transform or
   remove them.
9. Using a random-seed cross-validation strategy rather than one split.
10. Reporting MAE, MSE, RMSE, R-squared, coefficients, residual plots, and
    prediction intervals.
11. Comparing linear regression with regularized and tree-based regressors.
12. Persisting the fitted pipeline with `joblib`.
13. Validating and accepting new vehicle input before inference.
14. Adding unit tests for encoding, missing values, splitting, and metrics.
15. Adding a script entry point and a configuration file for paths and seeds.

## Troubleshooting

### `FileNotFoundError`

This commonly means the hard-coded Windows path does not exist or the CSV was
moved. Update the path in `main.py` or apply the `Path(__file__)` example above.

### Activation scripts cannot run

Use the process-scoped execution-policy command shown in the installation
section, or run the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

### Missing packages

Reactivate the virtual environment and reinstall:

```powershell
python -m pip install -r requirements.txt
```

### Package versions cannot install

Confirm that the installed Python version supports the pinned versions in
`requirements.txt`. If it does not, create a compatible environment rather
than mixing dependency files or modifying the lock list ad hoc.

### The R-squared value differs

Confirm that you are using the same CSV, package versions, 90/10 split,
`random_state=2`, and preprocessing code. Different input rows or package
implementations can change the final floating-point result slightly.
