# Car Price Prediction

This repository demonstrates a used-car selling-price prediction workflow in
Python. It reads a CSV dataset, derives a brand and model variant from each
listing name, one-hot encodes categorical features, combines those features
with numeric columns, trains a scikit-learn Lasso regression model, and
evaluates it with R-squared.

The project is currently an experimental, console-and-plot script:

- The complete workflow is in [`main.py`](./main.py).
- The dataset is [`car_price_details.csv`](./car_price_details.csv).
- It prints training and test R-squared scores.
- It displays actual-versus-predicted scatter plots for both sets.
- It does not provide a web/API interface or save a trained model.

## Table of contents

- [Current result](#current-result)
- [Dataset](#dataset)
- [Feature engineering decisions](#feature-engineering-decisions)
- [Preprocessing pipeline](#preprocessing-pipeline)
- [Train/test split](#traintest-split)
- [Model choice and evaluation](#model-choice-and-evaluation)
- [Execution flow](#execution-flow)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Visualization behavior](#visualization-behavior)
- [Hurdles and resolutions](#hurdles-and-resolutions)
- [Current limitations](#current-limitations)
- [Troubleshooting](#troubleshooting)
- [Recommended next changes](#recommended-next-changes)

## Current result

The active model is `Lasso` with scikit-learn defaults. On the included data,
the current output is approximately:

| Evaluation set | R-squared |
|---|---:|
| Training set (3,906 rows) | `0.7801426963908437` |
| Test set (434 rows) | `0.7770431496926423` |

The similar training and test scores suggest that the current Lasso model
generalizes reasonably on this holdout for the configured split. These values
are not a universal benchmark: the exact output depends on the dataset, split,
feature engineering, package versions, and unresolved convergence behavior.

The model currently emits a scikit-learn convergence warning:

```text
ConvergenceWarning: Objective did not converge. You might want to increase
the number of iterations, check the scale of the features or consider
increasing regularisation.
Duality gap: 3.423881e+13
tolerance: 1.306e+11
```

The warning means the default `Lasso` stopping criteria were reached before
the optimization converged to the required tolerance. The final R-squared
values are therefore from a fit that is not fully converged.

## Dataset

[`car_price_details.csv`](./car_price_details.csv) contains 4,340 records and
8 source columns:

| Column | Type | Description |
|---|---|---|
| `name` | String | Listing name containing brand and model/variant text |
| `year` | Integer | Manufacturing year |
| `selling_price` | Integer | Target variable |
| `km_driven` | Integer | Reported kilometres driven |
| `fuel` | String | Fuel category |
| `seller_type` | String | Seller category |
| `transmission` | String | Transmission category |
| `owner` | String | Ownership-history category |

The dataset has no source-level null values. There are 763 exact duplicate
rows, and no duplicate-row removal is currently performed.

### Numeric summary

| Column | Minimum | 25th percentile | Median | 75th percentile | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `year` | 1992 | 2011 | 2014 | 2016 | 2020 | 2013.09 |
| `selling_price` | 20,000 | 208,750 | 350,000 | 600,000 | 8,900,000 | 504,127.25 |
| `km_driven` | 1 | 35,000 | 60,000 | 90,000 | 806,599 | 66,215.78 |

### Categorical source values

| Column | Observed values |
|---|---|
| `fuel` | Petrol: 2,123; Diesel: 2,153; CNG: 40; LPG: 23; Electric: 1 |
| `seller_type` | Individual: 3,244; Dealer: 994; Trustmark Dealer: 102 |
| `transmission` | Manual: 3,892; Automatic: 448 |
| `owner` | First Owner: 2,832; Second Owner: 1,106; Third Owner: 304; Fourth & Above Owner: 81; Test Drive Car: 17 |

The source `name` column contains 1,491 unique listing strings.

## Feature engineering decisions

The script does not treat `name` as a single 1,491-category feature. It creates
two derived features instead:

```text
name
  -> brand
  -> model_variant
```

The original `name` column is then dropped.

### Brand extraction

The script uses a fixed, sorted brand list. Sorting by descending name length
is intentional: it allows multi-word brands such as `Land Rover`,
`Mercedes-Benz`, `Aston Martin`, and `Rolls-Royce` to be evaluated before
shorter strings. It also reduces the risk that a short match would incorrectly
capture part of a longer name.

The extraction process:

1. Checks whether the name is missing. Missing values return `Unknown`.
2. Trims the name and lowercases it for comparison.
3. Checks whether the listing starts with a known brand.
4. Returns the canonical brand spelling when a prefix matches.
5. Returns `Other` for an unrecognized prefix.

The known prefixes are:

```text
Land Rover, Mercedes-Benz, Aston Martin, Rolls-Royce, Maruti, Hyundai,
Mahindra, Tata, Honda, Toyota, Ford, Volkswagen, Renault, Chevrolet,
Skoda, Nissan, Datsun, Fiat, BMW, Audi, Jaguar, Volvo, Jeep, Kia, MG,
Isuzu, Force, Porsche, Lexus
```

This produces 26 unique brands in the current full dataset:

```text
Maruti 1280
Hyundai 821
Mahindra 365
Tata 361
Honda 252
Ford 238
Toyota 206
Chevrolet 188
Renault 146
Volkswagen 107
Skoda 68
Nissan 64
Datsun 37
Fiat 37
Mercedes-Benz 35
Audi 60
BMW 39
Jaguar 6
Land Rover 5
Volvo 4
Jeep 3
MG 2
Isuzu 1
Force 1
Kia 1
Other 13
```

Brand counts are useful for understanding encoding and generalization. Rare
training brands are handled by `OneHotEncoder`'s infrequent-category mechanism.

### Model variant extraction

Once a brand is identified, the script removes the brand prefix from `name`
and strips surrounding whitespace:

```python
def extract_model(row):
    name = str(row["name"])
    brand = row["brand"]

    if brand in ["Unknown", "Other"]:
        return name

    return name[len(brand):].strip()
```

For example:

```text
Maruti Wagon R LXI Minor -> brand: Maruti
                            model_variant: Wagon R LXI Minor
```

This extraction keeps model and variant text. It can still be highly sparse:
the full dataset contains 1,491 unique model variants.

For `Unknown` or `Other`, the original name is retained as the model variant
because no reliable brand prefix is available to remove.

## Preprocessing pipeline

The current implementation does not use a scikit-learn `Pipeline` or
`ColumnTransformer`. It performs each preprocessing step explicitly.

### Target and features

`selling_price` is removed from the feature matrix and assigned to `y`.
The original `name` is also removed after deriving `brand` and
`model_variant`.

Final feature groups are:

| Feature group | Columns |
|---|---|
| Numeric | `year`, `km_driven` |
| Categorical | `fuel`, `seller_type`, `transmission`, `owner`, `brand`, `model_variant` |

### One-hot encoding

The categorical columns are passed to:

```python
OneHotEncoder(
    handle_unknown="ignore",
    min_frequency=5,
    sparse_output=False,
)
```

The decisions are:

- **`handle_unknown="ignore"`:** unseen categories in the test set become
  zero-valued for all known columns instead of causing a transformation error.
- **`min_frequency=5`:** categories seen fewer than five times in the training
  fit are combined into an infrequent-category column, reducing sparse-feature
  cardinality.
- **`sparse_output=False`:** encoder output is dense so it can be concatenated
  directly with NumPy numeric arrays.
- No separate fixed category list is supplied.

The encoder is fit only on training categorical data. Test categories are
then transformed with that fitted encoder.

This was a direct solution to the problem of a very large `name`-derived
feature space. Instead of preserving every model variant and unknown listing,
the script preserves the brand/model structure while controlling sparsity.

### Encoder dimensions

The training encoder contains raw category counts of:

| Categorical column | Training categories |
|---|---:|
| `fuel` | 5 |
| `seller_type` | 3 |
| `transmission` | 2 |
| `owner` | 5 |
| `brand` | 26 |
| `model_variant` | 1,416 |

After applying `min_frequency=5` and one-hot encoding, the categorical matrix
has 232 columns. Combining it with `year` and `km_driven` produces 234 final
feature columns.

One fuel category, `Electric`, becomes infrequent because it appears only once
in the source. Rare brands, including `Force`, `Isuzu`, `Jeep`, `Kia`, `MG`,
and `Volvo`, are also grouped when they occur below the training threshold.
More than a thousand model variants are grouped into infrequent categories.

### Numeric/categorical combination

Numerical features are converted to NumPy and concatenated with the dense
encoded array:

```python
x_train_final = np.concatenate(
    [x_train[numerical_features].to_numpy(), x_train_encoded],
    axis=1,
)
```

No scaling or normalization is applied. The numeric columns are therefore
present on their original scales, while one-hot columns contain 0/1 values.
This scale difference is relevant to Lasso optimization and is the primary
known explanation for the reported convergence warning.

### Commented preprocessing alternatives

The file contains commented-out ordinal mappings for `fuel`, `seller_type`,
`transmission`, and `owner`, plus earlier Linear Regression experiments. These
blocks are not executed. The active implementation uses one-hot encoding for
all six categorical features and Lasso regression.

## Train/test split

The split is created only after deriving the brand and model variant:

```python
train_test_split(
    x,
    y,
    test_size=0.1,
    random_state=2,
)
```

| Setting | Value |
|---|---|
| Training rows | 3,906 (90%) |
| Test rows | 434 (10%) |
| Shuffle | Enabled by scikit-learn default |
| Random seed | 2 |
| Stratification | Not used |

The fixed random seed makes the holdout reproducible. It does not ensure
class or price-stratum representation because the split is random and not
stratified.

## Model choice and evaluation

The active model is scikit-learn's Lasso implementation:

```python
lasso_regression_model = Lasso()
lasso_regression_model.fit(x_train_final, y_train)
```

### Why Lasso was selected

The code retains earlier Linear Regression experiments as comments but
currently uses Lasso. The comment states that linear regression performs well
on positively correlated datasets. Lasso is also a linear model, so it keeps
linear interpretability while applying L1 regularization that can reduce
less-useful coefficients to zero.

The project does not tune `alpha`, fit a regularization path, or compare
validation scores across multiple alpha values. `Lasso()` therefore uses the
scikit-learn default `alpha=1.0`, and `max_iter=1000`.

### R-squared calculations

The script calculates separate scores:

```python
training_data_prediction = lasso.predict(x_train_final)
print(metrics.r2_score(y_train, training_data_prediction))

testing_data_prediction = lasso.predict(x_test_final)
print(metrics.r2_score(y_test, testing_data_prediction))
```

R-squared is:

```text
1 - (sum of squared residuals / total sum of squares)
```

| Result | R-squared |
|---|---:|
| Training | `0.7801426963908437` |
| Test | `0.7770431496926423` |

The training result was roughly `78.01%` and the test result roughly
`77.70%`. The close values indicate no obvious large holdout generalization
gap under this split, but they do not establish that the model converged or
that it is production-ready.

The script does not print MAE, MSE, RMSE, MAPE, coefficients, residual values,
or a cross-validation result in its active path.

## Execution flow

1. Import Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, and encoders.
2. Read the CSV from the absolute Windows path
   `D:\car_price_prediction\car_price_details.csv`.
3. Build a sorted brand-prefix list, longest names first.
4. Extract `brand` from `name`.
5. Remove the brand prefix to derive `model_variant`.
6. Drop `name`.
7. Separate `selling_price` into `y`.
8. Create a deterministic 90/10 training/test split.
9. Define numeric and categorical feature lists.
10. Fit the one-hot encoder on training categories only.
11. Transform train and test categories.
12. Concatenate numeric and encoded arrays.
13. Fit the default Lasso model.
14. Predict on the training matrix and print training R-squared.
15. Display a training actual-versus-predicted scatter plot.
16. Predict on the test matrix and print test R-squared.
17. Display a test actual-versus-predicted scatter plot.

## Project structure

```text
car_price_prediction/
|-- main.py                 # Feature engineering, encoding, Lasso, plots
|-- car_price_details.csv   # Used-car source dataset
|-- requirements.txt        # Pinned package versions
|-- .venv/                  # Optional local virtual environment
`-- README.md               # Project documentation
```

The virtual environment is environment-specific and should normally be omitted
from source control.

## Requirements

The repository includes a Python 3.13 virtual environment, but no
`pyproject.toml` or Python-version declaration. A compatible Python 3 release
is required.

Key packages:

| Package | Role |
|---|---|
| pandas | CSV loading and table manipulation |
| NumPy | Dense feature-array operations and concatenation |
| scikit-learn | Split, one-hot encoding, Lasso, and metrics |
| Matplotlib | Actual-versus-predicted plotting |
| Seaborn | Installed visualization dependency |
| SciPy | Numerical backend used by scikit-learn |

The complete pinned package list is [`requirements.txt`](./requirements.txt).

Unused imports in the active source include `Seaborn` and some optional
dependencies. The script imports `LinearRegression`, `matplotlib.pyplot`,
`seaborn`, and `numpy`; Lasso and NumPy are active, while the others are not
required by the final execution path.

## Installation

Run commands from the repository root.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the project

From an activated environment:

```powershell
python main.py
```

Using the repository virtual environment directly:

```powershell
.\.venv\Scripts\python.exe main.py
```

Expected printed output begins with training R-squared and then test
R-squared:

```text
0.7801426963908437
0.7770431496926423
```

The run then opens two Matplotlib windows or saves none, depending on the
configured plotting backend.

### Path problem

The script currently uses:

```python
pd.read_csv("D:\car_price_prediction\car_price_details.csv")
```

This works only in the original Windows directory layout. To run from another
checkout or operating system, replace it with:

```python
from pathlib import Path

data_path = Path(__file__).resolve().parent / "car_price_details.csv"
dataframe = pd.read_csv(data_path)
```

## Visualization behavior

The script creates a scatter plot after each Lasso prediction:

| Plot | X axis | Y axis | Title |
|---|---|---|---|
| Training | Actual training prices | Training predictions | Actual vs Predicted Prices |
| Test | Actual test prices | Test predictions | Actual vs Predicted Prices |

Matplotlib blocks in interactive desktop environments until a plot window is
closed. A server or headless environment may need a noninteractive backend,
for example:

```powershell
$env:MPLBACKEND="Agg"
python main.py
```

With `Agg`, plots still execute but do not appear interactively. The script
currently does not save the plot images to files.

## Hurdles and resolutions

### High-cardinality free-text names

**Problem:** the `name` column has 1,491 distinct values. Encoding it directly
would create a sparse, high-dimensional feature that would be difficult for the
model to generalize.

**Resolution:** derive brand and model-variant features, then one-hot encode
them. `model_variant` still has high cardinality, so `min_frequency=5` groups
rares categories.

### Rare categories

**Problem:** categories such as a one-row `Electric` brand and many model
variants may appear too infrequently to produce reliable coefficients.

**Resolution:** configure `min_frequency=5`, so infrequent training categories
share an infrequent-category column. Test categories unknown to the fitted
encoder are ignored rather than raising an error.

### Large categorical and numeric combined matrix

**Problem:** the encoded categorical features and numeric features use
different representations.

**Resolution:** use a dense one-hot matrix and concatenate it with the numeric
NumPy columns. The implementation is now explicit and no longer requires
ordinal integer mappings for categorical fields.

### Lasso optimizer scale and convergence

**Problem:** `year` and `km_driven` use much larger values than 0/1 encoded
columns. Lasso applies regularization in coefficient space and can therefore
converge slowly or raise a warning without a scale.

**Resolution/status:** the script currently has not implemented scaling.
It records the warning rather than hiding it. The fit produces scores but does
not satisfy the default convergence criterion.

### Absolute CSV location

**Problem:** the original CSV path is tied to `D:\car_price_prediction`.

**Resolution/status:** the problem remains in the source. A repository-relative
`Path(__file__)` approach is documented for a future hardening change.

### Commented experiments

**Problem:** the file contains multiple ordinal-mapping and Linear Regression
blocks that could make the active workflow difficult to identify.

**Resolution/status:** the README identifies the active Lasso path and notes
that commented code is not executed. The source would benefit from cleanup.

## Current limitations

1. **Hard-coded dataset path:** the project cannot run at another Windows path
   or on another OS without source changes.
2. **No data-persistence pipeline:** preprocessing and encoding are repeated
   only inside the script.
3. **No saved encoder or model:** deployment cannot use the existing fitted
   objects without retraining.
4. **No input prediction interface:** the script cannot receive a new listing
   and produce a price.
5. **No cross-validation:** only one random 90/10 split is tested.
6. **No stratification:** uneven price, model, or category distributions can
   affect holdout composition.
7. **No feature scaling:** the Lasso convergence warning remains possible.
8. **Default Lasso regularization:** `alpha` is not tuned.
9. **No pipeline:** data and encoder state cannot be versioned or persisted as
   one artifact.
10. **No duplicate cleanup:** 763 exact duplicate rows remain.
11. **No outlier review:** values as high as 806,599 km and 8.9 million selling
    units are used unchanged.
12. **No unknown-fuel correction:** fuel one-hot encoding handles all categories
    directly; no ordinal fallback is active.
13. **No model diagnostics:** the script does not print coefficients or errors.
14. **No automated tests:** extraction, encoding, split, and plotting have no
    test coverage.
15. **No saved plots:** training and test plots are shown only interactively.
16. **No license file:** reuse terms are not defined.
17. **Experimental commented code:** old mapping and model blocks remain in
    `main.py` and can confuse maintainers.

## Troubleshooting

### `FileNotFoundError`

Check whether `D:\car_price_prediction\car_price_details.csv` exists. Replace
the absolute path with the repository-relative `Path(__file__)` example above
if the project is checked out elsewhere.

### `ConvergenceWarning`

This is likely caused by unscaled numeric features, an insufficient iteration
count, or too-low regularization relative to the feature scale. The source
currently uses default `Lasso()` settings. Before changing code, scale numeric
features or test `max_iter` and `alpha` with cross-validation.

### Plot window does not appear

Check whether the environment is headless or whether Matplotlib is configured
for a noninteractive backend. On Windows, use a desktop backend. In a server or
terminal, set `MPLBACKEND=Agg` or save the plots explicitly.

### Test categories create an encoder error

The current encoder uses `handle_unknown="ignore"`, so unknown test categories
should not raise an encoding error. If errors occur, verify that the encoder
was fit on `x_train[categorical_features]` before transforming test data.

### `requirements.txt` installation fails

Confirm that Python and the pinned package versions are compatible. Recreate
the virtual environment instead of mixing packages from a system environment.

### R-squared differs from the documented values

Confirm the exact CSV, package versions, `random_state=2`, 90/10 split,
brand/model extraction code, encoder settings, and Lasso configuration.
Floating-point results can vary slightly across environments and software
versions.

## Recommended next changes

1. Replace the absolute CSV path with a path relative to `main.py`.
2. Add a scikit-learn `Pipeline` and `ColumnTransformer`.
3. Standardize or normalize `year` and `km_driven`.
4. Tune Lasso `alpha` with cross-validation.
5. Increase `max_iter` only after feature scaling and document the reason.
6. Add Ridge, Elastic Net, and tree-based baseline comparisons.
7. Remove or document 763 duplicate rows.
8. Investigate outliers and extreme selling prices.
9. Add cross-validation and repeated holdout scores.
10. Use a 60/20/20 or 80/10/10 train/validation/test strategy.
11. Report MAE, RMSE, MAPE, R-squared, and coefficient counts.
12. Save the fitted encoder/model together with preprocessing metadata.
13. Add a function or CLI that accepts a new car listing.
14. Save plots to output files and optionally display them.
15. Add unit tests for brand extraction, model extraction, rare categories,
    unknown test categories, and the split.
16. Remove obsolete commented ordinal mappings and Linear Regression blocks.
17. Add a `pyproject.toml` or explicit supported-Python declaration.
