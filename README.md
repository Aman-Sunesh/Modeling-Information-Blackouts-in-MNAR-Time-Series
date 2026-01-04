# Modeling Information Blackouts in MNAR Time Series (Traffic Sensors)

This repository studies **traffic-sensor blackouts** (contiguous missing intervals) and compares **MAR vs MNAR** state-space models for:

- **Blackout imputation**: reconstruct values *inside* blackout windows  
- **Post-blackout forecasting**: predict **+1 / +3 / +6** steps after a blackout ends

**Core idea:** treat the missingness mask as an *informative observation channel (MNAR)*, not something to ignore or impute away.

> **TL;DR**: an LDS explains the traffic dynamics; a logistic model explains *whether* each sensor is missing given the latent state.

---

## What’s inside

### Models
- **LOCF baseline** (last observation carried forward)
- **MAR LDS / Kalman**: linear Gaussian state-space model with **masked observations** (missing entries are skipped)
- **MNAR LDS (blackouts-as-signal)**: same LDS + logistic missingness model  
  $$p(m_{t,d}=1 \mid z_t)=\sigma(\phi_d^\top z_t)$$  
  Inference uses **EKF-style updates + RTS smoothing**, and training uses **EM**.

### Evaluation tasks
- **Imputation inside blackouts**: MAE / RMSE  
- **Forecast after blackout**: MAE / RMSE at horizons $k \in \{1,3,6\}$

---

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Choose a workflow

This repo supports two workflows:

#### A) Seattle Loop (primary dataset)
You generate the processed artifacts using the top-level notebooks:

1. `01_load_and_clean.ipynb`  
2. `02_missingness_eda.ipynb`  
3. `03_blackout_detection.ipynb`  
4. `04_build_xt_mt.ipynb`  
5. `05_phi_features.ipynb` *(optional: features for the missingness model)*  
6. `06_evaluation_windows.ipynb`

Then run:
- `main.ipynb` for a minimal end-to-end run, and  
- `experiments.ipynb` for the **full set of plots / tests / baselines** used in the write-up.

> **Note:** raw Seattle Loop data is typically not committed for licensing/size reasons. This repo assumes you have access to it and will place the cleaned panel where `data_interface.py` expects it.

#### B) METR-LA (synthetic evaluation windows)
Utilities for METR-LA live under `metr_la_tests/`.

**Expected input (not committed):**
```
data/
  METR-LA.h5
```

**Step 1 — Convert METR-LA into the generic arrays format**
Run:
- `metr_la_tests/01b_load_and_clean_metr_la.ipynb`

Outputs:
```
metr_la_tests/data_metr_la/
  x_t_nan.npy        # (T, D) float, NaNs are missing
  m_t.npy            # (T, D) uint8, 1 = missing
  timestamps.npy     # (T,) datetime64
  detector_ids.npy   # (D,) strings
```

**Step 2 — Create synthetic evaluation windows**
Run:
- `metr_la_tests/06_build_eval_windows_metr_la.py`

Outputs:
```
metr_la_tests/data_metr_la/
  evaluation_windows.parquet   # impute + forecast rows per window_id
```

**Step 3 — Train/evaluate on METR-LA**
Run:
- `metr_la_tests/main_metr_la.ipynb`

This notebook:
- loads arrays from `metr_la_tests/data_metr_la/`
- masks the evaluation windows to create training data
- trains **MAR** ($\Phi$ fixed / not updated) and **MNAR** ($\Phi$ updated) via EM
- evaluates **imputation + forecasting** on the same blackout windows
- compares against **LOCF**

---

## Evaluation windows manifest

The file `data/evaluation_windows.parquet` (and the METR-LA analogue under `metr_la_tests/data_metr_la/`) contains **one row per evaluation scenario**, with:

- `window_id`: integer ID; all rows with the same `window_id` refer to the same blackout interval on one detector  
- `detector_id`: detector identifier (matches the column name in a wide panel, or an entry in `detector_ids.npy`)  
- `blackout_start`: timestamp of the first time step *inside* the blackout (inclusive)  
- `blackout_end`: timestamp of the last time step *inside* the blackout (inclusive)  
- `test_type`:
  - `"impute"`: evaluate reconstruction over `[blackout_start, blackout_end]`
  - `"forecast"`: evaluate forecasting after the blackout
- `horizon_steps`:
  - for `test_type="impute"`: `0`
  - for `test_type="forecast"`: number of **5-minute** steps after `blackout_end` (typically `1`, `3`, `6`)

### How to apply in experiments

Given (`detector_id`, `blackout_start`, `blackout_end`):

1) **Imputation** (`test_type="impute"`):  
   - Temporarily set `x[t, d] = NaN` for all `t \in [blackout_start, blackout_end]`.
   - Run inference and compare model predictions to the held-out ground truth over that interval.

   **Important:** decide whether your imputation is **acausal** (RTS smoothing over the full sequence) or **online** (filtering only).  
   - Smoothing uses future observations *outside* the blackout and typically improves imputation.
   - Filtering uses only past information and is the correct “online” setting.

2) **Forecasting** (`test_type="forecast"`, `horizon_steps=h>0`):  
   - Condition only on information available up to `blackout_end` (plus the blackout mask, if your model uses it).
   - Predict the target at $t = blackout_end + h$ (in 5-minute steps) and compare to the actual value.

   **No leakage:** forecasting should not use RTS smoothing that conditions on observations after `blackout_end`.

Because evaluation windows are selected from stretches where the target readings exist (inside the blackout and at the forecast horizons), the “ground truth” is available for both tasks, making comparisons across methods reproducible.

---

## Minimal API sketch (scripts / notebooks)

Exact function names may differ slightly across notebooks; conceptually you want:

```python
import data_interface

# Load data in a model-friendly format
x_t, m_t, meta = data_interface.load_panel(...)          # or load_metr_la_panel(...)

# Optional: features for the missingness model (if you use them)
Phi, feat_meta = data_interface.load_missingness_features(...)

# Evaluation windows manifest
eval_windows = data_interface.get_eval_windows(data_dir="data")  # or metr_la_tests/data_metr_la
```

Common optional helper:
- `O_t_list`: list of observed indices per time `t` (useful for speed if you precompute which detectors are observed).

---

## Repository structure (current)

Top-level (typical):
- `data_interface.py` — data loading + evaluation window utilities
- `mnar_blackout_lds.py` — MNAR LDS (EKF/RTS + EM)
- `main.ipynb` — minimal end-to-end run
- `experiments.ipynb` — **all plots + baselines + final tests** (the “main” experiment notebook)
- `01_*.ipynb … 06_*.ipynb` — Seattle Loop processing pipeline

Data artifacts (currently committed in this repo snapshot):
- `data/` — e.g., `evaluation_windows.parquet`, detector/time feature arrays

METR-LA testing:
- `metr_la_tests/`
  - `01b_load_and_clean_metr_la.ipynb`
  - `06_build_eval_windows_metr_la.py`
  - `main_metr_la.ipynb`
  - `data_metr_la/` (generated artifacts; consider excluding if large)

---

## Metrics

### Imputation inside blackout windows
- Report **MAE** / **RMSE** against held-out ground truth values in blackout windows.  
- If windows vary in length, consider reporting both:
  - **window-averaged** (each window equal weight)
  - **length-weighted** (each time step equal weight)

### Post-blackout forecasting
For each blackout end time $b$, evaluate forecasts at horizons $k \in \{1,3,6\}$:

$$\hat{x}_{b+k} = C\,\mu_{b+k \mid b}$$

Report MAE/RMSE at each horizon.
