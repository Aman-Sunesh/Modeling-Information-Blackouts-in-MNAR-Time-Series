# Modelling Information Blackouts in MNAR Time Series (Traffic Sensors)

This repo studies **traffic sensor blackouts** (contiguous missing intervals) and compares **MAR vs MNAR** state‑space models for:

- **Blackout imputation**: reconstruct values *inside* blackout windows  
- **Post‑blackout forecasting**: predict **+1 / +3 / +6** steps after a blackout ends

**Core idea:** treat the missingness mask as an *informative observation channel (MNAR)*, not something to ignore or impute away.

---

## What’s inside

### Models
- **LOCF baseline** (last observation carried forward)
- **MAR LDS / Kalman**: linear Gaussian state‑space model with **masked observations** (missing entries are skipped)
- **MNAR LDS (Blackouts‑as‑signal)**: same LDS + logistic missingness model  
  $p(m_{t,d}=1 \mid z_t)=\sigma(\phi_d^\top z_t)$  
  Inference uses **EKF + RTS** and training uses **EM**.

### Evaluation tasks
- **Imputation inside blackouts**: MAE / RMSE
- **Forecast after blackout**: MAE / RMSE at $k \in \{1,3,6\}$
---

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Data options

This repo currently supports two workflows:

#### A) Seattle Loop 2015 (primary dataset)
Place your cleaned Seattle Loop panel under `data/` in whatever filename your `data_interface.py` expects, e.g.
```
data/
  seattle_loop_panel.parquet
```

Then run notebooks in this order (root directory):
1. `01_load_and_clean.ipynb`
2. `02_missingness_eda.ipynb`
3. `03_blackout_detection.ipynb`
4. `04_build_xt_mt.ipynb`
5. `05_phi_features.ipynb` *(optional)*
6. `06_evaluation_windows.ipynb`

Finally:
- run `main.ipynb`.

#### B) METR‑LA (synthetic evaluation windows)
This repo includes a dedicated folder `metr_la_tests/` containing the METR‑LA conversion + synthetic evaluation‑window generation utilities.

**Expected input (not committed):**
```
data/
  METR-LA.h5
```
> The notebook `metr_la_tests/01b_load_and_clean_metr_la.ipynb` reads `data/METR-LA.h5` via `pd.read_hdf(..., key="df")`.

**Step 1 — Convert METR‑LA into the generic arrays format**
Run:
- `metr_la_tests/01b_load_and_clean_metr_la.ipynb`

It produces:
```
metr_la_tests/
  data_metr_la/
    x_t_nan.npy        # (T, D) float, NaNs are missing
    m_t.npy            # (T, D) uint8, 1 = missing
    timestamps.npy     # (T,) datetime64
    detector_ids.npy   # (D,) strings
```

**Step 2 — Create synthetic evaluation windows**
Run:
- `metr_la_tests/06_build_eval_windows_metr_la.py`

It produces:
```
metr_la_tests/
  data_metr_la/
    evaluation_windows.parquet   # impute + forecast rows per window_id
```

**Step 3 — Train/evaluate on METR‑LA**
Run:
- `metr_la_tests/main_metr_la.ipynb`

This notebook:
- loads arrays from `metr_la_tests/data_metr_la/`
- masks the chosen evaluation windows to create training data
- trains **MAR** ($\Phi$ not updated) and **MNAR** ($\Phi$ updated) via EM
- evaluates **imputation + forecasting** on the same blackout windows
- compares against **LOCF**

---

## Repository structure (current)

Top‑level files (typical):
- `data_interface.py` — load panel data + utilities (Seattle + METR‑LA array loader)
- `mnar_blackout_lds.py` — MNAR LDS (EKF + RTS + EM)
- `main.ipynb`, `main(2).ipynb` — Seattle Loop experiments / scratch variants
- `01_*.ipynb … 06_*.ipynb` — Seattle Loop pipeline notebooks

METR‑LA testing:
- `metr_la_tests/` — METR‑LA conversion + synthetic windows + evaluation notebook
  - `01b_load_and_clean_metr_la.ipynb`
  - `06_build_eval_windows_metr_la.py`
  - `main_metr_la.ipynb`
  - `data_metr_la/` *(generated artifacts; consider ignoring in git if large)*

---

## Notes on the MNAR model

We augment the standard LDS:
- Dynamics: $z_t \sim \mathcal{N}(A z_{t-1}, Q)$
- Emissions: $x_t \sim \mathcal{N}(C z_t, R)$

with a **state‑dependent missingness mechanism**:
- Missingness: $p(m_{t,d}=1 \mid z_t)=\sigma(\phi_d^\top z_t)$

During filtering, the model performs an EKF‑style update using:
1) observed speed entries (standard LDS update), and  
2) the missingness mask as a pseudo‑observation (MNAR signal).

---

## Metrics

### Imputation inside blackout windows
- **MAE**, **RMSE** against held‑out ground truth values inside blackout windows  
  (length‑weighted aggregation is commonly used when windows vary in length).

### Post‑blackout forecasting
- For each blackout end time $b$, evaluate forecasts at horizons $k \in \{1,3,6\}$:

$$
\hat{x}_{b+k} = C\ \mu_{b+k \mid b}
$$
- Report MAE/RMSE at each horizon.

---

## Project context (DS‑GA 1018)
This work was developed as part of **DS‑GA 1018: Probabilistic Time Series Analysis** (NYU), focusing on MNAR modelling for structured sensor outages.
