# ============================================================
# Module: data_interface
# ------------------------------------------------------------
# - Loads the cleaned Seattle loop speed panel (5-min grid)
# - Exposes x_t (speeds) and m_t (missingness indicators)
# - Provides standardized evaluation blackout windows
# - Central place to document shapes / indexing conventions
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Global config
# ------------------------------------------------------------

# Base data directory (all .parquet / .npy live here)
DATA_DIR = Path("data")

# Time step between rows in the panel (minutes)
DT_MINUTES = 5


# ------------------------------------------------------------
# 1. Core panel loader: x_t and m_t
# ------------------------------------------------------------

def load_panel(
    data_dir: str | Path = DATA_DIR,
    return_meta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any] | None]:
    """
    Load the cleaned Seattle Loop panel and return (x_t, m_t)

    Let:
        - T = number of time steps (5-minute intervals over 2015)
        - D = number of detectors

    We represent:
        x_t[t, d] = observed speed at time index t for detector d
                    (float, NaN if missing)
        m_t[t, d] = 1 if the reading is missing at (t, d), else 0

    Parameters
    ----------
    data_dir : str or Path
        Directory containing 'seattle_loop_clean.parquet'
        and/or 'seattle_loop_clean.pkl'.
    return_meta : bool
        If True, also return a metadata dict with timestamps and detector IDs.

    Returns
    -------
    x_t : np.ndarray, shape (T, D), dtype=float
        Speed values on a strict 5-minute grid. NaN denotes missing.
    m_t : np.ndarray, shape (T, D), dtype=np.uint8
        Binary missingness matrix: 1 = missing, 0 = observed.
    meta : dict or None
        Only returned if return_meta=True. Contains:
            - "timestamps": np.ndarray of pandas.Timestamp, shape (T,)
            - "detectors":  np.ndarray of detector IDs (strings), shape (D,)
            - "dt_minutes": int, here always 5
    """
    data_dir = Path(data_dir)

    # Prefer parquet (small, fast); fall back to pickle if needed
    panel_path_parquet = data_dir / "seattle_loop_clean.parquet"
    panel_path_pickle = data_dir / "seattle_loop_clean.pkl"

    if panel_path_parquet.exists():
        wide = pd.read_parquet(panel_path_parquet)
    elif panel_path_pickle.exists():
        wide = pd.read_pickle(panel_path_pickle)
    else:
        raise FileNotFoundError(
            f"Could not find 'seattle_loop_clean.parquet' or "
            f"'seattle_loop_clean.pkl' under {data_dir}."
        )

    # Ensure deterministic column order (whatever is in the file)
    wide = wide.sort_index()  # sort by time
    # wide.columns is already the detector list, in fixed order

    # x_t: numeric values with NaNs for missing entries
    x_t = wide.to_numpy(dtype=float)  # shape (T, D)

    # m_t: binary mask (1 = missing, 0 = observed)
    m_t = np.isnan(x_t).astype(np.uint8)

    if not return_meta:
        return x_t, m_t, None

    timestamps = wide.index.to_numpy()          # shape (T,)
    detectors = wide.columns.to_numpy(dtype=str)  # shape (D,)

    meta = {
        "timestamps": timestamps,
        "detectors": detectors,
        "dt_minutes": DT_MINUTES,
    }

    return x_t, m_t, meta


# ------------------------------------------------------------
# 2. Helper: observed-set indices O_t
# ------------------------------------------------------------

def get_observed_indices(
    m_t: np.ndarray,
) -> List[np.ndarray]:
    """
    Compute the observed-set indices O_t for all time steps.

    Given m_t[t, d] in {0, 1}, we define:
        O_t = { d : m_t[t, d] == 0 }

    This is useful for the EKF code where the speed observation
    block is indexed by the set of observed detectors at time t.

    Parameters
    ----------
    m_t : np.ndarray, shape (T, D), dtype in {0, 1}
        Binary missingness matrix: 1 = missing, 0 = observed.

    Returns
    -------
    O_t_list : list of np.ndarray
        Length-T list; element t is a 1D np.ndarray of detector
        indices d (0 ≤ d < D) that are observed at time t.
    """
    if m_t.ndim != 2:
        raise ValueError(f"m_t should be 2D (T, D), got shape {m_t.shape}")

    T, D = m_t.shape
    O_t_list: List[np.ndarray] = []

    # For each time step t, find indices d where m_t[t, d] == 0
    for t in range(T):
        observed_d = np.where(m_t[t] == 0)[0]
        O_t_list.append(observed_d)

    return O_t_list


# ------------------------------------------------------------
# 3. Evaluation blackout windows
# ------------------------------------------------------------

def get_eval_windows(
    data_dir: str | Path = DATA_DIR,
    as_dataframe: bool = False,
) -> List[Dict[str, Any]] | pd.DataFrame:
    """
    Load the evaluation blackout windows used for imputation/forecasting.

    The corresponding file is created in `06_evaluation_windows.ipynb`
    and stored as 'evaluation_windows.parquet'.

    Each row corresponds to one test case, with columns like:
        - detector_id     : string ID of the detector
        - blackout_start  : pandas.Timestamp (inclusive)
        - blackout_end    : pandas.Timestamp (inclusive)
        - len_steps       : int length in 5-min steps
        - test_type       : "impute" or "forecast"
        - horizon_steps   : (optional) for forecast cases, e.g. 1, 3, 6

    Parameters
    ----------
    data_dir : str or Path
        Directory containing 'evaluation_windows.parquet'.
    as_dataframe : bool
        If True, return a pandas.DataFrame.
        If False, return a list of dicts (records).

    Returns
    -------
    windows : list of dict or pandas.DataFrame
        Evaluation windows ready to loop over in model code.
    """
    data_dir = Path(data_dir)
    path = data_dir / "evaluation_windows.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find 'evaluation_windows.parquet' under {data_dir}."
        )

    df = pd.read_parquet(path)

    if as_dataframe:
        return df

    return df.to_dict(orient="records")


# ------------------------------------------------------------
# 4. Blackout event tables
# ------------------------------------------------------------

def load_detector_blackouts(
    data_dir: str | Path = DATA_DIR,
    as_dataframe: bool = True,
) -> pd.DataFrame | List[Dict[str, Any]]:
    """
    Load per-detector blackout events.

    File is created in `03_blackout_detection.ipynb` as
    'blackout_events_detectors.parquet'.

    This encodes our formal definition of a per-detector blackout:
        - a contiguous run of NaNs in the speed panel
        - length >= MIN_LEN steps (MIN_LEN = 2 ⇒ ≥ 10 minutes)
        - and not touching the first/last time index (structural NA)

    Columns typically include:
        - detector    : string detector ID
        - start       : pandas.Timestamp (inclusive)
        - end         : pandas.Timestamp (inclusive)
        - len_steps   : int, blackout length in 5-min steps
        - len_minutes : int, blackout length in minutes

    Parameters
    ----------
    data_dir : str or Path
        Directory containing 'blackout_events_detectors.parquet'.
    as_dataframe : bool
        If True, return a pandas.DataFrame.
        If False, return a list of dicts.

    Returns
    -------
    events : pandas.DataFrame or list of dict
    """
    data_dir = Path(data_dir)
    path = data_dir / "blackout_events_detectors.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find 'blackout_events_detectors.parquet' under {data_dir}."
        )

    df = pd.read_parquet(path)

    if as_dataframe:
        return df

    return df.to_dict(orient="records")


def load_network_blackouts(
    data_dir: str | Path = DATA_DIR,
    as_dataframe: bool = True,
) -> pd.DataFrame | List[Dict[str, Any]]:
    """
    Load network-level blackout intervals.

    File is created in `03_blackout_detection.ipynb` as
    'blackout_events_network.parquet'.

    These events are defined using the fraction of detectors missing at
    each time step:
        - compute missing_frac_time[t] = (# missing detectors at t) / D
        - define a threshold THRESH (e.g. 0.10 = 10%)
        - mark contiguous runs where missing_frac_time[t] >= THRESH

    Columns typically include:
        - start              : pandas.Timestamp (inclusive)
        - end                : pandas.Timestamp (inclusive)
        - len_steps          : int, number of 5-min steps
        - frac_missing_start : float, missing fraction at start
        - frac_missing_max   : float, maximum missing fraction in window

    Parameters
    ----------
    data_dir : str or Path
        Directory containing 'blackout_events_network.parquet'.
    as_dataframe : bool
        If True, return a pandas.DataFrame.
        If False, return a list of dicts.

    Returns
    -------
    events : pandas.DataFrame or list of dict
    """
    data_dir = Path(data_dir)
    path = data_dir / "blackout_events_network.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find 'blackout_events_network.parquet' under {data_dir}."
        )

    df = pd.read_parquet(path)

    if as_dataframe:
        return df

    return df.to_dict(orient="records")


# ------------------------------------------------------------
# 5. Convenience: single entry point for model inputs
# ------------------------------------------------------------

def load_for_model(
    data_dir: str | Path = DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Dict[str, Any]]:
    """
    Convenience wrapper: load everything an EKF-style model needs.

    This is meant to be a one-liner in model code, e.g.:

        x_t, m_t, O_t_list, meta = load_for_model()

    Parameters
    ----------
    data_dir : str or Path
        Base data directory.

    Returns
    -------
    x_t : np.ndarray, shape (T, D)
        Speed observations (NaN = missing).
    m_t : np.ndarray, shape (T, D)
        Binary missingness mask (1 = missing, 0 = observed).
    O_t_list : list of np.ndarray
        Observed indices per time step.
    meta : dict
        Metadata with timestamps, detector IDs, dt_minutes.
    """
    x_t, m_t, meta = load_panel(data_dir=data_dir, return_meta=True)
    O_t_list = get_observed_indices(m_t)
    return x_t, m_t, O_t_list, meta


def load_missingness_features(
    data_dir: str | Path = DATA_DIR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load the feature tensor Phi[t, d, k] and associated metadata
    for the missingness / dropout model.

    These files are produced in `05_Feature_Engineering_for_Missingness_Model.ipynb`:

        - phi_features.npy      : Phi, shape (T, D, K)
        - time_features.npy     : time-based features, shape (T, F_time)
        - detector_features.npy : detector-based features, shape (D, F_det)
        - detector_ids.npy      : alignment between column index d and detector ID

    Convention for feature channels in Phi[:, :, k]:

        k = 0   : intercept (constant 1.0)
        k = 1–5 : time-based features
                  [sin(hour), cos(hour),
                   is_weekend, is_night, is_net_blackout]
        k = 6–8 : detector-based features
                  [mean_speed_d, std_speed_d, missing_frac_d]

    Parameters
    ----------
    data_dir : str or Path
        Base data directory (where the *.npy files live).

    Returns
    -------
    Phi : np.ndarray, shape (T, D, K), dtype=float32
        Design tensor for p(m_{t,d} = 1 | phi_{t,d}).
    meta : dict
        Metadata with:
            - "time_features"     : np.ndarray, shape (T, F_time)
            - "detector_features" : np.ndarray, shape (D, F_det)
            - "detector_ids"      : np.ndarray, shape (D,)
    """
    data_dir = Path(data_dir)

    phi_path      = data_dir / "phi_features.npy"
    time_path     = data_dir / "time_features.npy"
    det_path      = data_dir / "detector_features.npy"
    det_ids_path  = data_dir / "detector_ids.npy"

    if not phi_path.exists():
        raise FileNotFoundError(
            f"Could not find '{phi_path.name}' under {data_dir}. "
            "Run the feature-engineering notebook (05) first."
        )

    Phi = np.load(phi_path)

    meta: Dict[str, Any] = {}

    if time_path.exists():
        meta["time_features"] = np.load(time_path)
    if det_path.exists():
        meta["detector_features"] = np.load(det_path)
    if det_ids_path.exists():
        meta["detector_ids"] = np.load(det_ids_path, allow_pickle=True)

    return Phi, meta


# Example (for scripts / notebooks):
# x_t, m_t, O_t_list, meta = load_for_model()
# Phi, feat_meta = load_missingness_features()
# eval_windows = get_eval_windows()