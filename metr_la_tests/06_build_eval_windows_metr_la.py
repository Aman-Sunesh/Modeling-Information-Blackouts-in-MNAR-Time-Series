from pathlib import Path
import numpy as np
import pandas as pd

import data_interface  # for load_metr_la_panel

def build_synthetic_eval_windows_metr_la(
    h5_path: str | Path = "data/METR-LA.h5",
    out_dir: str | Path = "data_metr_la",
    num_base_windows: int = 400,
    blackout_lengths: list[int] | None = None,
    horizons: list[int] | None = None,
    seed: int = 42,
) -> None:
    """
    Create a METR-LA evaluation_windows.parquet with synthetic
    blackouts + forecast horizons, matching the Seattle schema.

    - num_base_windows: number of distinct blackout windows
    - blackout_lengths: list of candidate lengths (in 5-min steps)
    - horizons: forecast horizons (in steps), e.g. [1, 3, 6]
    """
    rng = np.random.default_rng(seed)

    if blackout_lengths is None:
        # e.g. 30, 60, 120 minutes
        blackout_lengths = [6, 12, 24]

    if horizons is None:
        horizons = [1, 3, 6]

    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # 1) Load clean panel (same as conversion)
    df = data_interface.load_metr_la_panel(h5_path)
    timestamps = df.index.to_numpy()
    detectors = df.columns.to_numpy(dtype=str)

    T = len(timestamps)
    D = len(detectors)

    rows = []

    for i in range(num_base_windows):
        # Pick random detector
        d_idx = rng.integers(low=0, high=D)
        det_id = detectors[d_idx]

        # Pick random blackout length
        L = int(rng.choice(blackout_lengths))

        # Ensure we have space for blackout + max horizon inside [0, T)
        max_h = max(horizons)
        max_start = T - (L + max_h + 1)
        if max_start <= 0:
            break

        start_idx = int(rng.integers(low=0, high=max_start))
        end_idx = start_idx + L - 1

        blackout_start = timestamps[start_idx]
        blackout_end = timestamps[end_idx]

        window_id = f"metr_la_w{i:04d}"

        # --- Imputation row ---
        rows.append(
            {
                "window_id": window_id,
                "test_type": "impute",
                "detector_id": det_id,
                "blackout_start": blackout_start,
                "blackout_end": blackout_end,
                "len_steps": L,
                "horizon_steps": np.nan,
            }
        )

        # --- Forecast rows (1, 3, 6-step, etc.) ---
        for h in horizons:
            rows.append(
                {
                    "window_id": window_id,
                    "test_type": "forecast",
                    "detector_id": det_id,
                    "blackout_start": blackout_start,
                    "blackout_end": blackout_end,
                    "len_steps": L,
                    "horizon_steps": h,
                }
            )

    eval_df = pd.DataFrame(rows)
    out_path = out_dir / "evaluation_windows.parquet"
    eval_df.to_parquet(out_path)

    print(f"Saved {len(eval_df)} eval windows to {out_path.resolve()}")

if __name__ == "__main__":
    build_synthetic_eval_windows_metr_la()