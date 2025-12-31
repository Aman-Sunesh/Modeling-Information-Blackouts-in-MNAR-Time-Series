import h5py
import pandas as pd
import numpy as np
from pathlib import Path

h5_path = Path("data/raw/METR-LA/METR-LA.h5")
with h5py.File(h5_path, "r") as f:
    print("Keys in H5:", list(f.keys()))

with h5py.File(h5_path, "r") as f:
    # this reads the bytes that contain the pickled DataFrame
    buf = io.BytesIO(f["df"][:])
    df = pickle.load(buf)

df.head(), df.shape, df.index[:5], df.columns[:5]