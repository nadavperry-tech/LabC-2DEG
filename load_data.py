"""
load_data.py -- read the raw LabVIEW text files in data/ and return clean pandas DataFrames.

Conventions (plan section 0.2):
  * run order = time in the file name (hh_mm_ss); the 'Time' column is only valid inside a file;
  * R1 = SR510 No.1 (pads 6-11, longitudinal pair, contains an odd admixture alpha*R_xy);
  * R2 = SR510 No.2 (pads 6-3, Hall pair);
  * readings are already in Ohm (lock-in voltage / nominal excitation current), resolution 5 Ohm.

Known file problems handled here (see ledger/rulings.md):
  * RHLabC-IV-12_46_02 is an exact prefix of RHLabC-IV-12_49_45 (earlier save)       -> dropped;
  * the first 101 rows of RHLabC-IV-14_37_02 are an exact copy of RVg-IV-14_30_57     -> trimmed.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from params import DATA_DIR

COLUMNS = ["t", "T", "B", "B2", "B3", "Vg", "R1", "R2", "empty1", "empty2", "I_leak"]

DROP = {"RHLabC-IV-12_46_02": "exact prefix of RHLabC-IV-12_49_45 (earlier save of the same run)"}
TRIM_HEAD = {"RHLabC-IV-14_37_02": (101, "rows 0-100 are an exact copy of RVg-IV-14_30_57")}


def run_name(path):
    return os.path.basename(path).replace("-11_11_2026.txt", "")


def clock_seconds(name):
    """hh_mm_ss in the file name -> seconds after midnight (used only for ordering)."""
    hh, mm, ss = map(int, re.search(r"(\d\d)_(\d\d)_(\d\d)$", name).groups())
    return 3600 * hh + 60 * mm + ss


def kind(name):
    return name.split("-IV-")[0]          # 'Rt', 'RVg' or 'RHLabC'


def list_runs(include_dropped=False):
    names = [run_name(p) for p in glob.glob(os.path.join(DATA_DIR, "*.txt"))]
    if not include_dropped:
        names = [n for n in names if n not in DROP]
    return sorted(names, key=clock_seconds)


def read_raw(name):
    path = os.path.join(DATA_DIR, f"{name}-11_11_2026.txt")
    arr = np.genfromtxt(path, skip_header=6, delimiter="\t")
    arr = arr[~np.isnan(arr).all(axis=1)]
    df = pd.DataFrame(arr[:, :len(COLUMNS)], columns=COLUMNS)
    df.attrs["name"] = name
    return df


def load(name):
    """Raw file minus known duplicate rows; adds bookkeeping columns."""
    if name in DROP:
        raise ValueError(f"{name} is dropped: {DROP[name]}")
    df = read_raw(name)
    if name in TRIM_HEAD:
        df = df.iloc[TRIM_HEAD[name][0]:].reset_index(drop=True)
    df["run"] = name
    df["order"] = list_runs().index(name)
    df.attrs["name"] = name
    return df


def sweep_direction(x):
    """+1 if x mainly increases, -1 if it mainly decreases, 0 if constant."""
    d = np.diff(np.asarray(x, float))
    d = d[np.abs(d) > 1e-6]
    if d.size == 0:
        return 0
    return int(np.sign(np.sum(np.sign(d))))


def hampel(y, window=7, n_sigma=4.0):
    """Hampel filter: replace points that deviate > n_sigma*MAD from the running median.
    Returns (filtered copy, boolean mask of replaced points)."""
    y = np.asarray(y, float).copy()
    k = window // 2
    med = pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    mad = pd.Series(np.abs(y - med)).rolling(window, center=True, min_periods=1).median().to_numpy()
    sigma = 1.4826 * np.maximum(mad, 1e-12)
    bad = np.abs(y - med) > n_sigma * np.maximum(sigma, 5.0)   # 5 Ohm = resolution floor
    y[bad] = med[bad]
    return y, bad


def field_runs(vg=0.5):
    """Field ramps/sweeps taken at constant gate voltage vg (RHLabC files)."""
    out = []
    for n in list_runs():
        if kind(n) != "RHLabC":
            continue
        df = load(n)
        if np.allclose(df["Vg"], vg, atol=1e-3):
            out.append(df)
    return out


def gate_runs():
    """Stepped gate sweeps (RVg files with B != 0)."""
    out = []
    for n in list_runs():
        if kind(n) != "RVg":
            continue
        df = load(n)
        if np.abs(df["B"]).max() > 0.1:
            out.append(df)
    return out
