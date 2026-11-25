"""
inventory.py -- plan section 0.2: one row per raw file, in run order (file-name time).

Output: tables/inventory.csv  (file, order, kind, rows kept, B / Vg / T ranges, sweep variable
and direction, role in the analysis, notes on duplicates).
"""
import numpy as np
import pandas as pd

from load_data import DROP, TRIM_HEAD, clock_seconds, kind, list_runs, load, read_raw, sweep_direction
from params import tab_path


def role(name, df):
    k = kind(name)
    if k == "Rt":
        return "cool-down R(T), B=0, Vg=0 (section 1)"
    if k == "RVg" and np.abs(df["B"]).max() < 0.01:
        return "B=0 gate points Vg=0.43-0.50 V (sections 7, A9)"
    if k == "RVg":
        d = "down" if sweep_direction(df["Vg"]) < 0 else "up"
        return f"gate sweep at B={df['B'].median():+.1f} T, {d} (sections 3, 5)"
    full = df["B"].min() < -1 and df["B"].max() > 1
    if full:
        return "full field sweep through B=0 at Vg=+0.5 V (sections 0.4, 2, 4, 6)"
    return "field ramp between gate sweeps at Vg=+0.5 V (sections 0.4, 0.6, 2, 4)"


def build():
    rows = []
    for name in list_runs(include_dropped=True):
        raw = read_raw(name)
        if name in DROP:
            df, note = raw, "DROPPED: " + DROP[name]
        else:
            df = load(name)
            note = ("TRIMMED: " + TRIM_HEAD[name][1]) if name in TRIM_HEAD else ""
        t = df["t"].to_numpy()
        if np.any(np.diff(t) <= 0):
            note += "; Time not monotonic"
        hold = np.sum(np.diff(t) > 30)
        if hold:
            note += f"; {hold} gap(s) >30 s in Time (hold)"
        sv = "T" if kind(name) == "Rt" else ("Vg" if kind(name) == "RVg" else "B")
        rows.append(dict(
            file=name, clock_s=clock_seconds(name), kind=kind(name),
            rows_raw=len(raw), rows_kept=0 if name in DROP else len(df),
            B_start=df["B"].iloc[0], B_end=df["B"].iloc[-1], B_min=df["B"].min(), B_max=df["B"].max(),
            Vg_start=df["Vg"].iloc[0], Vg_end=df["Vg"].iloc[-1],
            T_min=df["T"].min(), T_max=df["T"].max(),
            sweep_var=sv, sweep_dir=sweep_direction(df[{"T": "T", "Vg": "Vg", "B": "B"}[sv]]),
            leak_max_A=np.abs(df["I_leak"]).max(),
            role="(dropped)" if name in DROP else role(name, df), notes=note.strip("; "),
        ))
    inv = pd.DataFrame(rows)
    inv.insert(1, "order", range(len(inv)))
    inv.to_csv(tab_path("inventory.csv"), index=False, float_format="%.4g")
    return inv


if __name__ == "__main__":
    inv = build()
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(inv[["order", "file", "rows_kept", "B_start", "B_end", "Vg_start", "Vg_end",
                   "T_min", "T_max", "sweep_dir", "leak_max_A", "role", "notes"]].to_string(index=False))
