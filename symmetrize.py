"""
symmetrize.py -- plan section 0.4 (Onsager symmetrisation) and section 0.1 (admixture alpha).

  R_xx^S(B) = [R1(B) + R1(-B)]/2          R_xy^A(B) = [R2(B) - R2(-B)]/2
  R1^A(B)   = [R1(B) - R1(-B)]/2  (odd admixture in the longitudinal pair)
  alpha     = R1^A / R_xy^A        (0 for an ideal same-edge pair, 1 for a diagonal pair)

Field data (Vg = +0.5 V): +B and -B are paired only inside sets with the same sign of dB/dt
(ledger ruling), which cancels the lock-in delay to first order:
  S_down      : RHLabC 13_44_56 (one continuous sweep +5.34 -> -5.50 T)
  S_up        : ramps 14_02 ... 15_59 (all dB/dt > 0; taken between the gate sweeps)
  S_up_first  : rising part of 12_49_45 (+B) with the -B ramps   (cross-check of drift)
Primary curve = S_down (single sweep, single density); S_up is a later repeat taken after the
gate sweeps (density ~1.6 % higher) and is used only above 5.336 T and as a systematic (dX_set).
Gate data: +B and -B sweeps of the same direction (down/up) at each |B|.

Outputs
  tables/field_symmetrized_vg+0.5.csv  (|B| grid; Rxx_S, Rxy_A, R1_A, R2_S, alpha, T per set + mean)
  tables/gate_symmetrized.csv          (|B|, direction, Vg, Rxx_S, Rxy_A, R1_A, R2_S, alpha, T)
  tables/alpha_summary.csv
"""
import numpy as np
import pandas as pd

from load_data import gate_runs, list_runs, load, sweep_direction
from params import FIELD_GRID_STEP, tab_path

UP_RAMPS = ["RHLabC-IV-14_02_26", "RHLabC-IV-14_14_06", "RHLabC-IV-14_25_08", "RHLabC-IV-14_37_02",
            "RHLabC-IV-14_48_07", "RHLabC-IV-15_01_29", "RHLabC-IV-15_12_31", "RHLabC-IV-15_24_02",
            "RHLabC-IV-15_37_26", "RHLabC-IV-15_48_27", "RHLabC-IV-15_59_14"]
FIRST_UP = "RHLabC-IV-12_49_45"
DOWN = "RHLabC-IV-13_44_56"
B_GRID = np.round(np.arange(0.0, 5.495 + 1e-9, FIELD_GRID_STEP), 4)
MAX_GAP = 0.02      # T; grid points inside a larger data gap are set to NaN
RAMP_START_DROP = 3 # rows dropped at the start of each ramp (lock-in settling, ~6 s)


def rising_part(df):
    """Rows up to the first maximum of B (drops the hold and the start of the down-sweep)."""
    i = int(np.argmax(df["B"].to_numpy()))
    return df.iloc[: i + 1]


def branch(frames, sign):
    """Concatenate frames, keep points with sign(B) = sign, return a DataFrame sorted by |B|
    with duplicate fields averaged."""
    d = pd.concat(frames, ignore_index=True)
    d = d[np.sign(d["B"]) == sign].copy()
    d["absB"] = np.abs(d["B"]).round(4)
    d = d.groupby("absB", as_index=False)[["R1", "R2", "T"]].mean()
    return d.sort_values("absB")


def interp_branch(br, grid=B_GRID):
    x = br["absB"].to_numpy()
    out = {}
    for c in ("R1", "R2", "T"):
        out[c] = np.interp(grid, x, br[c].to_numpy(), left=np.nan, right=np.nan)
    # blank grid points that fall inside data gaps
    idx = np.searchsorted(x, grid)
    lo = np.clip(idx - 1, 0, len(x) - 1)
    hi = np.clip(idx, 0, len(x) - 1)
    gap = (x[hi] - x[lo]) > MAX_GAP
    for c in out:
        out[c][gap] = np.nan
    return out


def symmetrize_sets(pos_frames, neg_frames, grid=B_GRID, label=""):
    p = interp_branch(branch(pos_frames, +1), grid)
    n = interp_branch(branch(neg_frames, -1), grid)
    res = pd.DataFrame({"absB": grid})
    res[f"Rxx_S{label}"] = (p["R1"] + n["R1"]) / 2
    res[f"R1_A{label}"] = (p["R1"] - n["R1"]) / 2
    res[f"Rxy_A{label}"] = (p["R2"] - n["R2"]) / 2
    res[f"R2_S{label}"] = (p["R2"] + n["R2"]) / 2
    res[f"T{label}"] = (p["T"] + n["T"]) / 2
    res[f"R1_pos{label}"], res[f"R1_neg{label}"] = p["R1"], n["R1"]
    res[f"R2_pos{label}"], res[f"R2_neg{label}"] = p["R2"], n["R2"]
    with np.errstate(invalid="ignore", divide="ignore"):
        a = res[f"R1_A{label}"] / res[f"Rxy_A{label}"]
    res[f"alpha{label}"] = np.where(np.abs(res[f"Rxy_A{label}"]) > 100, a, np.nan)
    return res


def field_symmetrized():
    down = load(DOWN)
    assert sweep_direction(down["B"]) == -1
    ramps = []
    for n in UP_RAMPS:
        r = load(n)
        assert sweep_direction(r["B"]) == 1, n
        ramps.append(r.iloc[RAMP_START_DROP:])     # lock-in still settling while the magnet starts
    first = rising_part(load(FIRST_UP))
    s_down = symmetrize_sets([down], [down], label="_down")
    s_up = symmetrize_sets(ramps, ramps, label="_up")
    s_first = symmetrize_sets([first], ramps, label="_upfirst")
    res = s_down.merge(s_up, on="absB").merge(s_first, on="absB")
    # primary = S_down (one continuous sweep, one density); S_up only where S_down has no data
    use_down = np.isfinite(res["Rxx_S_down"]) & np.isfinite(res["Rxy_A_down"])
    res["src"] = np.where(use_down, "down", np.where(np.isfinite(res["Rxx_S_up"]), "up", ""))
    for q in ("Rxx_S", "R1_A", "Rxy_A", "R2_S", "T"):
        res[q] = np.where(use_down, res[f"{q}_down"], res[f"{q}_up"])
        res[f"d{q}_set"] = res[f"{q}_up"] - res[f"{q}_down"]      # late repeat minus primary (drift + 2nd-order lag)
    with np.errstate(invalid="ignore", divide="ignore"):
        res["alpha"] = np.where(np.abs(res["Rxy_A"]) > 100, res["R1_A"] / res["Rxy_A"], np.nan)
    return res


def gate_symmetrized():
    runs = gate_runs()
    rows = []
    fields = sorted({round(abs(r["B"].median()), 1) for r in runs})
    for b in fields:
        for direction, dsign in (("down", -1), ("up", +1)):
            pick = {}
            for r in runs:
                if round(abs(r["B"].median()), 1) == b and sweep_direction(r["Vg"]) == dsign:
                    pick[int(np.sign(r["B"].median()))] = r
            if set(pick) != {+1, -1}:
                continue
            p, n = pick[+1].copy(), pick[-1].copy()
            # repeated set-points (e.g. two readings at the end value) are averaged before pairing
            p = p.assign(Vgr=p["Vg"].round(3)).groupby("Vgr", as_index=False).mean(numeric_only=True)
            n = n.assign(Vgr=n["Vg"].round(3)).groupby("Vgr", as_index=False).mean(numeric_only=True)
            m = p.merge(n, on="Vgr", suffixes=("_p", "_n"))
            out = pd.DataFrame({
                "absB": b, "dir": direction, "Vg": m["Vgr"],
                "Rxx_S": (m["R1_p"] + m["R1_n"]) / 2, "R1_A": (m["R1_p"] - m["R1_n"]) / 2,
                "Rxy_A": (m["R2_p"] - m["R2_n"]) / 2, "R2_S": (m["R2_p"] + m["R2_n"]) / 2,
                "T": (m["T_p"] + m["T_n"]) / 2,
                "Bp": m["B_p"], "Bn": m["B_n"],
                "R1_pos": m["R1_p"], "R1_neg": m["R1_n"], "R2_pos": m["R2_p"], "R2_neg": m["R2_n"],
                "run_pos": pick[+1].attrs["name"], "run_neg": pick[-1].attrs["name"],
            })
            out["alpha"] = out["R1_A"] / out["Rxy_A"]
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def alpha_summary(fs, gs):
    rows = []
    for lab in ("_down", "_up"):
        for lo, hi in ((0.3, 0.5), (0.35, 0.45), (5.3, 5.495), (0.6, 5.495)):
            sel = (fs["absB"] >= lo) & (fs["absB"] <= hi)
            a = fs.loc[sel, f"alpha{lab}"].dropna()
            rows.append(dict(source=f"field sweep{lab or '_mean'}", range=f"|B| {lo}-{hi} T",
                             alpha_mean=a.mean(), alpha_std=a.std(), npts=len(a)))
    for (b, d), g in gs.groupby(["absB", "dir"]):
        a = g["alpha"]
        rows.append(dict(source=f"gate sweep {d}", range=f"|B|={b} T, Vg -0.5..+0.5",
                         alpha_mean=a.mean(), alpha_std=a.std(), npts=len(a),
                         alpha_at_vg_minus=a[g["Vg"] < -0.45].mean(), alpha_at_vg_plus=a[g["Vg"] > 0.45].mean()))
    return pd.DataFrame(rows)


def run():
    fs = field_symmetrized()
    gs = gate_symmetrized()
    fs.to_csv(tab_path("field_symmetrized_vg+0.5.csv"), index=False, float_format="%.6g")
    gs.to_csv(tab_path("gate_symmetrized.csv"), index=False, float_format="%.6g")
    al = alpha_summary(fs, gs)
    al.to_csv(tab_path("alpha_summary.csv"), index=False, float_format="%.4g")
    return fs, gs, al


def load_field():
    return pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))


def load_gate():
    return pd.read_csv(tab_path("gate_symmetrized.csv"))


if __name__ == "__main__":
    fs, gs, al = run()
    with pd.option_context("display.width", 200):
        print(al.to_string(index=False))
        for b in (0.0, 0.1, 0.2, 0.39, 0.5, 1.0, 2.0, 3.0, 4.0, 4.65, 5.0, 5.49):
            r = fs.iloc[np.argmin(abs(fs["absB"] - b))]
            print(f"B={r.absB:.3f}  Rxx_S down/up/first = {r.Rxx_S_down:7.1f} {r.Rxx_S_up:7.1f} {r.Rxx_S_upfirst:7.1f}  "
                  f"Rxy_A d/u/f = {r.Rxy_A_down:7.1f} {r.Rxy_A_up:7.1f} {r.Rxy_A_upfirst:7.1f}  R2_S={r.R2_S:6.1f}  alpha={r.alpha:.3f}  T={r['T']:.3f}")
