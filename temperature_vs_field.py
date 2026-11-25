"""
temperature_vs_field.py -- plan section 0.6: logged thermometer reading below 4.2 K in field.

Tests of "the reading error is a function of |B| only":
  1. master curve T_read(|B|) from all field-bearing files (ramps, full sweeps, gate-sweep holds);
     single-valuedness: rms residual of every file from the master curve;
  2. evenness: T(+|B|) - T(-|B|) inside the same sweep set; hysteresis: |B| rising vs falling;
     ramp vs hold: gate-sweep holds vs ramps at the same |B|; early vs late runs;
  3. shape: low field (|B| < 0.45 T) B^2 vs saturating forms; high field (1.2-5.5 T) linear vs quadratic;
  4. apparent sensor change: Si diode dV = (dV/dT) dT (DT-670: V = 1.58 V, dV/dT = -31.6 mV/K at 4.2 K,
     Lake Shore DT-670 specification); resistive sensor dR/R = |dlnR/dlnT| dT/T for 1.0 and 1.5.

Outputs: tables/thermometer_master_curve.csv, thermometer_tests.csv, thermometer_fits.csv, thermometer_sensor.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from load_data import kind, list_runs, load, sweep_direction
from params import T_BATH, tab_path

SI_DIODE = dict(V=1.58, dVdT=-31.6e-3)      # Lake Shore DT-670 specification table, 4.2 K
BIN = 0.02                                   # T


def collect():
    rows = []
    for n in list_runs():
        df = load(n)
        k = kind(n)
        if k == "Rt":
            df = df[df["T"] < 4.3]          # end of the cool-down, probe in final position
            typ = "cool-down end (B=0)"
        elif k == "RVg":
            typ = "gate-sweep hold" if abs(df["B"]).max() > 0.01 else "B=0 gate points"
        else:
            typ = "field sweep/ramp"
        d = pd.DataFrame({"run": n, "order": df["order"], "type": typ, "B": df["B"], "absB": df["B"].abs(),
                          "T": df["T"], "Vg": df["Vg"]})
        if typ == "field sweep/ramp":
            dB = np.gradient(d["B"].to_numpy())
            d["absB_dir"] = np.where(np.abs(dB) < 1e-5, 0, np.sign(dB * np.sign(d["B"].to_numpy())))   # +1 |B| rising
        else:
            d["absB_dir"] = 0
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def master_curve(allp):
    p = allp[allp["type"] != "gate-sweep hold"]
    edges = np.arange(0, 5.5 + BIN, BIN)
    idx = np.digitize(p["absB"], edges) - 1
    g = p.groupby(idx)
    mc = pd.DataFrame({"absB": g["absB"].mean(), "T_mean": g["T"].mean(), "T_std": g["T"].std(),
                       "n_points": g["T"].size(), "n_runs": g["run"].nunique()}).reset_index(drop=True)
    return mc.sort_values("absB")


def tests(allp, mc):
    rows = []
    Tm = lambda b: np.interp(b, mc["absB"], mc["T_mean"])
    for r, d in allp.groupby("run"):
        res = d["T"] - Tm(d["absB"])
        rows.append(dict(test="residual from master curve", run=r, type=d["type"].iloc[0],
                         absB_range=f"{d['absB'].min():.2f}-{d['absB'].max():.2f}",
                         mean_mK=1e3 * res.mean(), rms_mK=1e3 * np.sqrt(np.mean(res**2)),
                         T_spread_within_run_mK=1e3 * (d["T"].max() - d["T"].min()) if d["type"].iloc[0] == "gate-sweep hold" else np.nan))
    ramps = allp[allp["type"] == "field sweep/ramp"]
    grid = np.arange(0.05, 5.45, 0.05)

    def branch(sel):
        s = sel.groupby(sel["absB"].round(3))["T"].mean()
        return np.interp(grid, s.index, s.values, left=np.nan, right=np.nan)
    # evenness inside the single down-sweep 13_44 and inside the up-ramps
    d13 = ramps[ramps["run"] == "RHLabC-IV-13_44_56"]
    up = ramps[ramps["run"] != "RHLabC-IV-13_44_56"]
    for lab, s in (("13_44 (one sweep)", d13), ("up-ramps", up[up["run"] != "RHLabC-IV-12_49_45"])):
        diff = branch(s[s["B"] > 0]) - branch(s[s["B"] < 0])
        rows.append(dict(test="evenness T(+|B|)-T(-|B|)", run=lab, mean_mK=1e3 * np.nanmean(diff),
                         rms_mK=1e3 * np.sqrt(np.nanmean(diff**2)), max_abs_mK=1e3 * np.nanmax(np.abs(diff))))
    rise = branch(ramps[ramps["absB_dir"] > 0])
    fall = branch(ramps[ramps["absB_dir"] < 0])
    d = rise - fall
    rows.append(dict(test="hysteresis T(|B| rising)-T(|B| falling)", run="all ramps", mean_mK=1e3 * np.nanmean(d),
                     rms_mK=1e3 * np.sqrt(np.nanmean(d**2)), max_abs_mK=1e3 * np.nanmax(np.abs(d))))
    early = branch(ramps[ramps["run"].isin(["RHLabC-IV-12_49_45", "RHLabC-IV-13_44_56"])])
    late = branch(up[up["run"] != "RHLabC-IV-12_49_45"])
    d = late - early
    rows.append(dict(test="drift: late ramps - early sweeps (12_49, 13_44)", run="", mean_mK=1e3 * np.nanmean(d),
                     rms_mK=1e3 * np.sqrt(np.nanmean(d**2)), max_abs_mK=1e3 * np.nanmax(np.abs(d))))
    holds = allp[allp["type"] == "gate-sweep hold"]
    for b, h in holds.groupby(holds["absB"].round(1)):
        rows.append(dict(test="hold vs master (ramp) curve", run=f"|B|={b} T ({h['run'].nunique()} sweeps)",
                         mean_mK=1e3 * (h["T"].mean() - Tm(h["absB"].mean())),
                         T_spread_within_run_mK=1e3 * h.groupby("run")["T"].agg(lambda x: x.max() - x.min()).max(),
                         T_hold=h["T"].mean(), Vg_range=f"{h['Vg'].min():.2f}..{h['Vg'].max():.2f}"))
    return pd.DataFrame(rows)


def fits(mc):
    T0 = mc.loc[mc["absB"] < 0.02, "T_mean"].mean()
    out = []
    lo = mc[(mc["absB"] <= 0.45) & (mc["n_points"] >= 2)]
    x, y = lo["absB"].to_numpy(), lo["T_mean"].to_numpy() - T0
    sy = np.maximum(lo["T_std"].fillna(0.002).to_numpy(), 0.002)
    models = {
        "quadratic  -a B^2": (lambda B, a: -a * B**2, (2.0,)),
        "saturating -A (1-exp(-B^2/B0^2))": (lambda B, A, B0: -A * (1 - np.exp(-(B / B0) ** 2)), (0.4, 0.15)),
        "Lorentzian -A B^2/(B0^2+B^2)": (lambda B, A, B0: -A * B**2 / (B0**2 + B**2), (0.45, 0.1)),
    }
    for name, (f, p0) in models.items():
        p, cov = curve_fit(f, x, y, p0=p0, sigma=sy, maxfev=20000)
        chi2 = np.sum(((y - f(x, *p)) / sy) ** 2)
        out.append(dict(region="low field |B|<=0.45 T (T(0) subtracted)", model=name,
                        params=", ".join(f"{v:.4g}+-{s:.2g}" for v, s in zip(p, np.sqrt(np.diag(cov)))),
                        chi2=chi2, dof=len(x) - len(p), AIC=chi2 + 2 * len(p)))
    hi = mc[(mc["absB"] >= 1.2) & (mc["n_points"] >= 2)]
    x, y = hi["absB"].to_numpy(), hi["T_mean"].to_numpy()
    sy = np.maximum(hi["T_std"].fillna(0.002).to_numpy(), 0.002)
    for name, deg in (("linear  T0 + c1 B", 1), ("quadratic T0 + c1 B + c2 B^2", 2)):
        p, cov = np.polyfit(x, y, deg, w=1 / sy, cov="unscaled")
        chi2 = np.sum(((y - np.polyval(p, x)) / sy) ** 2)
        out.append(dict(region="high field 1.2-5.5 T", model=name,
                        params=", ".join(f"{v:.4g}+-{s:.2g}" for v, s in zip(p, np.sqrt(np.diag(cov)))),
                        chi2=chi2, dof=len(x) - deg - 1, AIC=chi2 + 2 * (deg + 1)))
    return pd.DataFrame(out), T0


def sensor_table(mc):
    rows = []
    for b in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.45):
        T = np.interp(b, mc["absB"], mc["T_mean"])
        dT = T - T_BATH
        r = dict(absB=b, T_read=T, dT_vs_bath=dT, dT_over_T_pct=100 * dT / T_BATH,
                 Si_diode_dV_mV=1e3 * SI_DIODE["dVdT"] * dT,
                 Si_diode_dV_over_V_pct=100 * SI_DIODE["dVdT"] * dT / SI_DIODE["V"])
        for s in (1.0, 1.5):
            r[f"resistive_dR_over_R_pct_dlnRdlnT={s}"] = 100 * s * abs(dT) / T_BATH
        rows.append(r)
    return pd.DataFrame(rows)


def run():
    allp = collect()
    mc = master_curve(allp)
    mc.to_csv(tab_path("thermometer_master_curve.csv"), index=False, float_format="%.5g")
    allp.to_csv(tab_path("thermometer_all_points.csv"), index=False, float_format="%.5g")
    t = tests(allp, mc)
    t.to_csv(tab_path("thermometer_tests.csv"), index=False, float_format="%.4g")
    f, T0 = fits(mc)
    f.to_csv(tab_path("thermometer_fits.csv"), index=False, float_format="%.5g")
    s = sensor_table(mc)
    s.to_csv(tab_path("thermometer_sensor.csv"), index=False, float_format="%.4g")
    return allp, mc, t, f, s, T0


def T_logged(absB):
    """Logged thermometer reading at |B| (master curve) -- used as the systematic alternative to T_BATH."""
    mc = pd.read_csv(tab_path("thermometer_master_curve.csv"))
    return np.interp(np.abs(absB), mc["absB"], mc["T_mean"])


if __name__ == "__main__":
    allp, mc, t, f, s, T0 = run()
    with pd.option_context("display.width", 230, "display.max_colwidth", 60):
        print("T(B=0) =", round(T0, 4))
        print(t[t["test"] != "residual from master curve"].to_string(index=False))
        r = t[t["test"] == "residual from master curve"]
        print("per-run residual rms [mK]: median", round(r["rms_mK"].median(), 1), " max", round(r["rms_mK"].max(), 1),
              r.loc[r["rms_mK"].idxmax(), "run"])
        print(f.to_string(index=False)); print(s.to_string(index=False))
        w = mc[mc["absB"].between(0.6, 2.0)]; i = w["T_mean"].idxmax()
        print("local max of T_read at |B| =", round(mc.loc[i, "absB"], 2), "T =", round(mc.loc[i, "T_mean"], 4))
