"""
checks.py -- numerical sanity checks on the outputs (run last by run_all.py).

Each check states the physical or bookkeeping expectation; a failure raises AssertionError.
"""
import os

import numpy as np
import pandas as pd

from params import FIG_DIR, R_K, e, h, tab_path

T = lambda name: pd.read_csv(tab_path(name))
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


inv = T("inventory.csv")
check(len(inv) == 40 and (inv["rows_kept"] == 0).sum() == 1, "40 files inventoried, exactly one dropped (12_46_02)")
check(inv.loc[inv["file"] == "RHLabC-IV-14_37_02", "rows_kept"].iloc[0] == 154, "14_37_02 trimmed to its 154 ramp rows")
check(inv["leak_max_A"].max() < 1e-9, "gate leakage < 1 nA in every file")

fs = T("field_symmetrized_vg+0.5.csv")
d = fs[fs["src"] == "down"]
check(np.nanmax(np.abs(d["R2_S"])) < 25, "even part of the Hall pair < 25 Ohm in the single sweep (S_down region)")
check(fs["alpha"].between(0.5, 0.65).mean() > 0.95, "alpha within 0.50-0.65 for >95 % of |B| (fixed admixture)")
gs = T("gate_symmetrized.csv")
check(gs.groupby(["absB", "dir"]).size().eq(100).all(), "every gate pair symmetrised on 100 common Vg points")

sd = T("sdh_results.csv").set_index("pair")
for key in sd.index:
    check(abs(sd.loc[key, "fan_intercept"]) < 3 * max(sd.loc[key, "s_fan_intercept"], 0.05),
          f"{key}: Landau-fan intercept consistent with 0 (Berry phase 0, even nu at minima)")
    check(abs(sd.loc[key, "n_fft_1e15"] - sd.loc[key, "n_fan_1e15"]) < 2 * sd.loc[key, "sn_fft_1e15"],
          f"{key}: FFT and fan densities agree")

pl = T("hall_plateaus.csv")
check(pl["k_plateau"].std() / pl["k_plateau"].mean() < 0.005, "plateau readings share one scale factor to < 0.5 %")
cal = T("hall_calibration.csv")
k, sk = cal.iloc[0]["value"], cal.iloc[0]["sigma"]
check(abs(k - pl["k_plateau"].mean()) < 2 * sk, "k from densities agrees with k from plateau readings")

mins = T("gate_sdh_minima.csv")
check((np.abs(mins["nu_continuous"] - mins["nu"]) < 0.45).all(), "every gate-sweep R_xx minimum within 0.45 of an even Hall-region nu")
cap = T("gate_capacitance.csv")
c_h, c_s = cap.iloc[0], cap[cap["method"].str.startswith("SdH minima")].iloc[0]
check(abs(c_h["dn_dVg"] - c_s["dn_dVg"]) < 2 * np.hypot(c_h["s_dn_dVg"], c_s["s_dn_dVg"]), "Hall and SdH dn/dVg agree")

tt = T("thermometer_tests.csv")
holds = tt[tt["test"] == "hold vs master (ramp) curve"]
check((holds["mean_mK"].abs() < 10).all(), "gate-sweep holds lie within 10 mK of the ramp master curve (no ramp heating)")
check((holds["T_spread_within_run_mK"] <= 2).all(), "T constant to <= 2 mK during every fixed-field gate sweep")

rp = T("rt_fit_params.csv").iloc[0]
check(rp["rms_resid_pct"] < 3, "two-channel R(T) fit: rms residual < 3 %")
dr = T("dingle_results.csv").iloc[0]
check(0.05 < dr["tau_q_ps"] < 5, "tau_q in a physical range (0.05-5 ps)")
s = T("summary_vg0.csv").set_index("quantity")
n = s.loc["n used (best), syst = spread of the three", "value"] * 1e15
check(abs(s.loc["E_F", "value"] - np.pi * (1.054571817e-34) ** 2 * n / (0.067 * 9.1093837e-31) / e * 1e3) < 0.01,
      "E_F recomputed from n matches the table")
need = ["fig01_cooldown_RT.pdf", "fig02_symmetrisation.pdf", "fig03_thermometer_vs_field.pdf", "fig04_hall_plateaus.pdf",
        "fig05_sdh_fft_fan.pdf", "fig06_density_vs_gate.pdf", "fig07_gate_sweeps.pdf", "fig08_colourmap_landau_fan.pdf",
        "fig09_sdh_spacing_capacitance.pdf", "fig10_dingle.pdf", "figA9_exponential_fit.pdf"]
check(all(os.path.getsize(os.path.join(FIG_DIR, f)) > 5000 for f in need if os.path.exists(os.path.join(FIG_DIR, f)))
      and all(os.path.exists(os.path.join(FIG_DIR, f)) for f in need), "all 11 figure PDFs exist and are non-trivial")

print(f"\n{len(fails)} failed check(s)")
if fails:
    raise SystemExit(1)
