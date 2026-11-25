"""
hall.py -- plan section 2a/2b: Hall density, plateau table, calibration factor k of the Hall channel.

  n_H = 1/(e * dR_xy^A/dB)                 low-field slope, 0.15-1.0 T (below the SdH onset)
  k   = n_QH / n_H = R_xy(read)/R_xy(true)  n_QH = Hall-region density from the R_xy plateau fan (sdh.py)
  k_late: the same ratio from the late up-ramps; it applies to the gate sweeps (same epoch), because the
          Hall reading scale drifted by ~1 % while n_QH stayed constant to 0.2 % (ledger ruling)
  true R_xy = R_xy(read)/k ;  true n from the Hall slope = k * n_H

Vg = 0 density comes from the stepped gate sweeps at |B| = 0.5 T (and 1.5 T as a check),
averaging the down and up sweeps (lock-in lag cancels, plan section 3).

Outputs
  tables/hall_density.csv       n_H at Vg = +0.5 and 0 V with fit ranges / systematics
  tables/hall_calibration.csv   k from the fan and from the individual plateaus; comparison values
  tables/hall_plateaus.csv      plateau table (B, R read, R/k, R_K/nu, nu_apparent, sigma_xy)
"""
import numpy as np
import pandas as pd

from params import LOCKIN_PHASE_DEG, R_K, e, h, tab_path
from sdh import load_results
from symmetrize import load_field, load_gate


def slope_fit(B, R):
    A = np.vstack([B, np.ones_like(B)]).T
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    dof = max(len(B) - 2, 1)
    s2 = np.sum((R - A @ coef) ** 2) / dof
    # resolution floor: 5 Ohm steps -> sigma 5/sqrt(12) per point
    s2 = max(s2, (5 / np.sqrt(12)) ** 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    return coef[0], np.sqrt(cov[0, 0]), coef[1], np.sqrt(cov[1, 1])


def n_hall_field(fs, col="Rxy_A", lo=0.15, hi=1.0):
    s = fs[(fs["absB"] >= lo) & (fs["absB"] <= hi) & np.isfinite(fs[col])]
    sl, ssl, ic, sic = slope_fit(s["absB"].to_numpy(), s[col].to_numpy())
    return 1 / (e * sl), ssl / (e * sl**2), ic, sic


def n_hall_gate(gs, absB, vg=0.0):
    out = {}
    for d in ("down", "up"):
        q = gs[(gs["absB"] == absB) & (gs["dir"] == d)].sort_values("Vg")
        # local linear fit of R_xy^A(Vg) over +-30 mV around vg
        s = q[np.abs(q["Vg"] - vg) <= 0.03]
        p = np.polyfit(s["Vg"], s["Rxy_A"], 1)
        R = np.polyval(p, vg)
        out[d] = absB / (e * R)
    return out


def run():
    fs = load_field()
    gs = load_gate()
    sdh = load_results().set_index("pair")
    rows = []
    n0, sn0, ic0, sic0 = n_hall_field(fs)
    rows.append(dict(Vg=0.5, method="R_xy^A slope 0.15-1.0 T, S_down (primary)", n_H_1e15=n0 / 1e15,
                     sn_stat_1e15=sn0 / 1e15, intercept_ohm=ic0, s_intercept_ohm=sic0))
    sysn = []
    for lab, col, lo, hi in (("S_up (late ramps)", "Rxy_A_up", 0.15, 1.0), ("range 0.15-0.5 T", "Rxy_A", 0.15, 0.5),
                             ("range 0.5-1.0 T", "Rxy_A", 0.5, 1.0), ("range 0.1-1.2 T", "Rxy_A", 0.1, 1.2)):
        n, sn, ic, sic = n_hall_field(fs, col, lo, hi)
        rows.append(dict(Vg=0.5, method=f"R_xy^A slope, {lab}", n_H_1e15=n / 1e15, sn_stat_1e15=sn / 1e15,
                         intercept_ohm=ic, s_intercept_ohm=sic))
        sysn.append(n - n0)
    sys_drift = abs(sysn[0])
    sys_range = max(abs(v) for v in sysn[1:])
    for b in (0.5, 1.5):
        g = n_hall_gate(gs, b, 0.0)
        rows.append(dict(Vg=0.0, method=f"gate sweeps |B|={b} T, down", n_H_1e15=g["down"] / 1e15))
        rows.append(dict(Vg=0.0, method=f"gate sweeps |B|={b} T, up", n_H_1e15=g["up"] / 1e15))
        rows.append(dict(Vg=0.0, method=f"gate sweeps |B|={b} T, mean(down,up)",
                         n_H_1e15=(g["down"] + g["up"]) / 2e15, sn_stat_1e15=abs(g["down"] - g["up"]) / 2e15))
    dens = pd.DataFrame(rows)
    dens["sn_sys_1e15"] = np.nan
    dens.loc[0, "sn_sys_1e15"] = np.hypot(sys_drift, sys_range) / 1e15
    dens.to_csv(tab_path("hall_density.csv"), index=False, float_format="%.5g")

    # ---- calibration k = n_QH / n_H  (Vg = +0.5 V, same sweep)
    qh = sdh.loc["Rxy_A (pads 6-3)"]
    xx = sdh.loc["Rxx_S (pads 6-11)"]
    n_qh, sn_qh = qh["n_fan_1e15"] * 1e15, qh["sn_fan_1e15"] * 1e15
    sn_h = np.hypot(sn0, np.hypot(sys_drift, sys_range))
    k = n_qh / n0
    sk = k * np.hypot(sn_qh / n_qh, sn_h / n0)
    pl = pd.read_csv(tab_path("sdh_plateaus_rxy.csv"))
    pl["R_K_over_nu"] = R_K / pl["nu"]
    pl["k_plateau"] = pl["Rxy_read"] / pl["R_K_over_nu"]
    pl["nu_apparent_read"] = R_K / pl["Rxy_read"]
    pl["Rxy_true"] = pl["Rxy_read"] / k
    pl["dev_from_RK_over_nu_pct"] = 100 * (pl["Rxy_true"] / pl["R_K_over_nu"] - 1)
    pl["B_nu_expected"] = n_qh * h / (pl["nu"] * e)
    # sigma_xy in e^2/h: rho_xx/rho_xy < 0.1 for any N_sq > 3 -> sigma_xy = 1/rho_xy to < 1 %
    Rxx_at = np.interp(pl["B_plateau"], fs["absB"], fs["Rxx_S"])
    pl["Rxx_S_read_at_plateau"] = Rxx_at
    pl["sigma_xy_e2h_approx"] = R_K / pl["Rxy_true"]
    pl.to_csv(tab_path("hall_plateaus.csv"), index=False, float_format="%.5g")
    k_pl, sk_pl = pl["k_plateau"].mean(), pl["k_plateau"].std(ddof=1) / np.sqrt(len(pl))
    cos_phase = np.cos(np.radians(LOCKIN_PHASE_DEG))
    late = sdh.loc["Rxy_A late up-ramps (pads 6-3)"]
    n_up = dens.iloc[1]["n_H_1e15"] * 1e15
    k_late = late["n_fan_1e15"] * 1e15 / n_up
    sk_late = k_late * np.hypot(late["sn_fan_1e15"] / late["n_fan_1e15"], np.hypot(dens.iloc[1]["sn_stat_1e15"] * 1e15, sys_range) / n_up)
    cal = pd.DataFrame([
        dict(quantity="k = n_QH/n_H (fan of R_xy plateaus / Hall slope)", value=k, sigma=sk),
        dict(quantity="k_late = same, late up-ramps (epoch of the gate sweeps)", value=k_late, sigma=sk_late),
        dict(quantity="k from plateau readings R_read/(R_K/nu), mean of 4", value=k_pl, sigma=sk_pl),
        dict(quantity="k_xx = n_xx/n_H (pads 6-11 SdH; NOT a calibration, other region)",
             value=xx["n_fan_1e15"] * 1e15 / n0, sigma=np.nan),
        dict(quantity="cos(24 deg) (phase-offset hypothesis)", value=cos_phase, sigma=np.nan),
        dict(quantity="(k - cos24)/sigma_k", value=(k - cos_phase) / sk, sigma=np.nan),
        dict(quantity="implied excitation current I = k * 1 uA [uA] (if the whole factor is the current)",
             value=k, sigma=sk),
        dict(quantity="implied extra phase if k = cos(phi) [deg]", value=np.degrees(np.arccos(min(k, 1))), sigma=np.nan),
        dict(quantity="n_QH [1e15 m^-2]", value=n_qh / 1e15, sigma=sn_qh / 1e15),
        dict(quantity="n_H(+0.5 V) [1e15 m^-2]", value=n0 / 1e15, sigma=sn_h / 1e15),
        dict(quantity="k_late * n_H(Vg=0) = true Hall-region density at Vg=0 [1e15 m^-2]",
             value=k_late * dens.query("Vg==0 and method.str.contains('0.5 T, mean')", engine="python")["n_H_1e15"].iloc[0],
             sigma=np.nan),
    ])
    cal.to_csv(tab_path("hall_calibration.csv"), index=False, float_format="%.5g")
    return dens, cal, pl


def load_k(late=False):
    """k of the early sweeps (12_49/13_44), or k_late of the up-ramps taken between the gate sweeps."""
    c = pd.read_csv(tab_path("hall_calibration.csv"))
    r = c.iloc[1 if late else 0]
    return float(r["value"]), float(r["sigma"])


if __name__ == "__main__":
    dens, cal, pl = run()
    with pd.option_context("display.width", 220, "display.max_colwidth", 90):
        print(dens.to_string(index=False)); print(cal.to_string(index=False)); print(pl.to_string(index=False))
