"""
dingle.py -- plan section 6 (Dingle plot, tau_q) and section 2c (spin splitting / g* bound).

Lifshitz-Kosevich, spin-degenerate, small amplitude:
   dR / R_ref = 4 X/sinh(X) exp(-pi / (omega_c tau_q)),   X = 2 pi^2 k_B T / (hbar omega_c),  omega_c = e B / m*
Dingle plot: ln[ (dR/R_ref) sinh(X) / (4 X) ]  vs 1/B  ->  slope = -pi m* / (e tau_q).

Data: R_xx^S at Vg = +0.5 V (primary field sweep), oscillation of the pads 6-11 path (n_xx region).
Amplitude dR = half peak-to-peak between consecutive extrema, placed at the 1/B midpoint.
Reference resistance: R_ref = R_xx(B=0) (primary; the oscillating 2DEG has no classical MR of its own)
and R_ref = R_bg(B) (polynomial background, systematic). Only points with dR/R_ref <= 0.5 (small-amplitude LK).
Temperature: T = 4.2 K (primary), logged T(B) (systematic), RSS profile over fixed T (degeneracy check).
The intercept at 1/B -> 0 should be 0; a negative intercept measures the fraction of the 6-11 path
that oscillates (ledger: two regions in series).

g* (2c): no odd-nu minima are resolved up to 5.33 T -> g* mu_B B < Gamma = hbar/(2 tau_q) at the
highest odd nu in range -> upper bound on g*.

Outputs: tables/dingle_points.csv, dingle_results.csv, dingle_T_profile.csv, spin_splitting.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from params import M_STAR, T_BATH, e, h, hbar, kB, mu_B, tab_path
from sdh import B_MAX, detrend
from symmetrize import load_field
from temperature_vs_field import T_logged

AMP_MAX = 0.5
B_LO = 1.15


def X(T, B):
    return 2 * np.pi**2 * kB * T * M_STAR / (hbar * e * B)


def amplitudes(fs):
    s = fs[(fs["absB"] >= B_LO) & (fs["absB"] <= B_MAX) & np.isfinite(fs["Rxx_S"])]
    B, R = s["absB"].to_numpy(), s["Rxx_S"].to_numpy()
    dR, bg = detrend(B, R, 3)
    sm = savgol_filter(dR, 11, 3)
    ext = np.sort(np.concatenate([find_peaks(sm, prominence=8)[0], find_peaks(-sm, prominence=8)[0]]))
    rows = []
    for i, j in zip(ext[:-1], ext[1:]):
        if np.sign(sm[i]) == np.sign(sm[j]):
            continue
        xm = 0.5 * (1 / B[i] + 1 / B[j])
        Bm = 1 / xm
        rows.append(dict(B=Bm, invB=xm, dR=abs(sm[i] - sm[j]) / 2, R_bg=np.interp(Bm, B, bg)))
    return pd.DataFrame(rows)


def dingle_fit(pts, Rref, T):
    y = np.log(pts["dR"] / Rref * np.sinh(X(T, pts["B"])) / (4 * X(T, pts["B"])))
    x = pts["invB"].to_numpy()
    p, cov = np.polyfit(x, y, 1, cov=True)
    tau = -np.pi * M_STAR / (e * p[0])
    stau = tau * np.sqrt(cov[0, 0]) / abs(p[0])
    return tau, stau, p[1], np.sqrt(cov[1, 1]), x, y


def t_profile(pts, R0, temps=(2.0, 3.0, 3.12, 3.5, 3.73, 4.2, 5.0, 6.0, 8.0)):
    """tau_q and residual sum of squares of the Dingle fit at fixed trial temperatures.
    A flat RSS(T) means the amplitude data alone cannot fix T (tau_q and T are degenerate)."""
    rows = []
    for T in temps:
        tau, stau, ic, sic, x, y = dingle_fit(pts, R0, T)
        p = np.polyfit(x, y, 1)
        rows.append(dict(T_K=T, tau_q_ps=tau * 1e12, s_tau_q_ps=stau * 1e12, intercept=ic,
                         rss=float(np.sum((y - np.polyval(p, x)) ** 2))))
    return pd.DataFrame(rows)


def run():
    fs = load_field()
    pts = amplitudes(fs)
    lo = fs[(fs["absB"] > 0.02) & (fs["absB"] < 0.1) & np.isfinite(fs["Rxx_S"])]
    R0 = np.polyval(np.polyfit(lo["absB"] ** 2, lo["Rxx_S"], 1), 0.0)     # B -> 0 extrapolation of R_xx^S
    pts["dR_over_R0"] = pts["dR"] / R0
    pts["dR_over_Rbg"] = pts["dR"] / pts["R_bg"]
    pts["T_logged"] = T_logged(pts["B"])
    pts["X_4p2K"] = X(T_BATH, pts["B"])
    use = pts[pts["dR_over_R0"] <= AMP_MAX].copy()
    pts["used"] = pts.index.isin(use.index)
    rows = []
    for lab, rref, T in (("primary: R_ref = R_xx(0), T = 4.2 K", R0, T_BATH),
                         ("R_ref = R_xx(0), T = logged T(B)", R0, use["T_logged"].to_numpy()),
                         ("R_ref = R_bg(B), T = 4.2 K", use["R_bg"].to_numpy(), T_BATH),
                         ("R_ref = R_xx(0), T = 4.2 K, all points (incl. dR/R0 > 0.5)", None, T_BATH)):
        P = pts if rref is None else use
        rr = R0 if rref is None else rref
        tau, stau, ic, sic, x, y = dingle_fit(P, rr, T)
        rows.append(dict(fit=lab, tau_q_ps=tau * 1e12, s_tau_q_ps=stau * 1e12, intercept=ic, s_intercept=sic,
                         oscillating_fraction=np.exp(ic), n_points=len(P), B_range=f"{P['B'].min():.2f}-{P['B'].max():.2f} T"))
    prof = t_profile(use, R0)
    prof.to_csv(tab_path("dingle_T_profile.csv"), index=False, float_format="%.5g")
    res = pd.DataFrame(rows)
    res["Gamma_meV"] = hbar / (2 * res["tau_q_ps"] * 1e-12) / e * 1e3
    res["R0_ohm"] = R0
    res.to_csv(tab_path("dingle_results.csv"), index=False, float_format="%.5g")
    pts.to_csv(tab_path("dingle_points.csv"), index=False, float_format="%.5g")

    # ---- 2c spin splitting
    tau = res.iloc[0]["tau_q_ps"] * 1e-12
    sdh = pd.read_csv(tab_path("sdh_results.csv")).set_index("pair")
    out = []
    for lab, n in (("pads 6-11 region", sdh.loc["Rxx_S (pads 6-11)", "n_fan_1e15"] * 1e15),
                   ("Hall-cross region", sdh.loc["Rxy_A (pads 6-3)", "n_fan_1e15"] * 1e15)):
        nu_min = h * n / (e * 5.33)
        nu_odd = int(np.ceil(nu_min)) | 1
        B_odd = h * n / (e * nu_odd)
        Gamma = hbar / (2 * tau)
        g_lw = Gamma / (mu_B * B_odd)
        g_th = 3.5 * kB * T_BATH / (mu_B * B_odd)
        out.append(dict(region=lab, n_1e15=n / 1e15, highest_odd_nu_in_range=nu_odd, B_at_odd_nu_T=B_odd,
                        resolved="no", Gamma_meV=Gamma / e * 1e3, g_bound_level_width=g_lw,
                        g_bound_thermal_3p5kT=g_th, g_upper_bound=max(g_lw, g_th),
                        bare_spin_gap_meV_g0p44=0.44 * mu_B * B_odd / e * 1e3,
                        cyclotron_gap_meV=hbar * e * B_odd / M_STAR / e * 1e3))
    sp = pd.DataFrame(out)
    sp.to_csv(tab_path("spin_splitting.csv"), index=False, float_format="%.4g")
    return pts, res, sp


def load_tau_q():
    r = pd.read_csv(tab_path("dingle_results.csv")).iloc[0]
    return r["tau_q_ps"] * 1e-12, r["s_tau_q_ps"] * 1e-12


if __name__ == "__main__":
    pts, res, sp = run()
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pts.to_string(index=False)); print(res.to_string(index=False)); print(sp.to_string(index=False))
