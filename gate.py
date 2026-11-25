"""
gate.py -- plan sections 3 and 5 (+ appendix A9).

Section 3: n(Vg) from R_xy^A at |B| = 0.5 T (1.5 T check), true n = k_late * n_H (k_late from hall.py, same epoch).
   n = (C/e) (Vg - V_th),  d_eff = eps0 eps_r / C,
   d_eff = d_geom + <z> + eps/C_q   with  <z> = 3/b (Fang-Howard),  b^3 = 33 m* e^2 (11 n/32) / (8 eps hbar^2),
                                         C_q = e^2 m* / (pi hbar^2)  (spin-degenerate 2D DOS).
   Hysteresis/lag: Vg shift of the same SdH minimum between up and down sweeps (= 2 tau |dVg/dt|),
   and the down-up density difference of the Hall n(Vg) in three Vg segments.
Section 5: R_xx^S minima in the gate sweeps -> n_nu = nu e B / h at Vg_nu (Hall-region nu, ledger ruling)
   -> second capacitance from the fan and from the spacing DeltaVg = 2 e^2 B / (h C);
   alignment of R_xx minima with R_xy plateau centres (inhomogeneity test).
Appendix A9: exponential fit of R_xx(Vg) and extrapolation to Vg = 0 (why it is not used).

Outputs: tables/gate_density.csv, gate_capacitance.csv, gate_sdh_minima.csv, gate_alignment.csv,
         gate_hysteresis.csv, appendix_A9_fits.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

from hall import load_k
from load_data import gate_runs, load, sweep_direction
from params import (EPS_R, EPS_R_ALT, LOCKIN_DELAY_S, LOCKIN_SETTLE_99_S, M_STAR, e, eps0, h, hbar,
                    tab_path)
from symmetrize import load_gate


def linfit(x, y, sy=None):
    w = None if sy is None else 1 / np.asarray(sy) ** 2
    p, cov = np.polyfit(x, y, 1, w=w, cov="unscaled" if sy is not None else True)
    return p, np.sqrt(np.diag(cov))


def density_vs_gate(gs, k):
    rows = []
    for b in (0.5, 1.5):
        for d in ("down", "up"):
            q = gs[(gs["absB"] == b) & (gs["dir"] == d)].sort_values("Vg")
            nH = b / (e * q["Rxy_A"].to_numpy())
            rows.append(pd.DataFrame({"absB": b, "dir": d, "Vg": q["Vg"], "n_H_read": nH, "n_true": k * nH}))
    return pd.concat(rows, ignore_index=True)


def capacitance_from_hall(dens, k, sk, absB=0.5, lo=-0.5, hi=0.5):
    out = {}
    for d in ("down", "up"):
        q = dens[(dens["absB"] == absB) & (dens["dir"] == d) & dens["Vg"].between(lo, hi)]
        p, sp = linfit(q["Vg"], q["n_true"])
        out[d] = (p, sp)
    slope = np.mean([out[d][0][0] for d in out])
    icpt = np.mean([out[d][0][1] for d in out])
    s_slope = np.hypot(abs(out["down"][0][0] - out["up"][0][0]) / 2, slope * sk / k)
    C = e * slope
    vth = -icpt / slope
    return dict(method=f"Hall, |B|={absB} T, Vg {lo}..{hi} V", dn_dVg=slope, s_dn_dVg=s_slope,
                n0=icpt, s_n0=np.hypot(abs(out["down"][0][1] - out["up"][0][1]) / 2, icpt * sk / k),
                C=C, sC=e * s_slope, V_th=vth,
                slope_down=out["down"][0][0], slope_up=out["up"][0][0],
                n0_down=out["down"][0][1], n0_up=out["up"][0][1])


def thickness(C, sC, n):
    eps = eps0 * EPS_R
    d_eff = eps / C
    sd = d_eff * sC / C
    d_alt = eps0 * EPS_R_ALT / C
    b = (33 * M_STAR * e**2 * (11 * n / 32) / (8 * eps * hbar**2)) ** (1 / 3)
    z_avg = 3 / b
    eps_over_Cq = eps * np.pi * hbar**2 / (e**2 * M_STAR)
    return dict(d_eff_nm=d_eff * 1e9, s_d_eff_nm=sd * 1e9, d_eff_epsAlGaAs_nm=d_alt * 1e9,
                z_avg_nm=z_avg * 1e9, eps_over_Cq_nm=eps_over_Cq * 1e9,
                d_geom_nm=(d_eff - z_avg - eps_over_Cq) * 1e9)


def sweep_rates():
    rows = []
    for r in gate_runs():
        dv = np.diff(r["Vg"].to_numpy())
        dt = np.diff(r["t"].to_numpy())
        ok = (np.abs(dv) > 1e-4) & (dt > 0)
        rows.append(dict(run=r.attrs["name"], absB=round(abs(r["B"].median()), 1),
                         dir="down" if sweep_direction(r["Vg"]) < 0 else "up",
                         rate_V_per_s=np.median(np.abs(dv[ok] / dt[ok])), dt_point_s=np.median(dt)))
    return pd.DataFrame(rows)


def hysteresis(dens, rates, mins):
    """(1) lock-in lag from SdH minima: Vg_up - Vg_down for the same (|B|, nu) -> tau_eff = shift/(2 rate);
       (2) Hall n(Vg) at |B| = 0.5 T: mean n_down - n_up at fixed Vg in three Vg segments."""
    rate = rates["rate_V_per_s"].median()
    rows = []
    piv = mins.pivot_table(index=["absB", "nu"], columns="dir", values="Vg_min").dropna()
    sh = (piv["up"] - piv["down"]).to_numpy()
    rows.append(dict(source="SdH minima pairs (up - down)", n=len(sh), dVg_mV=1e3 * sh.mean(),
                     s_dVg_mV=1e3 * sh.std(ddof=1) / np.sqrt(len(sh)), spread_mV=1e3 * sh.std(ddof=1),
                     rate_mV_per_s=1e3 * rate, tau_eff_s=sh.mean() / (2 * rate),
                     s_tau_eff_s=sh.std(ddof=1) / np.sqrt(len(sh)) / (2 * rate),
                     expected_2s_mV=1e3 * 2 * LOCKIN_DELAY_S * rate, expected_6p6s_mV=1e3 * 2 * LOCKIN_SETTLE_99_S * rate))
    dn = dens[(dens["absB"] == 0.5) & (dens["dir"] == "down")].set_index("Vg")["n_true"]
    up = dens[(dens["absB"] == 0.5) & (dens["dir"] == "up")].set_index("Vg")["n_true"]
    j = dn.index.intersection(up.index)
    diff = (dn.loc[j] - up.loc[j])
    for lo, hi in ((-0.5, -0.2), (-0.1, 0.15), (0.3, 0.5)):
        d = diff[(diff.index >= lo) & (diff.index <= hi)]
        q = dens[(dens["absB"] == 0.5) & dens["Vg"].between(lo, hi)]
        slope = np.polyfit(q["Vg"], q["n_true"], 1)[0]
        rows.append(dict(source=f"Hall n_down - n_up at fixed Vg, 0.5 T, Vg {lo}..{hi} V", n=len(d),
                         dn_1e15=d.mean() / 1e15, s_dn_1e15=d.std(ddof=1) / np.sqrt(len(d)) / 1e15,
                         local_dn_dVg_1e15=slope / 1e15, dVg_mV=1e3 * d.mean() / slope,
                         rate_mV_per_s=1e3 * rate, tau_eff_s=d.mean() / slope / (2 * rate)))   # >0 = normal lag
    return pd.DataFrame(rows)


def sdh_minima(gs, n_of_vg):
    """R_xx^S minima vs Vg; nu assigned from the Hall-region density n_of_vg(Vg, direction)
    (measured k*n_H at 0.5 T in the same sweep direction, so lag and non-linearity cancel)."""
    rows = []
    for (b, d), q in gs.groupby(["absB", "dir"]):
        if b < 1.0:
            continue
        q = q.sort_values("Vg")
        V, R = q["Vg"].to_numpy(), q["Rxx_S"].to_numpy()
        sm = savgol_filter(R, 7, 2)
        idx = find_peaks(-sm, prominence=max(20.0, 0.01 * np.median(R)))[0]
        for i in idx:
            if i < 3 or i > len(V) - 4:
                continue
            s = slice(i - 3, i + 4)
            p = np.polyfit(V[s], sm[s], 2)
            v0 = -p[1] / (2 * p[0])
            nu_c = h * n_of_vg(v0, d) / (e * b)
            nu = 2 * np.round(nu_c / 2)
            rows.append(dict(absB=b, dir=d, Vg_min=v0, nu_continuous=nu_c, nu=nu, n_nu=nu * e * b / h,
                             Rxx_S_min=np.interp(v0, V, sm)))
    return pd.DataFrame(rows)


def rxy_plateaus_vs_gate(gs):
    rows = []
    for (b, d), q in gs.groupby(["absB", "dir"]):
        if b < 1.0:
            continue
        q = q.sort_values("Vg")
        V, R = q["Vg"].to_numpy(), q["Rxy_A"].to_numpy()
        dR = savgol_filter(R, 9, 2, deriv=1, delta=V[1] - V[0])   # dR_xy/dVg < 0 (n grows with Vg)
        mag = np.abs(dR)
        idx = find_peaks(-mag, prominence=0.15 * np.median(mag))[0]
        for i in idx:
            if i < 4 or i > len(V) - 5:
                continue
            rows.append(dict(absB=b, dir=d, Vg_plateau=V[i], Rxy_A_plateau=R[i], slope_rel=mag[i] / np.median(mag)))
    return pd.DataFrame(rows)


def alignment(mins, plats, k, tol=0.07):
    """Match each R_xx minimum (nu) with the R_xy plateau whose k-corrected value is within tol of R_K/nu."""
    from params import R_K
    rows = []
    for _, m in mins.iterrows():
        p = plats[(plats["absB"] == m["absB"]) & (plats["dir"] == m["dir"])].copy()
        p["Rtrue"] = p["Rxy_A_plateau"] / k
        p = p[np.abs(p["Rtrue"] / (R_K / m["nu"]) - 1) < tol]
        r = dict(absB=m["absB"], dir=m["dir"], nu=m["nu"], Vg_Rxx_min=m["Vg_min"], R_K_over_nu=R_K / m["nu"])
        if not p.empty:
            pp = p.iloc[np.argmin(np.abs(p["Vg_plateau"] - m["Vg_min"]))]
            r.update(Vg_Rxy_plateau=pp["Vg_plateau"], dVg_mV=1e3 * (pp["Vg_plateau"] - m["Vg_min"]),
                     Rxy_plateau_true=pp["Rtrue"], plateau_dev_pct=100 * (pp["Rtrue"] / (R_K / m["nu"]) - 1))
        rows.append(r)
    return pd.DataFrame(rows)


def capacitance_from_sdh(mins):
    out = {}
    for d in ("down", "up"):
        q = mins[mins["dir"] == d]
        p, sp = linfit(q["Vg_min"], q["n_nu"])
        out[d] = (p, sp, len(q))
    slope = np.mean([out[d][0][0] for d in out])
    icpt = np.mean([out[d][0][1] for d in out])
    s_slope = np.hypot(abs(out["down"][0][0] - out["up"][0][0]) / 2, np.mean([out[d][1][0] for d in out]))
    s_icpt = np.hypot(abs(out["down"][0][1] - out["up"][0][1]) / 2, np.mean([out[d][1][1] for d in out]))
    return dict(method="SdH minima in gate sweeps (fan n_nu vs Vg_nu), mean down/up", dn_dVg=slope,
                s_dn_dVg=s_slope, n0=icpt, s_n0=s_icpt, C=e * slope, sC=e * s_slope, V_th=-icpt / slope,
                slope_down=out["down"][0][0], slope_up=out["up"][0][0], n0_down=out["down"][0][1],
                n0_up=out["up"][0][1], n_points=out["down"][2] + out["up"][2])


def spacing_vs_B(mins):
    """DeltaVg between consecutive even-nu minima at each |B| -> C = 2 e^2 B / (h DeltaVg)."""
    rows = []
    for (b, d), q in mins.groupby(["absB", "dir"]):
        q = q.sort_values("Vg_min")
        for (v1, n1), (v2, n2) in zip(q[["Vg_min", "nu"]].values[:-1], q[["Vg_min", "nu"]].values[1:]):
            if n2 - n1 == 2:
                rows.append(dict(absB=b, dir=d, nu_pair=f"{int(n1)}-{int(n2)}", dVg=v2 - v1))
    sp = pd.DataFrame(rows)
    # fit DeltaVg = (2 e^2/(h C)) B  through the origin
    x, y = sp["absB"].to_numpy(), sp["dVg"].to_numpy()
    a = np.sum(x * y) / np.sum(x * x)
    sa = np.sqrt(np.sum((y - a * x) ** 2) / (len(x) - 1) / np.sum(x * x))
    C = 2 * e**2 / (h * a)
    return sp, dict(method="spacing DeltaVg of consecutive SdH minima vs B (through origin)",
                    dVg_per_T=a, s_dVg_per_T=sa, C=C, sC=C * sa / a, dn_dVg=C / e, s_dn_dVg=C * sa / a / e)


# ----------------------------------------------------------------- appendix A9
def expo(v, rinf, a, v0):
    return rinf + a * np.exp(-v / v0)


def power(v, a, p, vth):
    return a * np.clip(v - vth, 1e-6, None) ** (-p)


def appendix_A9(gs, vth_hall):
    rows = []
    b0 = load("RVg-IV-12_14_49")
    V0, R0 = b0["Vg"].to_numpy(), b0["R1"].to_numpy()
    q = gs[gs["absB"] == 0.5].groupby("Vg", as_index=False)["Rxx_S"].mean()
    V5, R5 = q["Vg"].to_numpy(), q["Rxx_S"].to_numpy()
    for lab, V, R in (("B=0, 9 points Vg 0.43-0.50 V (12_14_49)", V0, R0), ("|B|=0.5 T, Rxx_S, Vg -0.5..0.5 V", V5, R5)):
        sR = np.full_like(R, 5 / np.sqrt(12) + 0.0)
        try:
            p, cov = curve_fit(expo, V, R, p0=(R.min() * 0.8, R.max() * 0.3, 0.2), sigma=sR, maxfev=20000)
            perr = np.sqrt(np.diag(cov))
            # extrapolation to Vg=0 with parameter covariance (MC)
            mc = np.random.default_rng(1).multivariate_normal(p, cov, 4000)
            ext = np.array([expo(0.0, *m) for m in mc])
            chi2 = np.sum(((R - expo(V, *p)) / sR) ** 2) / max(len(R) - 3, 1)
            rows.append(dict(data=lab, model="R_inf + A exp(-Vg/V0)", p1=p[0], sp1=perr[0], p2=p[1], sp2=perr[1],
                             p3=p[2], sp3=perr[2], chi2_red=chi2, R_at_Vg0=expo(0.0, *p),
                             R_at_Vg0_16=np.percentile(ext, 16), R_at_Vg0_84=np.percentile(ext, 84)))
        except RuntimeError as err:
            rows.append(dict(data=lab, model="R_inf + A exp(-Vg/V0)", note=f"fit failed: {err}"))
        try:
            p, cov = curve_fit(lambda v, a, pw: power(v, a, pw, vth_hall), V, R, p0=(R.mean(), 1.0), sigma=sR, maxfev=20000)
            perr = np.sqrt(np.diag(cov))
            chi2 = np.sum(((R - power(V, *p, vth_hall)) / sR) ** 2) / max(len(R) - 2, 1)
            rows.append(dict(data=lab, model=f"Drude A (Vg-V_th)^-p, V_th={vth_hall:.3f} V fixed", p1=p[0], sp1=perr[0],
                             p2=p[1], sp2=perr[1], chi2_red=chi2, R_at_Vg0=power(0.0, *p, vth_hall)))
        except RuntimeError as err:
            rows.append(dict(data=lab, model="Drude power law", note=f"fit failed: {err}"))
    return pd.DataFrame(rows)


def run():
    k, sk = load_k(late=True)
    gs = load_gate()
    dens = density_vs_gate(gs, k)
    dens.to_csv(tab_path("gate_density.csv"), index=False, float_format="%.6g")
    rates = sweep_rates()

    c_h = capacitance_from_hall(dens, k, sk, 0.5)
    c_h15 = capacitance_from_hall(dens, k, sk, 1.5)
    c_hneg = capacitance_from_hall(dens, k, sk, 0.5, -0.5, -0.2)
    c_hpos = capacitance_from_hall(dens, k, sk, 0.5, 0.3, 0.5)
    # measured Hall-region density at 0.5 T, same sweep direction, used to assign nu
    def n_of_vg(v, d):
        q = dens[(dens["absB"] == 0.5) & (dens["dir"] == d)].sort_values("Vg")
        return np.interp(v, q["Vg"], q["n_true"])
    mins = sdh_minima(gs, n_of_vg)
    mins.to_csv(tab_path("gate_sdh_minima.csv"), index=False, float_format="%.5g")
    hyst = hysteresis(dens, rates, mins)
    hyst.to_csv(tab_path("gate_hysteresis.csv"), index=False, float_format="%.4g")
    plats = rxy_plateaus_vs_gate(gs)
    al = alignment(mins, plats, k)
    al.to_csv(tab_path("gate_alignment.csv"), index=False, float_format="%.4g")
    c_s = capacitance_from_sdh(mins)
    sp, c_sp = spacing_vs_B(mins)
    sp.to_csv(tab_path("gate_sdh_spacing.csv"), index=False, float_format="%.4g")

    rows = []
    for c in (c_h, c_h15, c_hneg, c_hpos, c_s, c_sp):
        r = dict(c)
        r.update(thickness(c["C"], c["sC"], c.get("n0", c_h["n0"])))
        rows.append(r)
    cap = pd.DataFrame(rows)
    cap["C_uF_per_cm2"] = cap["C"] * 1e2       # F/m^2 -> uF/cm^2
    cap["dn_dVg_1e15_per_V"] = cap["dn_dVg"] / 1e15
    cap["n0_1e15"] = cap["n0"] / 1e15
    cap.to_csv(tab_path("gate_capacitance.csv"), index=False, float_format="%.5g")
    a9 = appendix_A9(gs, c_h["V_th"])
    a9.to_csv(tab_path("appendix_A9_fits.csv"), index=False, float_format="%.5g")
    return dict(dens=dens, rates=rates, hyst=hyst, mins=mins, plats=plats, al=al, cap=cap, sp=sp, a9=a9)


if __name__ == "__main__":
    out = run()
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60):
        print(out["hyst"].to_string(index=False))
        print(out["cap"][["method", "dn_dVg_1e15_per_V", "s_dn_dVg", "n0_1e15", "V_th", "C_uF_per_cm2",
                          "d_eff_nm", "s_d_eff_nm", "z_avg_nm", "eps_over_Cq_nm", "d_geom_nm"]].to_string(index=False))
        print(out["mins"].to_string(index=False))
        print(out["al"].to_string(index=False))
        print(out["a9"].to_string(index=False))
        print(out["rates"].groupby("dir")[["rate_V_per_s", "dt_point_s"]].median())
