"""
summary.py -- plan sections 0.5 (geometry N_sq), 7 (B = 0 values without a B = 0 gate sweep),
8 (summary table at Vg = 0, T ~ 4.2 K) and the uncertainty budget.

Density at Vg = 0 (ledger ruling): three determinations are reported --
  k_late * n_H(0)    Hall slope at 0.5 T in the gate sweeps, corrected with k of the same epoch,
  n_SdH(0)           intercept of the SdH-minima fan in the gate sweeps (plan's n_SdH),
  n_xx               SdH of the ungated part of the 6-11 path (field sweep), = Vg = 0 state if ungated.
Derived quantities use n_xx (smallest error); the spread of the three is a systematic.

  mu = N_sq / (e n R_xx),  tau_tr = m* mu / e,  l = hbar k_F mu / e,  k_F = sqrt(2 pi n),  E_F = pi hbar^2 n / m*
N_sq = L/W of pads 6-11 = 2 (TA, params.N_SQ) is used for all geometry-dependent quantities.
Own-data estimates, kept as consistency checks of that value (geometry_estimates.csv):
  (1) tau_tr >= tau_q        ->  N_sq >= e^2 n R_xx tau_q / m*            (satisfied by N_sq = 2)
  (2) phonon calibration     ->  N_sq = mu_ph(77 K) e n [R(77 K) - R(4.2 K)]   (Matthiessen; gives ~15, i.e.
      the rise of R between 4 and 77 K is ~8x larger than phonon scattering of a 2DEG of constant n allows)
mobility_vs_T.csv: mu(T) = N_sq/(e n R(T)) for T <= 100 K at constant n, and the mobilities that correspond
to the fitted acoustic and LO-phonon terms of rt_model (to compare with phonon-limited literature values).

Outputs: tables/geometry_estimates.csv, b0_values.csv, mobility_two_point.csv, summary_vg0.csv,
         uncertainty_budget.csv, mobility_vs_T.csv, plan_claims_check.csv
"""
import numpy as np
import pandas as pd

from dingle import load_tau_q
from hall import load_k
from load_data import load
from params import I_EXC_NOMINAL, LOCKIN_PHASE_DEG, MU_PH, M_STAR, N_SQ, e, hbar, tab_path


def get(table, col, row_filter=None, value="value"):
    t = pd.read_csv(tab_path(table))
    if row_filter is not None:
        t = t[row_filter(t)]
    return t


def densities():
    k, sk = load_k(late=True)          # gate sweeps belong to the late epoch
    k_early, sk_early = load_k()
    hd = pd.read_csv(tab_path("hall_density.csv"))
    nh0 = hd[hd["method"] == "gate sweeps |B|=0.5 T, mean(down,up)"].iloc[0]
    nh5 = hd.iloc[0]
    cap = pd.read_csv(tab_path("gate_capacitance.csv"))
    fan = cap[cap["method"].str.startswith("SdH minima")].iloc[0]
    sdh = pd.read_csv(tab_path("sdh_results.csv")).set_index("pair")
    xx = sdh.loc["Rxx_S (pads 6-11)"]
    qh = sdh.loc["Rxy_A (pads 6-3)"]
    d = dict(
        k=k_early, sk=sk_early, k_late=k, sk_late=sk,
        nH0_read=nh0["n_H_1e15"] * 1e15, s_nH0_read=nh0["sn_stat_1e15"] * 1e15,
        nH0_true=k * nh0["n_H_1e15"] * 1e15,
        s_nH0_true=k * nh0["n_H_1e15"] * 1e15 * np.hypot(sk / k, nh0["sn_stat_1e15"] / nh0["n_H_1e15"]),
        nSdH0=fan["n0"], s_nSdH0=fan["s_n0"],
        nxx=xx["n_fan_1e15"] * 1e15, s_nxx=xx["sn_fan_1e15"] * 1e15,
        nH5_read=nh5["n_H_1e15"] * 1e15, s_nH5_read=np.hypot(nh5["sn_stat_1e15"], nh5["sn_sys_1e15"]) * 1e15,
        nQH5=qh["n_fan_1e15"] * 1e15, s_nQH5=qh["sn_fan_1e15"] * 1e15,
    )
    vals = np.array([d["nH0_true"], d["nSdH0"], d["nxx"]])
    d["n0_best"], d["s_n0_stat"], d["s_n0_sys"] = d["nxx"], d["s_nxx"], np.max(np.abs(vals - d["nxx"]))
    return d


def b0_values():
    rt = pd.read_csv(tab_path("rt_summary.csv")).iloc[0]
    rows = [dict(Vg=0.0, source="cool-down end, T < 4.3 K (13 points)", R_ohm=rt["R_end_mean"], s_R_ohm=rt["R_end_std"]),
            dict(Vg=0.0, source="cool-down, T < 13 K (before final drop; systematic)", R_ohm=rt["R_below13K_mean"],
                 s_R_ohm=rt["R_below13K_std"])]
    for run in ("RHLabC-IV-13_44_56", "RHLabC-IV-15_01_29"):
        d = load(run)
        s = d[np.abs(d["B"]) < 0.005]
        rows.append(dict(Vg=0.5, source=f"{run}: R1 at |B| < 5 mT ({len(s)} pts)", R_ohm=s["R1"].mean(),
                         s_R_ohm=max(s["R1"].std(), 5 / np.sqrt(12))))
    fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
    lo = fs[(fs["absB"] > 0.02) & (fs["absB"] < 0.1) & np.isfinite(fs["Rxx_S"])]
    p, cov = np.polyfit(lo["absB"] ** 2, lo["Rxx_S"], 1, cov=True)
    rows.append(dict(Vg=0.5, source="R_xx^S(B) extrapolated B^2 -> 0 (|B| 0.02-0.1 T)", R_ohm=p[1], s_R_ohm=np.sqrt(cov[1, 1])))
    b = load("RVg-IV-12_14_49")
    s = b[b["Vg"] >= 0.495]
    rows.append(dict(Vg=0.5, source="RVg 12_14_49, B = 0, Vg = 0.50 V", R_ohm=s["R1"].mean(), s_R_ohm=max(s["R1"].std(), 5 / np.sqrt(12))))
    t = pd.DataFrame(rows)
    t.to_csv(tab_path("b0_values.csv"), index=False, float_format="%.5g")
    return t


def geometry(d, tau_q, R0):
    rt = pd.read_csv(tab_path("rt_clean.csv"))
    rv = pd.read_csv(tab_path("rt_values.csv")).set_index("T")
    near = rt[np.abs(rt["T"] - 77.0) < 1.5]
    R77 = near["R"].mean()
    dR = R77 - R0
    mu, smu = MU_PH[77.0]
    n, sn = d["n0_best"], np.hypot(d["s_n0_stat"], d["s_n0_sys"])
    Nsq = mu * e * n * dR
    sN = Nsq * np.sqrt((smu / mu) ** 2 + (sn / n) ** 2 + (5.0 / dR) ** 2)
    Nmin = e**2 * n * R0 * tau_q / M_STAR
    mu77_data = N_SQ / (e * n * R77)                       # what L/W = 2 implies for the 77 K mobility
    rows = [dict(method="TA value (used)", N_sq=N_SQ, s_N_sq=np.nan, note="L/W of pads 6-11, no uncertainty given"),
            dict(method="check: lower bound tau_tr >= tau_q", N_sq=Nmin, s_N_sq=np.nan,
                 note=f"N_sq >= value -> consistent with {N_SQ:g}"),
            dict(method="check: phonon calibration at 77 K (Matthiessen)", N_sq=Nsq, s_N_sq=sN,
                 note=f"{Nsq/N_SQ:.1f} x the TA value -> NOT consistent. mu_ph={mu} +- {smu} m^2/Vs (literature); "
                      f"L/W = 2 implies mu(77 K) = {mu77_data*1e4:.3g} cm^2/Vs at constant n, vs ~2e5 phonon-limited. "
                      f"R(77K)={R77:.0f} Ohm, R0={R0:.0f} Ohm, parallel-channel share at 77 K = "
                      f"{rv.loc[77.0, 'par_fraction_of_current']:.1e}")]
    g = pd.DataFrame(rows)
    g.to_csv(tab_path("geometry_estimates.csv"), index=False, float_format="%.4g")
    return g, Nsq, sN


def mobility_vs_T(n):
    """mu(T) for N_sq = 2 at constant n (assumption: n of the 2DEG does not change below 100 K), and the
    mobilities equivalent to the fitted acoustic (Bloch-Grueneisen) and LO-phonon terms of rt_model."""
    rv = pd.read_csv(tab_path("rt_values.csv"))
    rv = rv[rv["T"] <= 100.0]
    to_mu = lambda R: N_SQ / (e * n * R) * 1e4            # cm^2/Vs
    t = pd.DataFrame(dict(T_K=rv["T"], R_ohm=rv["R_data"], R_sq_ohm=rv["R_data"] / N_SQ,
                          mu_cm2Vs=to_mu(rv["R_data"]),
                          mu_ac_fit_cm2Vs=to_mu(rv["R_ac"]), mu_LO_fit_cm2Vs=to_mu(rv["R_LO"])))
    t.to_csv(tab_path("mobility_vs_T.csv"), index=False, float_format="%.4g")
    return t


def per_nsq(n, R):
    """Geometry-dependent quantities for N_sq = 1."""
    mu = 1 / (e * n * R)
    kF = np.sqrt(2 * np.pi * n)
    return dict(mu=mu, tau_tr=M_STAR * mu / e, l=hbar * kF * mu / e)


def run():
    d = densities()
    tau_q, s_tau_q = load_tau_q()
    b0 = b0_values()
    R0 = b0.iloc[0]["R_ohm"]
    sR0 = np.hypot(b0.iloc[0]["s_R_ohm"], b0.iloc[1]["R_ohm"] - R0)    # end-of-cool-down drop as systematic
    g, Nsq_ph, sN_ph = geometry(d, tau_q, R0)
    Nsq, sN = N_SQ, 0.0                                    # TA value; no uncertainty given
    mobility_vs_T(d["n0_best"])

    # ---- two-point mobility (section 7)
    R05 = b0[b0["Vg"] == 0.5]["R_ohm"].mean()
    sR05 = b0[b0["Vg"] == 0.5]["R_ohm"].std()
    mrows = []
    for lab, n, sn, R, sR, note in (
            ("Vg = 0 V", d["n0_best"], np.hypot(d["s_n0_stat"], d["s_n0_sys"]), R0, sR0, "homogeneous (gated = ungated density)"),
            ("Vg = +0.5 V (effective)", d["nQH5"], d["s_nQH5"], R05, sR05,
             "path 6-11 is inhomogeneous at +0.5 V (gated n_QH + ungated n_xx in series): effective value only")):
        q = per_nsq(n, R)
        mrows.append(dict(state=lab, n_1e15=n / 1e15, s_n_1e15=sn / 1e15, R_xx_ohm=R, s_R_ohm=sR,
                          mu_per_Nsq_m2Vs=q["mu"], mu_per_Nsq_cm2Vs=q["mu"] * 1e4, mu_at_Nsq2_cm2Vs=q["mu"] * 1e4 * Nsq,
                          s_mu_at_Nsq2_cm2Vs=q["mu"] * 1e4 * Nsq * np.sqrt((sN / Nsq) ** 2 + (sn / n) ** 2 + (sR / R) ** 2),
                          note=note))
    mob = pd.DataFrame(mrows)
    mob.to_csv(tab_path("mobility_two_point.csv"), index=False, float_format="%.5g")

    # ---- summary table at Vg = 0
    n, sn = d["n0_best"], np.hypot(d["s_n0_stat"], d["s_n0_sys"])
    q = per_nsq(n, R0)
    kF = np.sqrt(2 * np.pi * n)
    EF = np.pi * hbar**2 * n / M_STAR
    cap = pd.read_csv(tab_path("gate_capacitance.csv"))
    ch, cs = cap.iloc[0], cap[cap["method"].str.startswith("SdH minima")].iloc[0]
    al = pd.read_csv(tab_path("alpha_summary.csv"))
    a_field = al[(al["source"] == "field sweep_down") & (al["range"] == "|B| 0.6-5.495 T")].iloc[0]
    a_gate = al[al["source"].str.startswith("gate")]["alpha_mean"]
    rel = lambda *xs: np.sqrt(np.sum(np.square(xs)))
    rows = [
        ("n_H (Hall reading, 0.5 T, Vg=0)", d["nH0_read"] / 1e15, d["s_nH0_read"] / 1e15, "1e15 m^-2", "as read; not calibrated"),
        ("k = n_QH/n_H (Hall-channel calibration)", d["k"], d["sk"], "", "Vg=+0.5 V field sweep; plateau readings give 0.8033(5)"),
        ("k_late (same ratio, late up-ramps; applied to gate sweeps)", d["k_late"], d["sk_late"], "", "Hall scale drifted ~1 % between epochs"),
        ("n_H x k_late (Vg=0)", d["nH0_true"] / 1e15, d["s_nH0_true"] / 1e15, "1e15 m^-2", ""),
        ("n_SdH (gate-sweep fan intercept, Vg=0)", d["nSdH0"] / 1e15, d["s_nSdH0"] / 1e15, "1e15 m^-2", "plan's n_SdH"),
        ("n_xx (SdH of ungated part of 6-11 path)", d["nxx"] / 1e15, d["s_nxx"] / 1e15, "1e15 m^-2", "used for derived quantities"),
        ("n used (best), syst = spread of the three", n / 1e15, sn / 1e15, "1e15 m^-2", ""),
        ("R_xx(B=0, Vg=0)", R0, sR0, "Ohm", "cool-down end; as read"),
        ("mu x (1/N_sq)", q["mu"], q["mu"] * rel(sn / n, sR0 / R0), "m^2/Vs", "mu = N_sq x value"),
        ("mu at L/W = 2 (TA)", q["mu"] * Nsq * 1e4, q["mu"] * Nsq * 1e4 * rel(sn / n, sR0 / R0), "cm^2/Vs",
         f"x k = {d['k']:.3f} if the R_xx channel shares the Hall scale factor"),
        ("rho_xx = R_xx w/L (Vg=0)", R0 / Nsq, sR0 / Nsq, "Ohm/sq", ""),
        ("mu_q = e tau_q / m* (quantum mobility)", e * tau_q / M_STAR * 1e4, e * s_tau_q / M_STAR * 1e4, "cm^2/Vs", ""),
        ("tau_tr x (1/N_sq)", q["tau_tr"] * 1e12, q["tau_tr"] * 1e12 * rel(sn / n, sR0 / R0), "ps", ""),
        ("tau_tr at L/W = 2", q["tau_tr"] * Nsq * 1e12, q["tau_tr"] * Nsq * 1e12 * rel(sn / n, sR0 / R0), "ps", ""),
        ("tau_q (Dingle, T = 4.2 K)", tau_q * 1e12, s_tau_q * 1e12, "ps", "ungated region, same n as Vg=0"),
        ("tau_tr/tau_q x (1/N_sq)", q["tau_tr"] / tau_q, q["tau_tr"] / tau_q * rel(s_tau_q / tau_q, sn / n), "", ""),
        ("tau_tr/tau_q at L/W = 2", q["tau_tr"] * Nsq / tau_q, q["tau_tr"] * Nsq / tau_q * rel(sn / n, sR0 / R0, s_tau_q / tau_q), "",
         "R_bg(B) Dingle normalisation would halve this"),
        ("E_F", EF / e * 1e3, EF / e * 1e3 * sn / n, "meV", ""),
        ("k_F", kF / 1e8, kF / 1e8 * sn / (2 * n), "1e8 m^-1", ""),
        ("l_mfp x (1/N_sq)", q["l"] * 1e6, q["l"] * 1e6 * rel(sn / (2 * n), sR0 / R0), "um", ""),
        ("l_mfp at L/W = 2", q["l"] * Nsq * 1e6, q["l"] * Nsq * 1e6 * rel(sn / (2 * n), sR0 / R0), "um", ""),
        ("k_F l_mfp at L/W = 2 (>> 1: weak disorder)", kF * q["l"] * Nsq, kF * q["l"] * Nsq * rel(sn / n, sR0 / R0), "", ""),
        ("B where mu B = 1 (classical MR / Hall-angle scale)", 1 / (q["mu"] * Nsq), 1 / (q["mu"] * Nsq) * rel(sn / n, sR0 / R0), "T", ""),
        ("B where omega_c tau_q = 1 (SdH onset scale)", M_STAR / (e * tau_q), M_STAR / (e * tau_q) * s_tau_q / tau_q, "T", ""),
        ("C (Hall n(Vg), 0.5 T)", ch["C_uF_per_cm2"], ch["sC"] * 1e2, "uF/cm^2", "linear fit -0.5..0.5 V"),
        ("C (SdH minima fan)", cs["C_uF_per_cm2"], cs["sC"] * 1e2, "uF/cm^2", ""),
        ("d_eff = eps0 eps_r / C (Hall)", ch["d_eff_nm"], ch["s_d_eff_nm"], "nm", "eps_r = 12.9"),
        ("d_geom = d_eff - <z> - eps/C_q (Hall)", ch["d_geom_nm"], ch["s_d_eff_nm"], "nm", f"<z>={ch['z_avg_nm']:.1f} nm, eps/C_q={ch['eps_over_Cq_nm']:.1f} nm"),
        ("V_th (Hall linear fit)", ch["V_th"], np.nan, "V", ""),
        ("alpha (odd admixture in 6-11)", a_field["alpha_mean"], a_field["alpha_std"], "", f"gate sweeps {a_gate.min():.3f}-{a_gate.max():.3f}"),
        ("N_sq = L/W (TA)", Nsq, np.nan, "", "used"),
        ("N_sq lower bound (tau_tr >= tau_q)", g.iloc[1]["N_sq"], np.nan, "", "consistent"),
        ("N_sq phonon calibration (77 K), check only", Nsq_ph, sN_ph, "", "inconsistent with TA value (see geometry_estimates.csv)"),
    ]
    summ = pd.DataFrame(rows, columns=["quantity", "value", "uncertainty", "unit", "note"])
    summ.to_csv(tab_path("summary_vg0.csv"), index=False, float_format="%.5g")

    # ---- uncertainty budget
    hd = pd.read_csv(tab_path("hall_density.csv"))
    hy = pd.read_csv(tab_path("gate_hysteresis.csv"))
    dres = pd.read_csv(tab_path("dingle_results.csv"))
    sdhr = pd.read_csv(tab_path("sdh_results.csv"))
    seg = cap[cap["method"].str.contains("-0.5..-0.2 V|0.3..0.5 V", regex=True)]
    nH5 = hd.iloc[0]["n_H_1e15"]
    budget = [
        ("readout resolution 5 Ohm", "stat", "every R reading", f"sigma = {5/np.sqrt(12):.1f} Ohm per point", "digitisation step / sqrt(12)"),
        ("Hall slope fit", "stat", "n_H(+0.5 V)", f"{100*hd.iloc[0]['sn_stat_1e15']/nH5:.2f} %", "least squares, 0.15-1.0 T"),
        ("Hall fit range", "sys", "n_H(+0.5 V)", f"{100*max(abs(hd.iloc[i]['n_H_1e15']-nH5) for i in (2,3,4))/nH5:.2f} %", "0.15-0.5 / 0.5-1.0 / 0.1-1.2 T"),
        ("density drift early vs late (gate cycling)", "sys", "n_H, n_QH at +0.5 V", f"n_H {100*(hd.iloc[1]['n_H_1e15']-nH5)/nH5:+.1f} %", "S_up - S_down"),
        ("Hall-channel scale k", "sys", "all Hall densities, plateau values", f"k = {d['k']:.4f} +- {d['sk']:.4f} ({100*d['sk']/d['k']:.1f} %)", "n_QH/n_H; plateau readings 0.8033(5)"),
        ("lock-in phase 24 deg", "sys", "k (hypothesis)", f"cos24 = {np.cos(np.radians(LOCKIN_PHASE_DEG)):.3f}: excluded as sole cause ({(d['k']-np.cos(np.radians(24)))/d['sk']:.0f} sigma)", "hall_calibration.csv"),
        ("excitation current", "-", "all R", f"I = {I_EXC_NOMINAL*1e6:.1f} uA confirmed (1 V, 1 MOhm; program logs R = V/I)", "TA, 25 Nov -> k is not a current error"),
        ("R_xx channel scale (does it share k?)", "sys", "R_xx, rho_xx, mu, tau_tr, l_mfp", f"if yes: R_xx x {1/d['k']:.2f}, mu x {d['k']:.2f} ({100*(d['k']-1):.0f} %)", "unknown; no internal calibration of channel No.1"),
        ("lock-in lag", "sys", "Vg of gate features; B of field features",
         f"tau_eff = {hy.iloc[0]['tau_eff_s']:.1f} +- {hy.iloc[0]['s_tau_eff_s']:.1f} s; {hy.iloc[0]['dVg_mV']/2:.0f} mV per sweep; dB = tau*dB/dt ~ 6 mT", "SdH minima up vs down; cancelled by averaging / same-direction symmetrisation"),
        ("admixture alpha", "sys", "R_xx, R_xy before symmetrisation", f"alpha = {a_field['alpha_mean']:.3f} +- {a_field['alpha_std']:.3f} (field), {a_gate.min():.3f}-{a_gate.max():.3f} (gate)", "removed exactly by Onsager symmetrisation"),
        ("SdH background / range", "sys", "f, n_xx", f"{sdhr.iloc[0]['sf_sys_T']:.3f} T of {sdhr.iloc[0]['f_T']:.2f} T", "deg 2-5, B_min 0.8-1.2 T, S_up"),
        ("n(Vg) non-linearity", "sys", "C, d_eff", f"local dn/dVg {seg['dn_dVg_1e15_per_V'].min():.2f}-{seg['dn_dVg_1e15_per_V'].max():.2f} vs global {ch['dn_dVg_1e15_per_V']:.2f} 1e15/V", "segment fits"),
        ("eps_r (GaAs 12.9 vs AlGaAs 12.2)", "sys", "d_eff", f"{100*(1-12.2/12.9):.1f} %", ""),
        ("density at Vg=0: method spread", "sys", "n0 and all derived", f"{100*d['s_n0_sys']/d['n0_best']:.1f} %", "k n_H(0), n_SdH(0), n_xx"),
        ("temperature for Dingle (4.2 K vs logged)", "sys", "tau_q", f"{100*(dres.iloc[1]['tau_q_ps']-dres.iloc[0]['tau_q_ps'])/dres.iloc[0]['tau_q_ps']:+.0f} %", "dingle_results.csv; T itself not fixable (dingle_T_profile.csv)"),
        ("Dingle normalisation R(0) (used) vs R_bg(B)", "sys", "tau_q, tau_tr/tau_q, Gamma", f"{100*(dres.iloc[2]['tau_q_ps']-dres.iloc[0]['tau_q_ps'])/dres.iloc[0]['tau_q_ps']:+.0f} % (tau_q {dres.iloc[2]['tau_q_ps']:.2f} vs {dres.iloc[0]['tau_q_ps']:.2f} ps)", "R(0) chosen (Nadav, 25 Nov); R_bg(B) quoted as systematic"),
        ("Dingle small-amplitude cut", "sys", "tau_q", f"{100*(dres.iloc[3]['tau_q_ps']-dres.iloc[0]['tau_q_ps'])/dres.iloc[0]['tau_q_ps']:+.0f} %", "all points vs dR/R0 <= 0.5"),
        ("L/W = 2 (TA)", "sys", "mu, tau_tr, l_mfp, tau_tr/tau_q (all proportional to L/W)", "no uncertainty given by TA", "own phonon estimate 15 +- 5 disagrees (geometry_estimates.csv)"),
        ("R_xx(0) end-of-cool-down drop", "sys", "R0, mu", f"{sR0:.0f} Ohm", "839 (T<4.3 K) vs 857 (T<13 K)"),
    ]
    bud = pd.DataFrame(budget, columns=["source", "type", "affects", "magnitude", "how estimated"])
    bud.to_csv(tab_path("uncertainty_budget.csv"), index=False)
    plan_claims(d, b0, R0, Nsq_ph, sN_ph)
    return d, b0, g, mob, summ, bud


def plan_claims(d, b0, R0, Nsq, sN):
    """Every number quoted in the analysis plan, re-measured."""
    fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
    tm = pd.read_csv(tab_path("thermometer_master_curve.csv"))
    hy = pd.read_csv(tab_path("gate_hysteresis.csv")).iloc[0]
    pl = pd.read_csv(tab_path("hall_plateaus.csv"))
    rt = pd.read_csv(tab_path("rt_summary.csv")).iloc[0]
    al = pd.read_csv(tab_path("alpha_summary.csv"))
    z = pd.concat([load(r) for r in ("RHLabC-IV-13_44_56", "RHLabC-IV-15_01_29")])
    z5 = z[np.abs(z["B"]) < 0.005]
    r039 = np.interp(0.39, fs["absB"], fs["Rxx_S_down"])
    T = lambda b: np.interp(b, tm["absB"], tm["T_mean"])
    a_g = al[al["source"].str.startswith("gate")]["alpha_mean"]
    p6 = pl[pl["nu"] == 6].iloc[0]
    rows = [
        ("0.1", "No.2 reads 0 +- 5 Ohm at B = 0", f"R2 at |B|<5 mT: {z5['R2'].min():.0f}..{z5['R2'].max():.0f} Ohm", "confirmed"),
        ("0.1", "alpha = 0.57-0.60, constant for Vg -0.5..+0.5", f"field {al.iloc[3]['alpha_mean']:.3f}; gate sweeps {a_g.min():.3f}-{a_g.max():.3f}", "confirmed"),
        ("0.2", "12_49_45: B 0 -> 5.5 T", "starts at 0.477 T; B<0.48 T at +0.5 V only in 13_44 and 15_01", "corrected"),
        ("0.2", "13_44_56 is one of the stepped-sweep field ramps", "full sweep +5.34 -> -5.50 T at Vg=+0.5 V (primary field data)", "corrected"),
        ("0.2", "12_46_02 is an earlier save of 12_49_45", "exact prefix (774 identical rows)", "confirmed"),
        ("0.2", "(not in plan) 14_37_02", "first 101 rows are a copy of RVg 14_30_57", "new"),
        ("0.4", "symmetrised MR ~ +7 % at 0.39 T", f"R_xx^S(0.39 T)/R(0) - 1 = {100*(r039/b0.iloc[4]['R_ohm']-1):+.1f} %", "confirmed"),
        ("0.4", "R_xy = 0 within 5 Ohm at |B| < 5 mT", f"{z5['R2'].abs().max():.0f} Ohm", "confirmed"),
        ("0.6", "T_read 4.16 K at B=0, ~3.73 K at 0.4-0.5 T, 3.12 K at 5.5 T", f"{T(0.0):.3f} / {T(0.45):.3f} / {T(5.48):.3f} K", "confirmed"),
        ("0.6", "drop ~0.4 K within ~0.25 T, then slow decline", f"T(0)-T(0.25 T) = {T(0)-T(0.25):.3f} K; local max {tm[tm['absB'].between(0.6,2)]['T_mean'].max():.3f} K near 1.05 T", "confirmed (+ local max)"),
        ("1", "flat ~855-860 Ohm below ~13 K", f"{rt['R_below13K_mean']:.0f} +- {rt['R_below13K_std']:.0f} Ohm; final {rt['R_end_mean']:.0f} Ohm at 4.16 K", "confirmed; 18 Ohm drop at the end"),
        ("1", "parallel channel maximum near 150 K", f"R_max = {rt['Rmax']:.0f} Ohm at {rt['Tmax_R']:.0f} K", "confirmed"),
        ("2a", "raw flat R_xy ~ 3455 Ohm at 4.55-4.75 T (nu ~ 7.5)", f"plateau {p6['Rxy_read']:.0f} Ohm at {p6['B_plateau']:.2f} T = nu 6 of n_QH, k-corrected {p6['Rxy_true']:.0f} vs R_K/6 = {p6['R_K_over_nu']:.0f}", "resolved"),
        ("2b", "k ~ cos 24 deg", f"k = {d['k']:.3f} +- {d['sk']:.3f}; cos 24 = 0.914", "NOT confirmed"),
        ("2b", "single n_SdH", f"two: n_xx = {d['nxx']/1e15:.3f} (pads 6-11), n_QH = {d['nQH5']/1e15:.3f} (pads 6-3) 1e15 m^-2 at +0.5 V", "corrected"),
        ("3", "hysteresis ~ 25 mV from lock-in lag", f"{hy['dVg_mV']:.0f} +- {hy['s_dVg_mV']:.0f} mV (tau_eff {hy['tau_eff_s']:.1f} s) + anomalous loop at Vg 0..0.15 V", "confirmed + anomaly"),
        ("0.5", "N_sq ~ 15-20 (back-of-envelope)", f"own phonon estimate {Nsq:.1f} +- {sN:.1f}; TA: L/W = {N_SQ:g}", "contradicted by TA value"),
    ]
    pd.DataFrame(rows, columns=["plan_section", "plan_statement", "measured", "verdict"]).to_csv(
        tab_path("plan_claims_check.csv"), index=False)


if __name__ == "__main__":
    d, b0, g, mob, summ, bud = run()
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        print(b0.to_string(index=False)); print(g.to_string(index=False)); print(mob.to_string(index=False))
        print(summ.to_string(index=False)); print(bud.to_string(index=False))
