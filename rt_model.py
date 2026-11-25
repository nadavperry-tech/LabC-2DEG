"""
rt_model.py -- plan section 1: cool-down R(T) at B = 0, Vg = 0 (Rt-IV-11_59_06).

Two parallel channels (A1-A3):
  R(T)     = 1 / ( 1/R_2D(T) + G_par(T) )
  R_2D(T)  = R_0 + A T G_BG(T_BG/T) + B_LO / (exp(T_LO/T) - 1)          (Matthiessen)
  G_BG(z)  = (3/2) int_0^1 x^3/sqrt(1-x^2) (zx)^2 e^{zx}/(e^{zx}-1)^2 dx   (-> 1 for T >> T_BG)
  R_par(T) = R_s + R_a exp(E_a / k_B T)       (parallel channel in the doped AlGaAs: activated freeze-out
                                               at low T, saturating once the donors are ionised; the pure
                                               G_0 exp(-E_a/k_B T) form of the plan is kept as a check)
T_BG = 2 hbar k_F v_s / k_B with k_F = sqrt(2 pi n(Vg=0)) is fixed; T_LO = hbar omega_LO / k_B fixed
(36.2 meV) in the main fit and free in a check. Fit in ln R with 1 % weights.

Cleaning: rows with t < 126 s (R stuck at 150 Ohm, then the range change) are dropped; isolated
range-change spikes are removed with a Hampel filter.

Outputs: tables/rt_clean.csv, rt_fit_params.csv, rt_arrhenius.csv, rt_values.csv
"""
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares

from load_data import hampel, load
from params import HBAR_OMEGA_LO_meV, V_SOUND_LA, e, hbar, kB, tab_path

T_START_S = 126.0
KB_EV = kB / e


def _g_bg_exact(z):
    if z < 1e-3:
        return 1.0
    f = lambda x: x**3 / np.sqrt(1 - x**2) * (z * x) ** 2 * np.exp(z * x) / np.expm1(z * x) ** 2
    return 1.5 * quad(f, 0, 1, limit=200)[0]


_Z = np.logspace(-3, 2.5, 400)
_G = np.array([_g_bg_exact(z) for z in _Z])


def g_bg(z):
    return np.interp(np.log(z), np.log(_Z), _G, left=1.0, right=_G[-1])


def t_bg(n):
    return 2 * hbar * np.sqrt(2 * np.pi * n) * V_SOUND_LA / kB


def clean():
    df = load("Rt-IV-11_59_06")
    df = df[df["t"] >= T_START_S].copy()
    R, bad = hampel(df["R1"].to_numpy(), window=7, n_sigma=4.0)
    df["R_raw"] = df["R1"]
    df["R"] = R
    df["spike"] = bad
    return df


NAMES = {"sat": ["R0_ohm", "A_ohm_per_K", "B_LO_ohm", "Rs_par_ohm", "Ra_par_ohm", "Ea_eV"],
         "pure": ["R0_ohm", "A_ohm_per_K", "B_LO_ohm", "G0_S", "Ea_eV"]}


def model_parts(T, p, TBG, TLO, kind="sat"):
    T = np.atleast_1d(np.asarray(T, float))
    npar = len(NAMES[kind])
    R0, A, BLO = p[:3]
    TLO = p[npar] if len(p) > npar else TLO
    r_ac = A * T * g_bg(TBG / T)
    r_lo = BLO / np.expm1(TLO / T)
    r2d = R0 + r_ac + r_lo
    if kind == "sat":
        Rs, Ra, Ea = p[3:6]
        with np.errstate(over="ignore"):
            gpar = 1 / (Rs + Ra * np.exp(np.minimum(Ea / (KB_EV * T), 700)))
    else:
        G0, Ea = p[3:5]
        gpar = G0 * np.exp(-Ea / (KB_EV * T))
    return r2d, r_ac, r_lo, gpar, 1 / (1 / r2d + gpar)


def fit(T, R, TBG, TLO, free_TLO=False, Tmin=0.0, kind="sat"):
    sel = T >= Tmin
    T, R = T[sel], R[sel]
    if kind == "sat":
        p0, lo, hi = [850, 6.0, 9e4, 800, 1e-2, 0.15], [0, 0, 0, 0, 0, 0], [5000, 100, 1e7, 1e5, 1e6, 1.0]
    else:
        p0, lo, hi = [850, 6.0, 9e4, 0.2, 0.1], [0, 0, 0, 0, 0], [5000, 100, 1e7, 1e3, 1.0]
    if free_TLO:
        p0, lo, hi = p0 + [TLO], lo + [200], hi + [800]

    def res(p):
        return (np.log(model_parts(T, p, TBG, TLO, kind)[4]) - np.log(R)) / 0.01

    r = least_squares(res, p0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * max(1.0, np.sum(r.fun**2) / (len(T) - len(p0)))
    return r.x, np.sqrt(np.diag(cov)), np.sum(r.fun**2), len(T) - len(p0)


def arrhenius(T, R, p, TBG, TLO):
    """G_par extracted from the data: 1/R - 1/R_2D(model), and the local activation energy."""
    r2d = model_parts(T, p, TBG, TLO)[0]
    gpar = 1 / R - 1 / r2d
    ok = (T > 130) & (gpar > 0)
    x = 1 / (KB_EV * T[ok])
    y = np.log(gpar[ok])
    d = pd.DataFrame({"T": T[ok], "inv_kT_per_eV": x, "ln_Gpar": y})
    d = d.sort_values("inv_kT_per_eV")
    win = 25
    Ea_loc = -pd.Series(np.gradient(d["ln_Gpar"].rolling(win, center=True, min_periods=5).mean(), d["inv_kT_per_eV"]))
    d["Ea_local_eV"] = Ea_loc.rolling(win, center=True, min_periods=5).median().to_numpy()
    return d


def run():
    df = clean()
    T, R = df["T"].to_numpy(), df["R"].to_numpy()
    dens = pd.read_csv(tab_path("hall_calibration.csv"))
    n0 = float(dens[dens["quantity"].str.startswith("k_late * n_H(Vg=0)")]["value"].iloc[0]) * 1e15
    TBG = t_bg(n0)
    TLO = HBAR_OMEGA_LO_meV * 1e-3 * e / kB
    rows = []
    fits = {}
    for lab, kind, free in (("main: saturating parallel channel, T_LO fixed", "sat", False),
                            ("check: saturating parallel channel, T_LO free", "sat", True),
                            ("check: pure activated G0 exp(-Ea/kT) (plan form), T_LO fixed", "pure", False)):
        pp, ss, c2, dof = fit(T, R, TBG, TLO, free_TLO=free, kind=kind)
        fits[lab] = (pp, kind)
        names = NAMES[kind]
        r = dict(fit=lab, **{n: v for n, v in zip(names, pp)}, **{"s_" + n: v for n, v in zip(names, ss)},
                 T_LO_K=pp[len(names)] if free else TLO, T_BG_K=TBG, chi2_ln=c2, dof=dof,
                 rms_resid_pct=np.sqrt(c2 / (dof + len(pp))))
        if free:
            r["s_T_LO_K"] = ss[len(names)]
        rows.append(r)
    p = fits[rows[0]["fit"]][0]
    pars = pd.DataFrame(rows)
    pars["hbar_omega_LO_meV"] = pars["T_LO_K"] * kB / e * 1e3
    pars.to_csv(tab_path("rt_fit_params.csv"), index=False, float_format="%.5g")

    r2d, rac, rlo, gpar, rtot = model_parts(T, p, TBG, TLO)
    out = df[["t", "T", "R_raw", "R", "spike"]].copy()
    out["R_fit"], out["R_2D_fit"], out["R_ac_fit"], out["R_LO_fit"], out["G_par_fit"] = rtot, r2d, rac, rlo, gpar
    out.to_csv(tab_path("rt_clean.csv"), index=False, float_format="%.6g")
    arr = arrhenius(T, R, p, TBG, TLO)
    arr.to_csv(tab_path("rt_arrhenius.csv"), index=False, float_format="%.5g")

    low = df[df["T"] < 13]
    vals = []
    for t0 in (4.16, 10, 13, 20, 40, 77, 100, 150, 200, 280):
        i = np.argmin(np.abs(T - t0))
        r2, ra, rl, gp, rt = model_parts(np.array([t0]), p, TBG, TLO)
        vals.append(dict(T=t0, R_data=R[i], R_fit=rt[0], R_2D_fit=r2[0], R_par_fit=1 / gp[0],
                         R_ac=ra[0], R_LO=rl[0], par_fraction_of_current=gp[0] * rt[0]))
    vals = pd.DataFrame(vals)
    vals.to_csv(tab_path("rt_values.csv"), index=False, float_format="%.5g")
    end = df[df["T"] < 4.3]
    summary = dict(R_end_mean=end["R"].mean(), R_end_std=end["R"].std(), R_end_n=len(end),
                   R_below13K_mean=low["R"].mean(), R_below13K_std=low["R"].std(),
                   Tmax_R=T[np.argmax(R)], Rmax=R.max(), n_spikes=int(df["spike"].sum()))
    pd.DataFrame([summary]).to_csv(tab_path("rt_summary.csv"), index=False, float_format="%.5g")
    return pars, vals, arr, summary


def load_params():
    return pd.read_csv(tab_path("rt_fit_params.csv")).iloc[0]


if __name__ == "__main__":
    pars, vals, arr, summary = run()
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pars.to_string(index=False)); print(vals.to_string(index=False)); print(summary)
        print(arr.iloc[::40][["T", "Ea_local_eV"]].to_string(index=False))
