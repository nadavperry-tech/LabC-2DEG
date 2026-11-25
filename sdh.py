"""
sdh.py -- plan section 4 (SdH vs 1/B), Vg = +0.5 V field sweep.

Two voltage pairs are analysed separately (ledger ruling on the two densities):
  * R_xx^S (pads 6-11): background-subtracted oscillation, FFT in 1/B, Landau fan of the minima;
  * R_xy^A (pads 6-3):  residual after a linear fit, FFT in 1/B, plateau centres (minima of dR_xy/dB).

For spin-degenerate levels (spin splitting not resolved, section 2c) the density is
  n = 2 e f / h,       and the fan  nu = (h n / e) (1/B) + gamma  with nu = 2N at R_xx minima.

Outputs
  tables/sdh_fan_rxx.csv        minima of R_xx^S: B, 1/B, assigned nu
  tables/sdh_plateaus_rxy.csv   plateau centres of R_xy^A: B, R_xy, assigned nu
  tables/sdh_results.csv        f, n from FFT and fan for both pairs, with uncertainties and systematics
  tables/sdh_fft.csv            FFT spectra (for the figure)
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from params import SDH_BG_DEG, SDH_BMIN, e, h, tab_path
from symmetrize import load_field

B_MAX = 5.33          # upper edge = end of S_down (13_44_56)
N_FFT = 1 << 16


def detrend(B, R, deg):
    p = np.polyfit(B, R, deg)
    return R - np.polyval(p, B), np.polyval(p, B)


def fft_inverse_B(B, y, n_uniform=4096):
    """Hann-windowed, zero-padded FFT of y(1/B). Returns frequency [T], amplitude, uniform-grid spacing."""
    x = 1.0 / B
    order = np.argsort(x)
    xu = np.linspace(x.min(), x.max(), n_uniform)
    yu = np.interp(xu, x[order], y[order])
    yu = (yu - yu.mean()) * np.hanning(n_uniform)
    amp = np.abs(np.fft.rfft(yu, n=N_FFT))
    freq = np.fft.rfftfreq(N_FFT, xu[1] - xu[0])
    return freq, amp, x.max() - x.min()


def fft_peak(freq, amp, fmin=4.0, fmax=40.0):
    """Main peak with 3-point parabolic refinement; sigma_f = FWHM/2.355 / sqrt(number of periods) is
    NOT used -- we quote half the FFT resolution 1/(2*Delta(1/B)) as a conservative 1-sigma."""
    m = (freq > fmin) & (freq < fmax)
    i = np.flatnonzero(m)[np.argmax(amp[m])]
    y0, y1, y2 = amp[i - 1], amp[i], amp[i + 1]
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    df = freq[1] - freq[0]
    return freq[i] + d * df


def refine_extremum(B, y, i, half=3):
    s = slice(max(i - half, 0), min(i + half + 1, len(B)))
    p = np.polyfit(B[s], y[s], 2)
    return -p[1] / (2 * p[0]) if p[0] != 0 else B[i]


def fan_fit(Bmin, nu):
    """Linear fit nu = a (1/B) + gamma; returns n = a e/h, sigma_n, gamma, sigma_gamma."""
    x = 1.0 / np.asarray(Bmin)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, nu, rcond=None)
    dof = max(len(x) - 2, 1)
    s2 = np.sum((nu - A @ coef) ** 2) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    a, g = coef
    return a * e / h, np.sqrt(cov[0, 0]) * e / h, g, np.sqrt(cov[1, 1])


def assign_even_nu(Bmin, n_guess):
    """Assign the even nu closest to h n/(e B); check that consecutive minima differ by 2."""
    nu = 2 * np.round(h * n_guess / (e * np.asarray(Bmin)) / 2)
    return nu


def rxx_analysis(fs, column="Rxx_S", deg=SDH_BG_DEG, bmin=SDH_BMIN, bmax=B_MAX):
    s = fs[(fs["absB"] >= bmin) & (fs["absB"] <= bmax) & np.isfinite(fs[column])]
    B, R = s["absB"].to_numpy(), s[column].to_numpy()
    dR, bg = detrend(B, R, deg)
    freq, amp, span = fft_inverse_B(B, dR)
    f = fft_peak(freq, amp)
    sm = savgol_filter(dR, 15, 3)
    imin = find_peaks(-sm, prominence=15)[0]
    Bmin = np.array([refine_extremum(B, sm, i) for i in imin])
    Bmin = Bmin[Bmin > 1.2]
    nu = assign_even_nu(Bmin, 2 * e * f / h)
    n_fan, sn_fan, gam, sgam = fan_fit(Bmin, nu)
    return dict(B=B, R=R, dR=dR, bg=bg, freq=freq, amp=amp, f=f, sf=1 / (2 * span),
                Bmin=Bmin, nu=nu, n_fan=n_fan, sn_fan=sn_fan, gamma=gam, sgamma=sgam)


def rxy_analysis(fs, column="Rxy_A", bmin=SDH_BMIN, bmax=5.48):
    s = fs[(fs["absB"] >= bmin) & (fs["absB"] <= bmax) & np.isfinite(fs[column])]
    B, R = s["absB"].to_numpy(), s[column].to_numpy()
    dR, bg = detrend(B, R, 1)
    freq, amp, span = fft_inverse_B(B, dR)
    f = fft_peak(freq, amp)
    n_qh = 2 * e * f / h
    # plateau centres: local minima of the smoothed slope dR_xy/dB that fall below 60 % of the mean slope
    slope = savgol_filter(R, 31, 2, deriv=1, delta=B[1] - B[0])
    mean_slope = np.polyfit(B, R, 1)[0]
    ip = find_peaks(-slope, prominence=0.15 * mean_slope)[0]
    ip = ip[slope[ip] < 0.6 * mean_slope]
    Bp = np.array([refine_extremum(B, slope, i, half=8) for i in ip])
    Rp = np.interp(Bp, B, R)
    nu = assign_even_nu(Bp, n_qh)
    n_fan, sn_fan, gam, sgam = fan_fit(Bp, nu) if len(Bp) >= 3 else (np.nan,) * 4
    return dict(B=B, R=R, dR=dR, freq=freq, amp=amp, f=f, sf=1 / (2 * span), Bp=Bp, Rp=Rp, nu=nu,
                n_fan=n_fan, sn_fan=sn_fan, gamma=gam, sgamma=sgam, slope=slope, mean_slope=mean_slope)


def run():
    fs = load_field()
    xx = rxx_analysis(fs)
    xy = rxy_analysis(fs)
    # systematics on n_xx: background degree, lower field edge, late repeat (S_up)
    alt = {f"deg{d}": rxx_analysis(fs, deg=d)["f"] for d in (2, 4, 5)}
    alt.update({f"bmin{b}": rxx_analysis(fs, bmin=b)["f"] for b in (0.8, 1.2)})
    alt["S_up"] = rxx_analysis(fs, column="Rxx_S_up", bmax=5.45)["f"]
    alt_xy = {"S_up": rxy_analysis(fs, column="Rxy_A_up")["f"], "bmin1.5": rxy_analysis(fs, bmin=1.5)["f"]}
    sys_f = np.max(np.abs(np.array(list(alt.values())) - xx["f"]))
    sys_fxy = np.max(np.abs(np.array(list(alt_xy.values())) - xy["f"]))

    pd.DataFrame({"B_min": xx["Bmin"], "invB": 1 / xx["Bmin"], "nu": xx["nu"],
                  "nu_continuous_fft": 2 * xx["f"] / xx["Bmin"]}).to_csv(
        tab_path("sdh_fan_rxx.csv"), index=False, float_format="%.5g")
    pd.DataFrame({"B_plateau": xy["Bp"], "invB": 1 / xy["Bp"], "Rxy_read": xy["Rp"], "nu": xy["nu"]}).to_csv(
        tab_path("sdh_plateaus_rxy.csv"), index=False, float_format="%.5g")
    pd.DataFrame({"freq_T": xx["freq"], "amp_rxx": xx["amp"] / xx["amp"].max(),
                  "amp_rxy": np.interp(xx["freq"], xy["freq"], xy["amp"]) / xy["amp"].max()}).query(
        "freq_T < 40").to_csv(tab_path("sdh_fft.csv"), index=False, float_format="%.5g")

    xy_late = rxy_analysis(fs, column="Rxy_A_up")
    rows = []
    for lab, r, sysf in (("Rxx_S (pads 6-11)", xx, sys_f), ("Rxy_A (pads 6-3)", xy, sys_fxy),
                         ("Rxy_A late up-ramps (pads 6-3)", xy_late, 0.0)):
        n_fft = 2 * e * r["f"] / h
        sn_fft = 2 * e * np.hypot(r["sf"], sysf) / h
        rows.append(dict(pair=lab, f_T=r["f"], sf_stat_T=r["sf"], sf_sys_T=sysf,
                         n_fft_1e15=n_fft / 1e15, sn_fft_1e15=sn_fft / 1e15,
                         n_fan_1e15=r["n_fan"] / 1e15, sn_fan_1e15=r["sn_fan"] / 1e15,
                         fan_intercept=r["gamma"], s_fan_intercept=r["sgamma"],
                         nu_assigned=" ".join(str(int(v)) for v in r["nu"])))
    res = pd.DataFrame(rows)
    res.to_csv(tab_path("sdh_results.csv"), index=False, float_format="%.5g")
    return xx, xy, res, alt, alt_xy


def load_results():
    return pd.read_csv(tab_path("sdh_results.csv"))


if __name__ == "__main__":
    xx, xy, res, alt, alt_xy = run()
    with pd.option_context("display.width", 220):
        print(res.to_string(index=False))
    print("R_xx minima B:", np.round(xx["Bmin"], 3), "nu:", xx["nu"])
    print("R_xy plateau B:", np.round(xy["Bp"], 3), "R:", np.round(xy["Rp"]), "nu:", xy["nu"])
    print("alt f (Rxx):", {k: round(v, 3) for k, v in alt.items()}, " alt f (Rxy):", {k: round(v, 3) for k, v in alt_xy.items()})
