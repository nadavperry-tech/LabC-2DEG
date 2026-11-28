"""fig05_sdh_fft_fan.py -- Fig. 5: SdH oscillations vs 1/B, FFT and Landau fan for both voltage pairs (plan 4).
Inputs: tables/field_symmetrized_vg+0.5.csv, sdh_fft.csv, sdh_fan_rxx.csv, sdh_plateaus_rxy.csv, sdh_results.csv.
Output: figures/fig05_sdh_fft_fan.pdf"""
import numpy as np
import pandas as pd

from params import FULL_W, e, fig_path, h, legend_below, panel_label, plot_style, tab_path
from sdh import B_MAX, detrend

plt = plot_style()
fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
ff = pd.read_csv(tab_path("sdh_fft.csv"))
fx = pd.read_csv(tab_path("sdh_fan_rxx.csv"))
fy = pd.read_csv(tab_path("sdh_plateaus_rxy.csv"))
sd = pd.read_csv(tab_path("sdh_results.csv")).set_index("pair")
s = fs[(fs["absB"] >= 1.0) & (fs["absB"] <= B_MAX)]
B = s["absB"].to_numpy()
dxx, _ = detrend(B, s["Rxx_S"].to_numpy(), 3)
dxy, _ = detrend(B, s["Rxy_A"].to_numpy(), 1)
fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.95))          # figure*: full page width
ax[0].plot(1 / B, dxx, "-", color="k", label=r"$\Delta R_{xx}^S$ (cubic background removed)")
ax[0].plot(1 / B, dxy, "-", color="C3", label=r"$\Delta R_{xy}^A$ (linear background removed)")
ax[0].set_xlabel(r"$1/B$ [T$^{-1}$]"); ax[0].set_ylabel(r"$\Delta R$ [$\Omega$]")
ax[0].set_title(r"Oscillatory parts, $V_g$ = +0.5 V"); legend_below(ax[0])   # legends below the panels
ax[1].plot(ff["freq_T"], ff["amp_rxx"], "-", color="k", label=fr"$R_{{xx}}$: f = {sd.loc['Rxx_S (pads 6-11)','f_T']:.2f} T")
ax[1].plot(ff["freq_T"], ff["amp_rxy"], "-", color="C3", label=fr"$R_{{xy}}$: f = {sd.loc['Rxy_A (pads 6-3)','f_T']:.2f} T")
ax[1].set_xlim(0, 35); ax[1].set_xlabel("SdH frequency f [T]"); ax[1].set_ylabel("FFT amplitude [normalised]")
ax[1].set_title(r"FFT in $1/B$ (Hann window)"); legend_below(ax[1])
x = np.linspace(0, 0.9, 10)
fan_lines = []
for lab, d, col, mk, key in (("$R_{xx}$ minima", fx, "k", "o", "Rxx_S (pads 6-11)"),
                             ("$R_{xy}$ plateau centres", fy, "C3", "s", "Rxy_A (pads 6-3)")):
    n, g = sd.loc[key, "n_fan_1e15"] * 1e15, sd.loc[key, "fan_intercept"]
    ax[2].plot(d["invB"], d["nu"], mk, color=col, ms=4, label=lab)
    ax[2].plot(x, h * n / e * x + g, "-", color=col, lw=0.8)
    fan_lines.append((key, n, g, col))
ax[2].set_xlim(0, 0.9); ax[2].set_ylim(0, 20)
ax[2].set_xlabel(r"$1/B$ [T$^{-1}$]"); ax[2].set_ylabel(r"filling factor $\nu$"); ax[2].set_title("Landau fan")
for a_, l_ in zip(ax, "abc"): panel_label(a_, l_)
legend_below(ax[2])
# Fit results written along their lines (no room for them in a legend at print size): black below, red above.
fig.canvas.draw()                                   # fixes the layout, so the on-screen slope of each line is known
for key, n, g, col in fan_lines:
    x0, below = (0.31, True) if key.startswith("Rxx") else (0.035, False)
    p0, p1 = ax[2].transData.transform([(x0, h * n / e * x0 + g), (x0 + 0.1, h * n / e * (x0 + 0.1) + g)])
    ang = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    normal = np.array([-np.sin(ang), np.cos(ang)])          # unit normal pointing to the upper-left side of the line
    off = 4.0 * (-normal if below else normal)               # pt
    ax[2].annotate(fr"n = {n/1e15:.3f}e15 m$^{{-2}}$, $\gamma$ = {g:+.2f}", (x0, h * n / e * x0 + g),
                   xytext=tuple(off), textcoords="offset points", color=col, fontsize=7.5,
                   rotation=np.degrees(ang), rotation_mode="anchor", ha="left", va="top" if below else "bottom")
plt.savefig(fig_path("fig05_sdh_fft_fan.pdf"))
