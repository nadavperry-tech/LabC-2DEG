"""fig04_hall_plateaus.py -- Fig. 4: R_xy^A and R_xx^S vs B at Vg = +0.5 V with filling factors (plan 2a/2b).
Inputs: tables/field_symmetrized_vg+0.5.csv, hall_plateaus.csv, hall_calibration.csv, sdh_results.csv.
Output: figures/fig04_hall_plateaus.pdf"""
import numpy as np
import pandas as pd

from hall import load_k
from params import R_K, e, fig_path, h, plot_style, tab_path

plt = plot_style()
fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
pl = pd.read_csv(tab_path("hall_plateaus.csv"))
sd = pd.read_csv(tab_path("sdh_results.csv")).set_index("pair")
k, sk = load_k()
n_qh = sd.loc["Rxy_A (pads 6-3)", "n_fan_1e15"] * 1e15
n_xx = sd.loc["Rxx_S (pads 6-11)", "n_fan_1e15"] * 1e15
fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
B = fs["absB"]
ax[0].plot(B, fs["Rxy_A"], "-", color="0.5", label=r"$R_{xy}^A$ as read")
ax[0].plot(B, fs["Rxy_A"] / k, "-", color="k", lw=1.4, label=fr"$R_{{xy}}^A / k$, k = {k:.3f}")
for nu in range(4, 16, 2):
    ax[0].axhline(R_K / nu, color="C0", lw=0.6, ls="--")
    ax[0].text(0.05, R_K / nu + 60, fr"$R_K/{nu}$", color="C0", fontsize=9)
ax[0].plot(pl["B_plateau"], pl["Rxy_true"], "o", color="C3", ms=7, mfc="none", label="plateau centres (min dR/dB)")
ax[0].set_ylabel(r"$R_{xy}$ [$\Omega$]"); ax[0].set_ylim(0, 7000)
ax[0].set_title(r"Hall resistance, $V_g$ = +0.5 V (pads 6-3)"); ax[0].legend(loc="lower right")
ax[1].plot(B, fs["Rxx_S"], "-", color="k", label=r"$R_{xx}^S$ (pads 6-11)")
ymax = 7500
for nu in range(4, 18, 2):
    b1 = h * n_xx / (e * nu)
    if b1 < 5.5:
        ax[1].axvline(b1, color="C0", ls=":", lw=0.9); ax[1].text(b1, ymax * 0.97, str(nu), color="C0", ha="center", fontsize=9)
    b2 = h * n_qh / (e * nu)
    if b2 < 5.5:
        ax[1].axvline(b2, color="C3", ls="--", lw=0.7); ax[1].text(b2, ymax * 0.05, str(nu), color="C3", ha="center", fontsize=9)
ax[1].plot([], [], ":", color="C0", label=fr"even $\nu$ for $n_{{xx}}$ = {n_xx/1e15:.2f}e15 m$^{{-2}}$ (top labels)")
ax[1].plot([], [], "--", color="C3", label=fr"even $\nu$ for $n_{{QH}}$ = {n_qh/1e15:.2f}e15 m$^{{-2}}$ (bottom labels)")
ax[1].set_ylim(0, ymax); ax[1].set_xlabel("|B| [T]"); ax[1].set_ylabel(r"$R_{xx}^S$ [$\Omega$] (as read)")
ax[1].set_title(r"Longitudinal resistance, $V_g$ = +0.5 V"); ax[1].legend(loc="upper left")
plt.tight_layout(); plt.savefig(fig_path("fig04_hall_plateaus.pdf"))
