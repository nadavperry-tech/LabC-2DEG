"""fig04_hall_plateaus.py -- Fig. 4: R_xy^A and R_xx^S vs B at Vg = +0.5 V with filling factors (plan 2a/2b).
Inputs: tables/field_symmetrized_vg+0.5.csv, hall_plateaus.csv, hall_calibration.csv, sdh_results.csv.
Output: figures/fig04_hall_plateaus.pdf"""
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from hall import load_k
import os

from params import COL_W, FULL_W, R_K, e, fig_path, h, legend_below, panel_label, plot_style, tab_path

WIDE = os.environ.get("FIG04_WIDE") == "1"     # alternative layout: the two panels side by side, full page width

plt = plot_style()
fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
pl = pd.read_csv(tab_path("hall_plateaus.csv"))
sd = pd.read_csv(tab_path("sdh_results.csv")).set_index("pair")
k, sk = load_k()
n_qh = sd.loc["Rxy_A (pads 6-3)", "n_fan_1e15"] * 1e15
n_xx = sd.loc["Rxx_S (pads 6-11)", "n_fan_1e15"] * 1e15
if WIDE:
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.25), sharex=True)                      # figure*
    ax[0].set_xlabel("|B| [T]")
else:
    fig, ax = plt.subplots(2, 1, figsize=(COL_W, 5.6), sharex=True, height_ratios=[1, 1.15])   # one column
B = fs["absB"]
ax[0].plot(B, fs["Rxy_A"], "-", color="0.5", label=r"$R_{xy}^A$ as read")
ax[0].plot(B, fs["Rxy_A"] / k, "-", color="k", lw=1.1, label=fr"$R_{{xy}}^A / k$, k = {k:.3f}")
# R_K/nu labels in the right margin; R_K/12 and R_K/14 are nudged apart
right_margin = blended_transform_factory(ax[0].transAxes, ax[0].transData)
NUDGE = {10: 2.5, 14: -3.0}
for nu in range(4, 16, 2):
    ax[0].axhline(R_K / nu, color="C0", lw=0.6, ls="--")
    ax[0].annotate(fr"$R_K/{nu}$", (1.01, R_K / nu), xycoords=right_margin, xytext=(0, NUDGE.get(nu, 0)),
                   textcoords="offset points", color="C0", fontsize=8, va="center", ha="left", annotation_clip=False)
ax[0].plot(pl["B_plateau"], pl["Rxy_true"], "o", color="C3", ms=4.5, mfc="none", label="plateau centres (min dR/dB)")
ax[0].set_ylabel(r"$R_{xy}$ [$\Omega$]"); ax[0].set_ylim(0, 7000)
ax[0].set_title(r"Hall resistance, $V_g$ = +0.5 V (pads 6-3)"); legend_below(ax[0])          # legends below the panels: nothing to cover
ax[1].plot(B, fs["Rxx_S"], "-", color="k", label=r"$R_{xx}^S$ (pads 6-11)")
ymax = 7300                                    # just above 7 kOhm; the label rows sit above the data at the label fields (<= 4.94 T)
# Even-nu labels: one colour per density; within a density, alternate nu go to two staggered rows so that
# neighbouring labels (e.g. nu = 16 and 14 of n_xx, 0.18 T apart) do not overlap at print size.
ROW = {"xx": (0.975, 0.925), "qh": (0.870, 0.820)}
for nu in range(4, 18, 2):
    b1 = h * n_xx / (e * nu)
    if b1 < 5.5:
        ax[1].axvline(b1, color="C0", ls=":", lw=0.9)
        ax[1].text(b1, ymax * ROW["xx"][(nu // 2) % 2], str(nu), color="C0", ha="center", va="top", fontsize=7.5, fontweight="bold",
                   bbox=dict(fc="white", ec="none", pad=0.4))
    b2 = h * n_qh / (e * nu)
    if b2 < 5.5:
        ax[1].axvline(b2, color="C3", ls="--", lw=0.7)
        ax[1].text(b2, ymax * ROW["qh"][(nu // 2) % 2], str(nu), color="C3", ha="center", va="top", fontsize=7.5, fontweight="bold",
                   bbox=dict(fc="white", ec="none", pad=0.4))
ax[1].plot([], [], ":", color="C0", label=fr"even $\nu$, $n_{{xx}}$ = {n_xx/1e15:.2f}e15 m$^{{-2}}$")   # blue labels (upper rows)
ax[1].plot([], [], "--", color="C3", label=fr"even $\nu$, $n_{{QH}}$ = {n_qh/1e15:.2f}e15 m$^{{-2}}$")   # red labels (lower rows)
ax[1].set_ylim(0, ymax); ax[1].set_xlabel("|B| [T]"); ax[1].set_ylabel(r"$R_{xx}^S$ [$\Omega$] (as read)")
ax[1].set_title(r"Longitudinal resistance, $V_g$ = +0.5 V")
legend_below(ax[1])
panel_label(ax[0], "a"); panel_label(ax[1], "b")
plt.savefig(fig_path("fig04_hall_plateaus_wide.pdf" if WIDE else "fig04_hall_plateaus.pdf"))
