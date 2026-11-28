"""fig07_gate_sweeps.py -- Fig. 7: R_xy^A/k and R_xx^S vs Vg at each |B| with filling-factor labels (plan 5).
Inputs: tables/gate_symmetrized.csv, gate_sdh_minima.csv, hall_calibration.csv. Output: figures/fig07_gate_sweeps.pdf"""
import matplotlib.patheffects as pe
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from hall import load_k
from params import FULL_W, LINESTYLES, panel_label, MARKERS, R_K, fig_path, plot_style, tab_path

plt = plot_style()
gs = pd.read_csv(tab_path("gate_symmetrized.csv"))
m = pd.read_csv(tab_path("gate_sdh_minima.csv"))
k, _ = load_k(late=True)
fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.3))           # figure*: full page width
OUTLINE = [pe.withStroke(linewidth=2, foreground="white")]   # white halo keeps coloured labels legible over lines
for i, b in enumerate(sorted(gs["absB"].unique())):
    q = gs[gs["absB"] == b].groupby("Vg", as_index=False)[["Rxx_S", "Rxy_A"]].mean()   # mean of down and up
    ax[0].plot(q["Vg"], q["Rxy_A"] / k, LINESTYLES[i % 4], color=f"C{i}", label=f"|B| = {b} T")
    ax[1].plot(q["Vg"], q["Rxx_S"], LINESTYLES[i % 4], color=f"C{i}", label=f"|B| = {b} T")
    mm = m[(m["absB"] == b) & (m["dir"] == "down")]
    for _, r in mm.iterrows():
        y = q["Rxx_S"].iloc[(q["Vg"] - r["Vg_min"]).abs().argmin()]
        ax[1].plot(r["Vg_min"], y, MARKERS[i], color=f"C{i}", ms=4, mfc="none")
        ax[1].annotate(fr"$\nu$={int(r['nu'])}", (r["Vg_min"], y), textcoords="offset points", xytext=(0, -10),
                       ha="center", fontsize=7.5, fontweight="bold", color=f"C{i}", path_effects=OUTLINE)
right_margin = blended_transform_factory(ax[0].transAxes, ax[0].transData)   # x in axes units, y in ohm
NUDGE = {8: 1.5, 12: -3.0}                         # pt: R_K/10 and R_K/12 are only ~6 pt apart at print size
for nu in (4, 6, 8, 10, 12):
    ax[0].axhline(R_K / nu, color="k", lw=0.6, ls=":")
    ax[0].annotate(fr"$R_K/{nu}$", (1.01, R_K / nu), xycoords=right_margin, xytext=(0, NUDGE.get(nu, 0)),
                   textcoords="offset points", fontsize=8, va="center", ha="left", annotation_clip=False)
ax[0].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[0].set_ylabel(r"$R_{xy}^A / k$ [$\Omega$]")
ax[0].set_title("Hall resistance vs gate\n(mean of down and up sweeps)"); ax[0].legend()
ax[1].set_yscale("log"); ax[1].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[1].set_ylabel(r"$R_{xx}^S$ [$\Omega$] (as read)")
ax[1].set_ylim(top=2.4e4)                          # room for a one-block legend above the 5.5 T trace
ax[1].set_title("Longitudinal resistance vs gate\n" + r"open markers: SdH minima (Hall-region $\nu$)")
ax[1].legend(loc="upper center", ncol=3)
for a_, l_ in zip(ax, "ab"): panel_label(a_, l_)
plt.savefig(fig_path("fig07_gate_sweeps.pdf"))
