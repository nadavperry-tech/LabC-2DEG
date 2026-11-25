"""fig07_gate_sweeps.py -- Fig. 7: R_xy^A/k and R_xx^S vs Vg at each |B| with filling-factor labels (plan 5).
Inputs: tables/gate_symmetrized.csv, gate_sdh_minima.csv, hall_calibration.csv. Output: figures/fig07_gate_sweeps.pdf"""
import pandas as pd

from hall import load_k
from params import LINESTYLES, MARKERS, R_K, fig_path, plot_style, tab_path

plt = plot_style()
gs = pd.read_csv(tab_path("gate_symmetrized.csv"))
m = pd.read_csv(tab_path("gate_sdh_minima.csv"))
k, _ = load_k(late=True)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for i, b in enumerate(sorted(gs["absB"].unique())):
    q = gs[gs["absB"] == b].groupby("Vg", as_index=False)[["Rxx_S", "Rxy_A"]].mean()   # mean of down and up
    ax[0].plot(q["Vg"], q["Rxy_A"] / k, LINESTYLES[i % 4], color=f"C{i}", label=f"|B| = {b} T")
    ax[1].plot(q["Vg"], q["Rxx_S"], LINESTYLES[i % 4], color=f"C{i}", label=f"|B| = {b} T")
    mm = m[(m["absB"] == b) & (m["dir"] == "down")]
    for _, r in mm.iterrows():
        y = q["Rxx_S"].iloc[(q["Vg"] - r["Vg_min"]).abs().argmin()]
        ax[1].plot(r["Vg_min"], y, MARKERS[i], color=f"C{i}", ms=6, mfc="none")
        ax[1].annotate(fr"$\nu$={int(r['nu'])}", (r["Vg_min"], y), textcoords="offset points", xytext=(0, -14),
                       ha="center", fontsize=8, color=f"C{i}")
for nu in (4, 6, 8, 10, 12):
    ax[0].axhline(R_K / nu, color="k", lw=0.5, ls=":")
    ax[0].text(0.52, R_K / nu, fr"$R_K/{nu}$", fontsize=8, va="center")
ax[0].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[0].set_ylabel(r"$R_{xy}^A / k$ [$\Omega$]")
ax[0].set_title("Hall resistance vs gate (mean of down and up sweeps)"); ax[0].legend()
ax[1].set_yscale("log"); ax[1].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[1].set_ylabel(r"$R_{xx}^S$ [$\Omega$] (as read)")
ax[1].set_title(r"Longitudinal resistance vs gate; open markers: SdH minima (Hall-region $\nu$)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(fig_path("fig07_gate_sweeps.pdf"))
