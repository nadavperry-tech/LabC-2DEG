"""fig08_colourmap_landau_fan.py -- Fig. 8: R_xx^S(Vg, |B|) colour map with filling-factor lines (plan 5, A10).
Inputs: tables/gate_symmetrized.csv, gate_capacitance.csv, gate_sdh_minima.csv. Output: figures/fig08_colourmap_landau_fan.pdf"""
import numpy as np
import pandas as pd

from params import e, fig_path, h, plot_style, tab_path

plt = plot_style()
gs = pd.read_csv(tab_path("gate_symmetrized.csv"))
c = pd.read_csv(tab_path("gate_capacitance.csv"))
cs = c[c["method"].str.startswith("SdH minima")].iloc[0]
m = pd.read_csv(tab_path("gate_sdh_minima.csv"))
fields = sorted(gs["absB"].unique())
V = np.round(np.arange(-0.49, 0.495, 0.01), 3)
Z = []
for b in fields:
    q = gs[gs["absB"] == b].groupby("Vg")["Rxx_S"].mean()
    r = np.interp(V, q.index, q.values)
    Z.append(r / np.median(r))                        # normalised per field to show the oscillation
Z = np.array(Z)
edges_B = np.concatenate([[0.0], (np.array(fields[:-1]) + np.array(fields[1:])) / 2, [6.0]])
edges_V = np.concatenate([V - 0.005, [V[-1] + 0.005]])
fig, ax = plt.subplots(figsize=(9, 6))
pc = ax.pcolormesh(edges_V, edges_B, Z, cmap="viridis", shading="flat")
plt.colorbar(pc, ax=ax, label=r"$R_{xx}^S / \mathrm{median}_{V_g}(R_{xx}^S)$ at each |B|")
bb = np.linspace(0.3, 6, 100)
for nu in range(2, 22, 2):
    vg = (nu * e * bb / h - cs["n0"]) / cs["dn_dVg"]
    ok = (vg > -0.5) & (vg < 0.5)
    ax.plot(vg[ok], bb[ok], "w--", lw=0.8)
    if ok.any():
        j = np.flatnonzero(ok)[-1]
        ax.text(vg[j], min(bb[j], 5.8), fr"$\nu$={nu}", color="w", fontsize=8, ha="center", va="bottom")
for dr, mk in (("down", "v"), ("up", "^")):
    q = m[m["dir"] == dr]
    ax.plot(q["Vg_min"], q["absB"] + (0.12 if dr == "up" else -0.12), mk, color="r", ms=6, mfc="none",
            label=f"R$_{{xx}}$ minima, {dr} sweeps")
ax.set_ylim(0, 6); ax.set_xlim(-0.5, 0.5)
ax.set_xlabel(r"Gate voltage $V_g$ [V]"); ax.set_ylabel("|B| [T] (rows: the six measured fields)")
ax.set_title(r"$R_{xx}^S(V_g,B)$ with Landau-fan lines from the SdH-minima fit")
ax.legend(loc="lower left")
plt.tight_layout(); plt.savefig(fig_path("fig08_colourmap_landau_fan.pdf"))
