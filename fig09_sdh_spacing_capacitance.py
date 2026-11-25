"""fig09_sdh_spacing_capacitance.py -- Fig. 9: gate spacing of consecutive SdH minima vs B -> C (plan 5, A6).
Inputs: tables/gate_sdh_spacing.csv, gate_capacitance.csv. Output: figures/fig09_sdh_spacing_capacitance.pdf"""
import numpy as np
import pandas as pd

from params import fig_path, plot_style, tab_path

plt = plot_style()
sp = pd.read_csv(tab_path("gate_sdh_spacing.csv"))
c = pd.read_csv(tab_path("gate_capacitance.csv"))
r = c[c["method"].str.startswith("spacing")].iloc[0]
fig, ax = plt.subplots(figsize=(7, 5))
for dr, mk in (("down", "v"), ("up", "^")):
    q = sp[sp["dir"] == dr]
    ax.plot(q["absB"], 1e3 * q["dVg"], mk, ms=8, mfc="none", label=f"{dr} sweeps")
    for _, row in q.iterrows():
        ax.annotate(row["nu_pair"], (row["absB"], 1e3 * row["dVg"]), textcoords="offset points", xytext=(6, 0), fontsize=8)
x = np.linspace(0, 5.8, 10)
ax.plot(x, 1e3 * r["dVg_per_T"] * x, "-", color="k",
        label=fr"$\Delta V_g = 2e^2B/(hC)$: C = {r['C_uF_per_cm2']*1e3:.1f} nF/cm$^2$")
ax.set_xlim(0, 6); ax.set_ylim(0)
ax.set_xlabel("|B| [T]"); ax.set_ylabel(r"$\Delta V_g$ between consecutive minima [mV]")
ax.set_title(r"SdH minima spacing in $V_g$ ($\Delta\nu$ = 2)"); ax.legend()
plt.tight_layout(); plt.savefig(fig_path("fig09_sdh_spacing_capacitance.pdf"))
