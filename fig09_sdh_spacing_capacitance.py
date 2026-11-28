"""fig09_sdh_spacing_capacitance.py -- Fig. 9: gate spacing of consecutive SdH minima vs B -> C (plan 5, A6).
Inputs: tables/gate_sdh_spacing.csv, gate_capacitance.csv. Output: figures/fig09_sdh_spacing_capacitance.pdf"""
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from params import COL_W, fig_path, plot_style, tab_path

plt = plot_style()
sp = pd.read_csv(tab_path("gate_sdh_spacing.csv"))
c = pd.read_csv(tab_path("gate_capacitance.csv"))
r = c[c["method"].str.startswith("spacing")].iloc[0]
fig, ax = plt.subplots(figsize=(COL_W, 2.7))                   # one column
for dr, mk in (("down", "v"), ("up", "^")):
    q = sp[sp["dir"] == dr]
    ax.plot(q["absB"], 1e3 * q["dVg"], mk, ms=5, mfc="none", label=f"{dr} sweeps")
# One label per (field, filling-factor pair), at the mean of its down/up points, on the side of the points away
# from the fit line (left if the points lie above it, right if below). Two labels on the same side at one field
# (12-14 and 14-16 at 1.5 T): the lower one moves to the other side, slightly below its point.
lab = sp.groupby(["absB", "nu_pair"], as_index=False)["dVg"].mean().sort_values(["absB", "dVg"])
lab["right"] = 1e3 * lab["dVg"] < 1e3 * r["dVg_per_T"] * lab["absB"]
lab["dy"] = 0.0
for _, g in lab.groupby(["absB", "right"]):
    if len(g) > 1:
        lab.loc[g.index[0], "right"] = not g["right"].iloc[0]; lab.loc[g.index[0], "dy"] = -4.0
for b, g in lab.groupby("absB"):
    for _, row in g.iterrows():
        right, dy = row["right"], row["dy"]
        ax.annotate(r"$\nu$ = " + row["nu_pair"].replace("-", "–"), (b, 1e3 * row["dVg"]), textcoords="offset points",
                    xytext=(6 if right else -6, dy), ha="left" if right else "right", va="center", fontsize=7.5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
x = np.linspace(0, 5.8, 10)
ax.plot(x, 1e3 * r["dVg_per_T"] * x, "-", color="k",
        label=fr"$\Delta V_g = 2e^2B/(hC)$:" + "\n" + fr"C = {r['C_uF_per_cm2']*1e3:.1f} nF/cm$^2$")
ax.set_xlim(0, 6); ax.set_ylim(0)
ax.set_xlabel("|B| [T]"); ax.set_ylabel(r"$\Delta V_g$ between consecutive" + "\n" + "minima [mV]")
ax.set_title(r"SdH minima spacing in $V_g$ ($\Delta\nu$ = 2)"); ax.legend(loc="upper left")
plt.savefig(fig_path("fig09_sdh_spacing_capacitance.pdf"))
