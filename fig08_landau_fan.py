"""fig08_landau_fan.py -- Fig. 8: Landau fan in the (Vg, |B|) plane (plan 5, A10).
Inputs: tables/gate_capacitance.csv, gate_sdh_minima.csv. Output: figures/fig08_landau_fan.pdf

What it shows: every R_xx^S minimum found in the gate sweeps, placed at its (Vg, |B|), on top of the lines of
constant even filling factor nu = h n(Vg) / (e B), with n(Vg) = n0 + (dn/dVg) Vg from the linear fit to the SdH
minima. If the gate only changes the density, minima of the same nu measured at different fields must fall on one
line of the fan; the lines fan out from the gate voltage where n would reach zero (off the plot, V_th ~ -1.3 V).
Grey dotted rows mark the six fields at which gate sweeps were taken: the data exist only on those rows, which
is why this replaced the earlier colour map (six rows stretched into 1 T bands looked like continuous data).
nu = 2 lies above the highest measured field (5.5 T) and is drawn only to show where it would start."""
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

from params import COL_W, LEGEND_BOX, e, fig_path, h, plot_style, soft_frame, tab_path

NU = range(2, 18, 2)
B_TOP = 7.0          # T, high enough that the nu = 2 line is a visible segment (it starts at ~5.8 T)

plt = plot_style()
c = pd.read_csv(tab_path("gate_capacitance.csv"))
cs = c[c["method"].str.startswith("SdH minima")].iloc[0]
m = pd.read_csv(tab_path("gate_sdh_minima.csv"))
n_of = lambda vg: cs["n0"] + cs["dn_dVg"] * vg
halo = [pe.withStroke(linewidth=2, foreground="white")]

fig, ax = plt.subplots(figsize=(COL_W, 3.7))                   # one column
for i, b in enumerate(sorted(m["absB"].unique().tolist() + [0.5])):
    ax.axhline(b, color="0.6", ls=":", lw=1.0, zorder=0, label="fields of the gate sweeps" if i == 0 else None)
vg = np.linspace(-0.5, 0.5, 400)
for nu in NU:
    B = h * n_of(vg) / (e * nu)
    ok = B <= B_TOP
    ax.plot(vg[ok], B[ok], "-", color="0.15", lw=1.2 if nu == 2 else 0.8, zorder=1)
    if B[-1] <= B_TOP:     # line leaves through the right edge: label in the right margin
        ax.annotate(fr"$\nu$ = {nu}", (0.5 + 0.012, B[-1]), xytext=(0, {14: 1.0, 16: -2.5}.get(nu, 0)), textcoords="offset points",
                        ha="left", va="center", fontsize=7.5, fontweight="bold", annotation_clip=False)   # 14/16 nudged apart
    else:                  # line leaves through the top edge: label just inside the top, beside the line
        x_top = (nu * e * B_TOP / h - cs["n0"]) / cs["dn_dVg"]
        ax.text(x_top + 0.015, B_TOP - 0.08, fr"$\nu$ = {nu}", ha="left", va="top", fontsize=7.5, fontweight="bold",
                path_effects=halo)
for dr, mk, col in (("down", "v", "C0"), ("up", "^", "C1")):
    q = m[m["dir"] == dr]
    ax.plot(q["Vg_min"], q["absB"], mk, color=col, ms=5, mfc="none", mew=1.0, zorder=3, label=fr"$R_{{xx}}$ minima, {dr} sweeps")
ax.plot([], [], "-", color="0.15", label=r"$\nu = h\,n(V_g)/(eB)$, $n(V_g)$" + "\nfrom the SdH-minima fit")
ax.set_xlim(-0.5, 0.5); ax.set_ylim(0, B_TOP)
ax.set_xlabel(r"Gate voltage $V_g$ [V]"); ax.set_ylabel("|B| [T]")
ax.set_title(r"Landau fan: $R_{xx}^S$ minima in the ($V_g$, |B|) plane")
hs, ls = ax.get_legend_handles_labels(); order = [1, 2, 0, 3]          # column 1: minima; column 2: guides, fan
soft_frame(fig.legend([hs[i] for i in order], [ls[i] for i in order], loc="outside lower center", ncol=2, **LEGEND_BOX,
           labelspacing=0.3, columnspacing=1.0, handlelength=1.6))   # below the axes, over the full figure width
ax.grid(False)
plt.savefig(fig_path("fig08_landau_fan.pdf"))
