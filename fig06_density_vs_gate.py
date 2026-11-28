"""fig06_density_vs_gate.py -- Fig. 6: n(Vg) from Hall (k-corrected) and from SdH minima; lag/hysteresis (plan 3).
Inputs: tables/gate_density.csv, gate_capacitance.csv, gate_sdh_minima.csv. Output: figures/fig06_density_vs_gate.pdf"""
import numpy as np
import pandas as pd

from params import FULL_W, LEGEND_BOX, fig_path, panel_label, plot_style, soft_frame, tab_path

plt = plot_style()
d = pd.read_csv(tab_path("gate_density.csv"))
c = pd.read_csv(tab_path("gate_capacitance.csv"))
m = pd.read_csv(tab_path("gate_sdh_minima.csv"))

# Three panels side by side on the same V_g axis and one legend for all of them below. Hall traces: colour and a
# sparse marker = field (circle 0.5 T, square 1.5 T, so the fields stay apart in greyscale), solid = down,
# dashed = up; the SdH minima are triangles. (b) = (a) minus the Hall linear fit. To keep (b) readable it shows the
# Hall density averaged over the down and up sweeps (the lag cancels; the down - up difference is panel (c)), and
# each down/up pair of SdH minima (same B and nu, hence the same density) is joined by a thin line: the pair is
# displaced along V_g by the lag of the lock-in output, which the residual plot turns into a vertical offset.
fig, ax = plt.subplots(1, 3, figsize=(FULL_W, 2.95), sharex=True)   # figure*: full page width
a0, a0r = ax[0], ax[1]
ch = c.iloc[0]; cs = c[c["method"].str.startswith("SdH minima")].iloc[0]
n_fit = lambda vg: ch["n0"] + ch["dn_dVg"] * vg
STY = {(0.5, "down"): ("C0", "-"), (0.5, "up"): ("C0", "--"), (1.5, "down"): ("C1", "-"), (1.5, "up"): ("C1", "--")}
MK = {0.5: dict(marker="o", markevery=(0, 16)), 1.5: dict(marker="s", markevery=(8, 16))}
MS = dict(ms=2.6, mfc="white", mew=0.7)
H = {}
for (b, dr), (col, ls) in STY.items():
    q = d[(d["absB"] == b) & (d["dir"] == dr)].sort_values("Vg")
    H[(b, dr)], = a0.plot(q["Vg"], q["n_true"] / 1e15, ls, color=col, lw=0.9, **MK[b], **MS, label=fr"Hall $k_{{late}}\,n_H$, {b} T, {dr}")
for b in (0.5, 1.5):                                   # (b): mean of the down and up sweeps at equal V_g
    dn = d[(d["absB"] == b) & (d["dir"] == "down")].set_index("Vg")["n_true"]
    up = d[(d["absB"] == b) & (d["dir"] == "up")].set_index("Vg")["n_true"]
    j = dn.index.intersection(up.index).sort_values()
    mean = (dn.loc[j] + up.loc[j]) / 2
    a0r.plot(j, (mean - n_fit(j)) / 1e15, "-", color="C0" if b == 0.5 else "C1", lw=1.0, **MK[b], **MS)
for (bb, nu), pr in m.groupby(["absB", "nu"]):            # (b): join each down/up pair of SdH minima
    if len(pr) == 2:
        a0r.plot(pr["Vg_min"], (pr["n_nu"] - n_fit(pr["Vg_min"])) / 1e15, "-", color="0.6", lw=0.6, zorder=1)
for dr, mk in (("down", "v"), ("up", "^")):
    q = m[m["dir"] == dr]
    H[("SdH", dr)], = a0.plot(q["Vg_min"], q["n_nu"] / 1e15, mk, color="k", ms=4, mfc="none", mew=0.7, label=fr"SdH minima $\nu eB/h$, {dr}")
    a0r.plot(q["Vg_min"], (q["n_nu"] - n_fit(q["Vg_min"])) / 1e15, mk, color="k", ms=4, mfc="white", mew=0.7, zorder=3)
v = np.linspace(-0.5, 0.5, 10)
H["fit_h"], = a0.plot(v, n_fit(v) / 1e15, "-", color="0.55", lw=1.6, zorder=0,
                      label=fr"linear fit, Hall: {ch['dn_dVg']/1e15:.2f}e15 m$^{{-2}}$V$^{{-1}}$")
H["fit_s"], = a0.plot(v, (cs["n0"] + cs["dn_dVg"] * v) / 1e15, ":", color="k", lw=1,
                      label=fr"linear fit, SdH: {cs['dn_dVg']/1e15:.2f}e15 m$^{{-2}}$V$^{{-1}}$")
a0r.plot(v, (cs["n0"] + cs["dn_dVg"] * v - n_fit(v)) / 1e15, ":", color="k", lw=1)
a0r.axhline(0, color="0.55", lw=1.6, zorder=0)
a0.set_ylabel(r"Density n [10$^{15}$ m$^{-2}$]"); a0.set_title("Density (Hall region)")
a0r.set_ylabel(r"$n - n_{fit,Hall}$ [10$^{15}$ m$^{-2}$]"); a0r.set_title("Minus the Hall fit (Hall: sweep average)")
for b in (0.5, 1.5):
    dn = d[(d["absB"] == b) & (d["dir"] == "down")].set_index("Vg")["n_true"]
    up = d[(d["absB"] == b) & (d["dir"] == "up")].set_index("Vg")["n_true"]
    j = dn.index.intersection(up.index)
    ax[2].plot(j, (dn.loc[j] - up.loc[j]) / 1e15, "-", color="C0" if b == 0.5 else "C1", lw=0.9, **MK[b], **MS)
hy = pd.read_csv(tab_path("gate_hysteresis.csv")).iloc[0]
lag = 2 * hy["tau_eff_s"] * hy["rate_mV_per_s"] * 1e-3 * ch["dn_dVg"] / 1e15
H["lag"] = ax[2].axhline(lag, color="k", ls="--", lw=0.8, label=fr"lock-in lag, $\tau_{{eff}}$ = {hy['tau_eff_s']:.1f} s (from SdH minima)")
ax[2].axhline(0, color="k", lw=0.5)
ax[2].set_ylabel(r"$n_{down} - n_{up}$ [10$^{15}$ m$^{-2}$]"); ax[2].set_title("Down minus up sweep")
for a_ in ax: a_.set_xlabel(r"Gate voltage $V_g$ [V]"); a_.tick_params(labelbottom=True)
for a_, l_ in zip(ax, "abc"): panel_label(a_, l_)
# one legend for the whole figure, below the panels (filled column by column)
order = [(0.5, "down"), (0.5, "up"), ("SdH", "down"), (1.5, "down"), (1.5, "up"), ("SdH", "up"), "fit_h", "fit_s", "lag"]
soft_frame(fig.legend([H[o] for o in order], [H[o].get_label() for o in order], loc="outside lower center", ncol=3, **LEGEND_BOX,
           labelspacing=0.3, columnspacing=1.5))
plt.savefig(fig_path("fig06_density_vs_gate.pdf"))
