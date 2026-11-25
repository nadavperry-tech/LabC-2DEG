"""fig06_density_vs_gate.py -- Fig. 6: n(Vg) from Hall (k-corrected) and from SdH minima; lag/hysteresis (plan 3).
Inputs: tables/gate_density.csv, gate_capacitance.csv, gate_sdh_minima.csv. Output: figures/fig06_density_vs_gate.pdf"""
import numpy as np
import pandas as pd

from params import MARKERS, fig_path, plot_style, tab_path

plt = plot_style()
d = pd.read_csv(tab_path("gate_density.csv"))
c = pd.read_csv(tab_path("gate_capacitance.csv"))
m = pd.read_csv(tab_path("gate_sdh_minima.csv"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for i, (b, dr) in enumerate(((0.5, "down"), (0.5, "up"), (1.5, "down"), (1.5, "up"))):
    q = d[(d["absB"] == b) & (d["dir"] == dr)]
    ax[0].plot(q["Vg"], q["n_true"] / 1e15, MARKERS[i], ms=2.5, label=fr"Hall $k\,n_H$, |B| = {b} T, {dr}")
for dr, mk in (("down", "v"), ("up", "^")):
    q = m[m["dir"] == dr]
    ax[0].plot(q["Vg_min"], q["n_nu"] / 1e15, mk, color="k", ms=8, mfc="none", label=fr"SdH minima $\nu eB/h$, {dr}")
v = np.linspace(-0.5, 0.5, 10)
ch = c.iloc[0]; cs = c[c["method"].str.startswith("SdH minima")].iloc[0]
ax[0].plot(v, (ch["n0"] + ch["dn_dVg"] * v) / 1e15, "-", color="C0", lw=1, label=fr"linear fit (Hall): dn/dV$_g$ = {ch['dn_dVg']/1e15:.2f}e15 V$^{{-1}}$")
ax[0].plot(v, (cs["n0"] + cs["dn_dVg"] * v) / 1e15, "--", color="k", lw=1, label=fr"linear fit (SdH): dn/dV$_g$ = {cs['dn_dVg']/1e15:.2f}e15 V$^{{-1}}$")
ax[0].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[0].set_ylabel(r"Density n [10$^{15}$ m$^{-2}$]")
ax[0].set_title("Electron density vs gate voltage (Hall region)"); ax[0].legend(fontsize=8)
for b, mk in ((0.5, "o"), (1.5, "s")):
    dn = d[(d["absB"] == b) & (d["dir"] == "down")].set_index("Vg")["n_true"]
    up = d[(d["absB"] == b) & (d["dir"] == "up")].set_index("Vg")["n_true"]
    j = dn.index.intersection(up.index)
    ax[1].plot(j, (dn.loc[j] - up.loc[j]) / 1e15, mk + "-", ms=3, lw=0.6, label=f"|B| = {b} T")
hy = pd.read_csv(tab_path("gate_hysteresis.csv")).iloc[0]
lag = 2 * hy["tau_eff_s"] * hy["rate_mV_per_s"] * 1e-3 * ch["dn_dVg"] / 1e15
ax[1].axhline(lag, color="k", ls="--", lw=0.8, label=fr"lock-in lag, $\tau_{{eff}}$ = {hy['tau_eff_s']:.1f} s (from SdH minima)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[1].set_ylabel(r"$n_{down} - n_{up}$ [10$^{15}$ m$^{-2}$]")
ax[1].set_title("Down minus up sweep at equal $V_g$"); ax[1].legend()
plt.tight_layout(); plt.savefig(fig_path("fig06_density_vs_gate.pdf"))
