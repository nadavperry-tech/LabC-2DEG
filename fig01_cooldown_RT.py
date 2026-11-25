"""fig01_cooldown_RT.py -- Fig. 1: cool-down R(T) with the two-channel fit (plan section 1).
Inputs: tables/rt_clean.csv, rt_fit_params.csv, rt_arrhenius.csv (rt_model.py). Output: figures/fig01_cooldown_RT.pdf"""
import numpy as np
import pandas as pd

from params import fig_path, plot_style, tab_path

plt = plot_style()
d = pd.read_csv(tab_path("rt_clean.csv"))
p = pd.read_csv(tab_path("rt_fit_params.csv")).iloc[0]
a = pd.read_csv(tab_path("rt_arrhenius.csv"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

ax[0].plot(d["T"], d["R"], "o", ms=2, color="0.3", label="data (spikes removed)")
ax[0].plot(d.loc[d["spike"], "T"], d.loc[d["spike"], "R_raw"], "x", color="r", ms=5, label="removed spikes")
o = d.sort_values("T")
ax[0].plot(o["T"], o["R_fit"], "-", color="C0", lw=1.6, label="two-channel fit")
ax[0].plot(o["T"], o["R_2D_fit"], "--", color="C1", label=r"$R_{2D}$ (impurity + phonons)")
ax[0].plot(o["T"], 1 / o["G_par_fit"], ":", color="C2", lw=1.6, label=r"$R_{par}$ (doped AlGaAs)")
ax[0].set_ylim(0, 8000); ax[0].set_xlabel("Temperature T [K]"); ax[0].set_ylabel(r"Resistance R$_{6-11}$ [$\Omega$]")
ax[0].set_title("Cool-down at B = 0, $V_g$ = 0"); ax[0].legend(loc="upper right")

R0 = p["R0_ohm"]
lo = o[o["T"] < 120]
ax[1].loglog(lo["T"], lo["R"] - R0, "o", ms=2, color="0.3", label=r"data $R - R_0$")
ax[1].loglog(lo["T"], lo["R_ac_fit"], "--", color="C1", label="acoustic (Bloch-Grüneisen)")
ax[1].loglog(lo["T"], lo["R_LO_fit"], "-.", color="C3", label="LO phonons")
ax[1].loglog(lo["T"], lo["R_2D_fit"] - R0, "-", color="C0", label="sum")
ax[1].axvline(p["T_BG_K"], color="k", lw=0.8, ls=":"); ax[1].text(p["T_BG_K"] * 1.05, 2, r"$T_{BG}$", fontsize=10)
ax[1].set_ylim(1, 5000); ax[1].set_xlabel("Temperature T [K]"); ax[1].set_ylabel(r"$R - R_0$ [$\Omega$]")
ax[1].set_title(fr"Phonon part, $R_0$ = {R0:.0f} $\Omega$"); ax[1].legend(loc="upper left")

ax[2].plot(a["inv_kT_per_eV"], a["ln_Gpar"], "o", ms=2, color="0.3", label=r"$\ln G_{par}$ = ln(1/R - 1/R$_{2D}$)")
ax2 = ax[2].twinx()
ax2.plot(a["inv_kT_per_eV"], a["Ea_local_eV"], "-", color="C2", label=r"local $E_a$")
ax2.set_ylabel(r"local activation energy $E_a$ [eV]", color="C2")
ax[2].set_xlabel(r"$1/k_BT$ [eV$^{-1}$]"); ax[2].set_ylabel(r"$\ln (G_{par}$ [S]$)$")
ax[2].set_title(fr"Parallel channel, fit $E_a$ = {p['Ea_eV']*1e3:.0f} meV")
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, loc="lower left")
plt.tight_layout(); plt.savefig(fig_path("fig01_cooldown_RT.pdf"))
