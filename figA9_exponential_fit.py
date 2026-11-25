"""figA9_exponential_fit.py -- Fig. A1 (appendix A9): exponential R_xx(Vg) fits and why they are not used.
Inputs: data/RVg-IV-12_14_49, tables/gate_symmetrized.csv, appendix_A9_fits.csv, b0_values.csv.
Output: figures/figA9_exponential_fit.pdf"""
import numpy as np
import pandas as pd

from gate import expo, power
from load_data import load
from params import fig_path, plot_style, tab_path

plt = plot_style()
a = pd.read_csv(tab_path("appendix_A9_fits.csv"))
gs = pd.read_csv(tab_path("gate_symmetrized.csv"))
b0 = load("RVg-IV-12_14_49")
cap = pd.read_csv(tab_path("gate_capacitance.csv")).iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
v = np.linspace(-0.05, 0.55, 200)
ax[0].plot(b0["Vg"], b0["R1"], "o", color="k", ms=6, label="B = 0 data (RVg 12_14_49)")
ax[0].errorbar([0.0], [839.2], yerr=[18], fmt="D", color="C3", ms=7, label="R(B=0, Vg=0), cool-down end")
r = a[a["data"].str.startswith("B=0") & a["model"].str.startswith("R_inf")].iloc[0]
ax[0].plot(v, expo(v, r["p1"], r["p2"], r["p3"]), "-", color="C0", label="exponential fit")
ax[0].fill_between([-0.02, 0.02], r["R_at_Vg0_16"], r["R_at_Vg0_84"], color="C0", alpha=0.3, label="68 % band of extrapolation")
rp = a[a["data"].str.startswith("B=0") & a["model"].str.startswith("Drude")].iloc[0]
ax[0].plot(v, power(v, rp["p1"], rp["p2"], cap["V_th"]), "--", color="C2", label=r"Drude $(V_g-V_{th})^{-p}$")
ax[0].set_ylim(0, 1800); ax[0].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[0].set_ylabel(r"$R_{xx}$ [$\Omega$]")
ax[0].set_title("B = 0: extrapolation from 9 points to Vg = 0"); ax[0].legend(fontsize=8)
q = gs[gs["absB"] == 0.5].groupby("Vg", as_index=False)["Rxx_S"].mean()
ax[1].plot(q["Vg"], q["Rxx_S"], "o", ms=3, color="k", label=r"$R_{xx}^S$, |B| = 0.5 T")
r = a[a["data"].str.startswith("|B|") & a["model"].str.startswith("R_inf")].iloc[0]
vv = np.linspace(-0.5, 0.5, 200)
ax[1].plot(vv, expo(vv, r["p1"], r["p2"], r["p3"]), "-", color="C0", label=fr"exponential, $\chi^2_r$ = {r['chi2_red']:.0f}")
rp = a[a["data"].str.startswith("|B|") & a["model"].str.startswith("Drude")].iloc[0]
ax[1].plot(vv, power(vv, rp["p1"], rp["p2"], cap["V_th"]), "--", color="C2", label=fr"Drude power law, $\chi^2_r$ = {rp['chi2_red']:.0f}")
ax[1].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[1].set_ylabel(r"$R_{xx}^S$ [$\Omega$]")
ax[1].set_title("|B| = 0.5 T: full gate range"); ax[1].legend()
plt.tight_layout(); plt.savefig(fig_path("figA9_exponential_fit.pdf"))
