"""fig10_dingle.py -- Fig. 10: Dingle plot and T-degeneracy profile (plan section 6).
Inputs: tables/dingle_points.csv, dingle_results.csv, dingle_T_profile.csv. Output: figures/fig10_dingle.pdf"""
import numpy as np
import pandas as pd

from dingle import X
from params import T_BATH, fig_path, plot_style, tab_path

plt = plot_style()
p = pd.read_csv(tab_path("dingle_points.csv"))
r = pd.read_csv(tab_path("dingle_results.csv"))
pr = pd.read_csv(tab_path("dingle_T_profile.csv"))
R0 = r["R0_ohm"].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for lab, T, mk, col in (("T = 4.2 K", T_BATH, "o", "k"), ("T = logged T(B)", p["T_logged"], "s", "C3")):
    y = np.log(p["dR"] / R0 * np.sinh(X(T, p["B"])) / (4 * X(T, p["B"])))
    u = p["used"]
    ax[0].plot(p.loc[u, "invB"], y[u], mk, color=col, ms=6, label=f"{lab} (used)")
    ax[0].plot(p.loc[~u, "invB"], y[~u], mk, color=col, ms=6, mfc="none", label=fr"{lab} ($\Delta R/R_0$ > 0.5, not used)")
    rr = r[r["fit"].str.contains("logged" if "logged" in lab else "primary")].iloc[0]
    x = np.linspace(0, 0.85, 10)
    slope = -np.pi * 0.067 * 9.1093837e-31 / (1.602176634e-19 * rr["tau_q_ps"] * 1e-12)
    ax[0].plot(x, rr["intercept"] + slope * x, "-" if col == "k" else "--", color=col,
               label=fr"fit: $\tau_q$ = {rr['tau_q_ps']:.2f} $\pm$ {rr['s_tau_q_ps']:.2f} ps")
ax[0].set_xlabel(r"$1/B$ [T$^{-1}$]"); ax[0].set_ylabel(r"$\ln[\Delta R \sinh X / (4 R_0 X)]$")
ax[0].set_title(r"Dingle plot, pads 6-11, $V_g$ = +0.5 V"); ax[0].legend(fontsize=8)
ok = pr["tau_q_ps"] > 0
ax[1].plot(pr.loc[ok, "T_K"], pr.loc[ok, "tau_q_ps"], "o-", color="k", label=r"$\tau_q$ at fixed trial T")
ax2 = ax[1].twinx()
ax2.plot(pr["T_K"], pr["rss"], "s--", color="C3", label="residual sum of squares")
ax2.set_ylabel("RSS of Dingle fit", color="C3"); ax2.set_ylim(0, 1.5 * pr["rss"].max())
ax[1].set_xlabel("Assumed electron temperature T [K]"); ax[1].set_ylabel(r"$\tau_q$ [ps]")
ax[1].set_title(r"Degeneracy of $\tau_q$ and T (flat RSS)")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax[1].legend(h1 + h2, l1 + l2, loc="upper left")
plt.tight_layout(); plt.savefig(fig_path("fig10_dingle.pdf"))
