"""fig10_dingle.py -- Fig. 10: Dingle plot and T-degeneracy profile (plan section 6).
Inputs: tables/dingle_points.csv, dingle_results.csv, dingle_T_profile.csv. Output: figures/fig10_dingle.pdf"""
import numpy as np
import pandas as pd

from dingle import X
from matplotlib.legend_handler import HandlerTuple

from params import FULL_W, T_BATH, fig_path, legend_below, panel_label, plot_style, tab_path

plt = plot_style()
p = pd.read_csv(tab_path("dingle_points.csv"))
r = pd.read_csv(tab_path("dingle_results.csv"))
pr = pd.read_csv(tab_path("dingle_T_profile.csv"))
R0 = r["R0_ohm"].iloc[0]
fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 3.05))
LEG = []          # figure*: full page width
for lab, T, mk, col in (("T = 4.2 K", T_BATH, "o", "k"), ("T = logged T(B)", p["T_logged"], "s", "C3")):
    y = np.log(p["dR"] / R0 * np.sinh(X(T, p["B"])) / (4 * X(T, p["B"])))
    u = p["used"]
    hp, = ax[0].plot(p.loc[u, "invB"], y[u], mk, color=col, ms=4)
    ax[0].plot(p.loc[~u, "invB"], y[~u], mk, color=col, ms=4, mfc="none")       # open symbols: one legend entry below
    rr = r[r["fit"].str.contains("logged" if "logged" in lab else "primary")].iloc[0]
    x = np.linspace(0, 0.85, 10)
    slope = -np.pi * 0.067 * 9.1093837e-31 / (1.602176634e-19 * rr["tau_q_ps"] * 1e-12)
    hl, = ax[0].plot(x, rr["intercept"] + slope * x, "-" if col == "k" else "--", color=col)
    LEG.append(((hp, hl), fr"{lab}: fit $\tau_q$ = {rr['tau_q_ps']:.2f} $\pm$ {rr['s_tau_q_ps']:.2f} ps"))   # symbol + its fit line
ax[0].set_xlabel(r"$1/B$ [T$^{-1}$]"); ax[0].set_ylabel(r"$\ln[\Delta R \sinh X / (4 R_0 X)]$")
ax[0].set_title(r"Dingle plot, pads 6-11, $V_g$ = +0.5 V")
ho, = ax[0].plot([], [], "o", color="0.35", ms=4, mfc="none")
LEG.append((ho, r"open symbols: $\Delta R/R_0$ > 0.5, not used in the fits"))
legend_below(ax[0], [l[0] for l in LEG], [l[1] for l in LEG], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
             handlelength=2.4)                      # legends below the panels: nothing to cover
ok = pr["tau_q_ps"] > 0
ax[1].plot(pr.loc[ok, "T_K"], pr.loc[ok, "tau_q_ps"], "o-", color="k", ms=3, label=r"$\tau_q$ at fixed trial T")
ax2 = ax[1].twinx()
ax2.plot(pr["T_K"], pr["rss"], "s--", color="C3", ms=3, label="residual sum of squares")
ax2.set_ylabel("RSS of Dingle fit", color="C3"); ax2.set_ylim(0, 1.5 * pr["rss"].max())
ax[1].set_xlabel("Assumed electron temperature T [K]"); ax[1].set_ylabel(r"$\tau_q$ [ps]")
ax[1].set_title(r"Degeneracy of $\tau_q$ and T (flat RSS)")
h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); legend_below(ax[1], h1 + h2, l1 + l2, ncol=2)
for a_, l_ in zip(ax, "ab"): panel_label(a_, l_)
plt.savefig(fig_path("fig10_dingle.pdf"))
