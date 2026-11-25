"""fig02_symmetrisation.py -- Fig. 2: raw vs Onsager-symmetrised signals and the admixture alpha (plan 0.1, 0.4).
Inputs: tables/field_symmetrized_vg+0.5.csv, gate_symmetrized.csv. Output: figures/fig02_symmetrisation.pdf"""
import pandas as pd

from params import MARKERS, fig_path, plot_style, tab_path

plt = plot_style()
fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
gs = pd.read_csv(tab_path("gate_symmetrized.csv"))
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
B = fs["absB"]
ax[0, 0].plot(B, fs["R1_pos_down"], "-", color="C3", label=r"raw $R_1(+B)$")
ax[0, 0].plot(B, fs["R1_neg_down"], "--", color="C0", label=r"raw $R_1(-B)$")
ax[0, 0].plot(B, fs["Rxx_S_down"], "-", color="k", lw=1.5, label=r"$R_{xx}^S = [R_1(B)+R_1(-B)]/2$")
ax[0, 0].set_xlabel("|B| [T]"); ax[0, 0].set_ylabel(r"$R$ [$\Omega$]"); ax[0, 0].set_title(r"Pads 6-11, $V_g$ = +0.5 V, single down-sweep")
ax[0, 0].legend()
ax[0, 1].plot(B, fs["R2_pos_down"], "-", color="C3", label=r"raw $R_2(+B)$")
ax[0, 1].plot(B, -fs["R2_neg_down"], "--", color="C0", label=r"raw $-R_2(-B)$")
ax[0, 1].plot(B, fs["Rxy_A_down"], "-", color="k", lw=1.2, label=r"$R_{xy}^A = [R_2(B)-R_2(-B)]/2$")
ax[0, 1].plot(B, fs["R2_S_down"] * 20, ":", color="C2", label=r"even part $R_2^S$ ($\times 20$)")
ax[0, 1].set_xlabel("|B| [T]"); ax[0, 1].set_ylabel(r"$R$ [$\Omega$]"); ax[0, 1].set_title(r"Pads 6-3 (Hall), $V_g$ = +0.5 V, single down-sweep")
ax[0, 1].legend()
ax[1, 0].plot(B, fs["alpha_down"], "-", color="C0", label="single down-sweep (13_44)")
ax[1, 0].plot(B, fs["alpha_up"], "--", color="C3", label="up-ramps (after gate sweeps)")
ax[1, 0].set_ylim(0.45, 0.7); ax[1, 0].set_xlabel("|B| [T]"); ax[1, 0].set_ylabel(r"$\alpha = R_1^A / R_{xy}^A$")
ax[1, 0].set_title(r"Odd admixture in pads 6-11 vs field ($V_g$ = +0.5 V)"); ax[1, 0].legend()
for i, b in enumerate(sorted(gs["absB"].unique())):
    q = gs[(gs["absB"] == b) & (gs["dir"] == "down")].sort_values("Vg")
    ax[1, 1].plot(q["Vg"], q["alpha"], MARKERS[i] + "-", ms=3, lw=0.6, label=f"|B| = {b} T")
ax[1, 1].set_ylim(0.45, 0.7); ax[1, 1].set_xlabel(r"Gate voltage $V_g$ [V]"); ax[1, 1].set_ylabel(r"$\alpha$")
ax[1, 1].set_title("Odd admixture vs gate voltage (down sweeps)"); ax[1, 1].legend(ncol=2)
plt.tight_layout(); plt.savefig(fig_path("fig02_symmetrisation.pdf"))
