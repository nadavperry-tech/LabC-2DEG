"""fig02_symmetrisation.py -- Fig. 2: raw vs Onsager-symmetrised signals and the admixture alpha (plan 0.1, 0.4).
Inputs: tables/field_symmetrized_vg+0.5.csv, gate_symmetrized.csv. Output: figures/fig02_symmetrisation.pdf

Display choices (state them in the caption):
  * alpha is shown bin-averaged (|B| bins of ALPHA_BIN_T, Vg bins of ALPHA_BIN_V), computed as the ratio of the
    bin means <R_1^A>/<R_xy^A>. Binning keeps every point in the average instead of dropping points, and
    suppresses the noise of the point-by-point ratio. The numbers in the text (alpha = 0.570) use all points.
  * The even part of the Hall pair R_2^S is drawn in its own sub-panel in ohm (same unit, own scale) instead of
    being multiplied by 20 or given a second y-axis; the raw +B and -B traces are drawn as sparse markers so
    they can be seen lying on R_xy^A."""
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from params import FULL_W, MARKERS, fig_path, panel_label, plot_style, tab_path

ALPHA_BIN_T = 0.10      # T
ALPHA_BIN_V = 0.05      # V
R2S_BIN_T = 0.05        # T
ALPHA_YLIM = (0.535, 0.625)


def binned(x, num, den, edges):
    """Bin centres and <num>/<den> per bin (bins with no finite points are dropped)."""
    df = pd.DataFrame({"x": x, "num": num, "den": den}).dropna()
    df["bin"] = pd.cut(df["x"], edges)
    g = df.groupby("bin", observed=True)
    out = pd.DataFrame({"x": g["x"].mean(), "num": g["num"].mean(), "den": g["den"].mean()})
    return out["x"].values, (out["num"] / out["den"]).values


plt = plot_style()
fs = pd.read_csv(tab_path("field_symmetrized_vg+0.5.csv"))
gs = pd.read_csv(tab_path("gate_symmetrized.csv"))
fig = plt.figure(figsize=(FULL_W, 5.0))                        # figure*: full page width
outer = GridSpec(2, 2, figure=fig)
a00 = fig.add_subplot(outer[0, 0])
inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], height_ratios=[3, 1], hspace=0.08)
a01 = fig.add_subplot(inner[0]); a01s = fig.add_subplot(inner[1], sharex=a01)
a10 = fig.add_subplot(outer[1, 0]); a11 = fig.add_subplot(outer[1, 1])
B = fs["absB"]

# (a) pads 6-11: raw +B / -B and the symmetric part
a00.plot(B, fs["R1_pos_down"], "-", color="C3", label=r"raw $R_1(+B)$")
a00.plot(B, fs["R1_neg_down"], "--", color="C0", label=r"raw $R_1(-B)$")
a00.plot(B, fs["Rxx_S_down"], "-", color="k", lw=1.2, label=r"$R_{xx}^S = [R_1(B)+R_1(-B)]/2$")
a00.set_xlabel("|B| [T]"); a00.set_ylabel(r"$R$ [$\Omega$]"); a00.set_title(r"Pads 6-11, $V_g$ = +0.5 V, single down-sweep")
a00.legend()

# (b) pads 6-3: raw traces as sparse markers on top of R_xy^A; even part in its own sub-panel
a01.plot(B, fs["Rxy_A_down"], "-", color="k", lw=1.0, label=r"$R_{xy}^A = [R_2(B)-R_2(-B)]/2$", zorder=2)
a01.plot(B, fs["R2_pos_down"], "o", color="C3", ms=3.5, mfc="none", markevery=(0, 40), label=r"raw $R_2(+B)$", zorder=3)
a01.plot(B, -fs["R2_neg_down"], "s", color="C0", ms=3.5, mfc="none", markevery=(20, 40), label=r"raw $-R_2(-B)$", zorder=3)
a01.set_ylabel(r"$R$ [$\Omega$]"); a01.set_title(r"Pads 6-3 (Hall), $V_g$ = +0.5 V, single down-sweep")
a01.legend(loc="lower right", fontsize=7.5, handlelength=1.3, labelspacing=0.25, borderaxespad=0.3)   # the empty triangle below the line
a01.set_ylim(bottom=-1000); a01.tick_params(labelbottom=False)
edges = np.arange(0, B.max() + R2S_BIN_T, R2S_BIN_T)
xb, yb = binned(B, fs["R2_S_down"], np.ones(len(B)), edges)
a01s.axhline(0, color="0.5", lw=0.6)
a01s.plot(xb, yb, "-", color="C2", lw=1.0)
a01s.text(0.99, 0.93, fr"even part $R_2^S$ ({R2S_BIN_T*1e3:.0f} mT bins)", transform=a01s.transAxes, ha="right", va="top",
          color="C2", fontsize=8)          # labelled in place: a legend box would cover the trace in this short panel
a01s.set_ylim(-14, 26); a01s.set_xlabel("|B| [T]"); a01s.set_ylabel(r"$R_2^S$ [$\Omega$]")

# (c) alpha vs field, bin-averaged
edges = np.arange(0, B.max() + ALPHA_BIN_T, ALPHA_BIN_T)
for col, mk, ls, lab in (("down", "o", "-", "single down-sweep (13_44)"), ("up", "s", "--", "up-ramps (after gate sweeps)")):
    xb, ab = binned(B, fs[f"R1_A_{col}"], fs[f"Rxy_A_{col}"], edges)
    keep = xb > 0.2                                   # below 0.2 T R_xy^A is too small for a meaningful ratio
    a10.plot(xb[keep], ab[keep], mk + ls, color="C0" if col == "down" else "C3", ms=3, lw=0.7, label=lab)
a10.set_ylim(*ALPHA_YLIM); a10.set_xlabel("|B| [T]"); a10.set_ylabel(r"$\alpha = R_1^A / R_{xy}^A$")
a10.set_title(fr"Odd admixture in pads 6-11 vs field ($V_g$ = +0.5 V)" + "\n" + fr"{ALPHA_BIN_T*1e3:.0f} mT bins")
a10.legend(loc="upper left")

# (d) alpha vs gate voltage, bin-averaged
vedges = np.arange(-0.5, 0.5 + ALPHA_BIN_V / 2, ALPHA_BIN_V)
for i, b in enumerate(sorted(gs["absB"].unique())):
    q = gs[(gs["absB"] == b) & (gs["dir"] == "down")].sort_values("Vg")
    xb, ab = binned(q["Vg"], q["R1_A"], q["Rxy_A"], vedges)
    a11.plot(xb, ab, MARKERS[i] + "-", color=f"C{i}", ms=3, lw=0.7, label=f"|B| = {b} T")
a11.set_ylim(*ALPHA_YLIM); a11.set_xlabel(r"Gate voltage $V_g$ [V]"); a11.set_ylabel(r"$\alpha$")
a11.set_title(fr"Odd admixture vs gate voltage (down sweeps)" + "\n" + fr"{ALPHA_BIN_V*1e3:.0f} mV bins")
a11.legend(loc="upper center", ncol=3)
for a_, l_ in zip((a00, a01, a10, a11), "abcd"): panel_label(a_, l_)
plt.savefig(fig_path("fig02_symmetrisation.pdf"))
