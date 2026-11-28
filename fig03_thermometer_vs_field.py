"""fig03_thermometer_vs_field.py -- Fig. 3: logged thermometer reading vs |B| (plan section 0.6).
Inputs: tables/thermometer_all_points.csv, thermometer_master_curve.csv, thermometer_fits.csv.
Output: figures/fig03_thermometer_vs_field.pdf"""
import numpy as np
import pandas as pd

from params import FULL_W, T_BATH, fig_path, legend_below, panel_label, plot_style, tab_path

plt = plot_style()
a = pd.read_csv(tab_path("thermometer_all_points.csv"))
mc = pd.read_csv(tab_path("thermometer_master_curve.csv"))
from matplotlib.gridspec import GridSpec

# Two rows. Top: (a) every logged point in one colour -- the runs collapse onto one curve; (b) low-field and
# (c) high-field regime of the |B|-binned master curve; legends below the panels. Bottom, full width: (d) the
# deviation of each point from the master curve in mK, one colour per group of measurements (first sweep,
# continuous sweep at B > 0 and B < 0, ramps between the gate sweeps at B < 0 and B > 0), and the gate-sweep holds
# as their means: during each 160 s hold the reading changed by at most one digit (1 mK), so no error bar is drawn.
# The collapse (all runs within +-8 mK of a ~1000 mK field effect) is the evidence that T_read depends on |B| only.
fig = plt.figure(figsize=(FULL_W, 4.45), layout="constrained")   # figure*: full page width
grid = GridSpec(2, 3, figure=fig, height_ratios=[1.55, 1])
ax0 = fig.add_subplot(grid[0, 0]); axr = fig.add_subplot(grid[1, :])
ax = [ax0, fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]
T_master = lambda b: np.interp(b, mc["absB"], mc["T_mean"])
r = a[a["type"] == "field sweep/ramp"]
h = a[a["type"] == "gate-sweep hold"]
z = a[a["absB"] < 0.005]
ax0.plot(r["absB"], r["T"], "o", color="0.45", ms=1.4, mec="none", label="sweeps and ramps\n(all runs, both signs of B)")
hm = h.groupby("absB")["T"].agg(["mean", "std"]).reset_index()
ax0.plot(hm["absB"], hm["mean"], "s", color="C3", ms=4, mfc="none", label="gate-sweep holds (mean)")
ax0.plot(z["absB"], z["T"], "D", color="C1", ms=3.5, label="B = 0 (cool-down end, 12_14)")
ax0.axhline(T_BATH, color="k", ls="--", lw=0.8)
ax0.text(0.98, T_BATH - 0.02, "LHe bath, 4.2 K", transform=ax0.get_yaxis_transform(), ha="right", va="top", fontsize=7.5)   # labelled on the line
ax0.set_xlabel("|B| [T]"); ax0.set_ylabel(r"$T_{read}$ [K]")
ax0.set_title("Thermometer reading, all runs")
ax0.set_ylim(3.0, 4.32); legend_below(ax0)
res = 1e3 * (r["T"] - T_master(r["absB"]))
cont = r["run"] == "RHLabC-IV-13_44_56"                     # continuous sweep +5.34 -> -5.50 T
GROUPS = (("first sweep, 0.5 to 5.5 T", r["run"] == "RHLabC-IV-12_49_45", "C0"),
          ("continuous sweep, B > 0", cont & (r["B"] >= 0), "C1"),
          ("continuous sweep, B < 0", cont & (r["B"] < 0), "C2"),
          ("ramps between gate sweeps, B < 0", (r["order"] > 3) & (r["B"] < 0), "C4"),
          ("ramps between gate sweeps, B > 0", (r["order"] > 3) & (r["B"] >= 0), "C3"))
for lab, sel, col in GROUPS:
    axr.plot(r["absB"][sel], res[sel], "o", color=col, ms=1.5, mec="none", label=lab)
hr = h.assign(res=1e3 * (h["T"] - T_master(h["absB"]))).groupby("run").agg(absB=("absB", "first"), mean=("res", "mean"),
                                                                           std=("res", "std"))
axr.plot(hr["absB"], hr["mean"], "s", color="k", ms=3.2, mfc="white", mew=0.7, zorder=4,
         label="gate-sweep holds (mean; constant to 1 mK)")
axr.axhline(0, color="k", lw=0.6)
axr.set_ylim(-21, 17); axr.set_xlabel("|B| [T]"); axr.set_ylabel(r"$\Delta T$ [mK]")             # Delta T = T_read - T_master, defined in the caption
axr.set_title("Deviation of each reading from the master curve")
lg = legend_below(axr, ncol=3, columnspacing=1.5)
for hnd in lg.legend_handles[:-1]: hnd.set_markersize(4)   # enlarge the small dots in the legend only

T0 = mc.loc[mc["absB"] < 0.02, "T_mean"].mean()
f = pd.read_csv(tab_path("thermometer_fits.csv"))
par = lambda key: [float(s.split("+-")[0]) for s in f[f["model"].str.startswith(key)]["params"].iloc[0].split(", ")]
A, B0 = par("saturating")
(a2,) = par("quadratic  -a")
lo = mc[mc["absB"] <= 0.6]
ax[1].plot(lo["absB"], lo["T_mean"] - T0, "o", ms=2, color="0.3", label="master curve (20 mT bins)")
x = np.linspace(0, 0.6, 200)
ax[1].plot(x, -A * (1 - np.exp(-(x / B0) ** 2)), "-", color="C0",
           label=r"$-A(1-e^{-B^2/B_0^2})$")
ax[1].plot(x, -a2 * x**2, "--", color="C3", label=r"$-aB^2$")
# fit parameters in the empty upper-right corner (the data have dropped to -0.43 K by 0.35 T)
ax[1].text(0.97, 0.95, "fits for |B| $\\leq$ 0.45 T:\n" + fr"A = {A:.3f} K" + "\n" + fr"$B_0$ = {B0:.3f} T", transform=ax[1].transAxes,
           ha="right", va="top", fontsize=7.5, linespacing=1.25)
ax[1].set_ylim(-0.6, 0.04); ax[1].set_xlabel("|B| [T]"); ax[1].set_ylabel(r"$T_{read}(B) - T_{read}(0)$ [K]")
ax[1].set_title("Low-field regime"); legend_below(ax[1])

hi = mc[mc["absB"] >= 1.2]
ax[2].plot(mc["absB"], mc["T_mean"], "o", ms=1.5, color="0.3", label="master curve")
c = [float(s.split("+-")[0]) for s in f[f["model"].str.startswith("quadratic T0")]["params"].iloc[0].split(", ")]
xb = np.linspace(1.2, 5.5, 100)
ax[2].plot(xb, np.polyval(c, xb), "-", color="C0", label="quadratic fit, 1.2-5.5 T")
cl = [float(s.split("+-")[0]) for s in f[f["model"].str.startswith("linear")]["params"].iloc[0].split(", ")]
ax[2].plot(xb, np.polyval(cl, xb), "--", color="C3", label="linear fit, 1.2-5.5 T")
ax[2].set_xlabel("|B| [T]"); ax[2].set_ylabel(r"$T_{read}$ [K]"); ax[2].set_title("High-field regime"); legend_below(ax[2])
panel_label(ax0, "a"); panel_label(ax[1], "b"); panel_label(ax[2], "c"); panel_label(axr, "d")
plt.savefig(fig_path("fig03_thermometer_vs_field.pdf"))
