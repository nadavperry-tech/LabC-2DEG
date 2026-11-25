"""fig03_thermometer_vs_field.py -- Fig. 3: logged thermometer reading vs |B| (plan section 0.6).
Inputs: tables/thermometer_all_points.csv, thermometer_master_curve.csv, thermometer_fits.csv.
Output: figures/fig03_thermometer_vs_field.pdf"""
import numpy as np
import pandas as pd

from params import T_BATH, fig_path, plot_style, tab_path

plt = plot_style()
a = pd.read_csv(tab_path("thermometer_all_points.csv"))
mc = pd.read_csv(tab_path("thermometer_master_curve.csv"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
r = a[a["type"] == "field sweep/ramp"]
sc = ax[0].scatter(r["absB"], r["T"], c=r["order"], s=6, cmap="viridis", marker="o")
neg = r[r["B"] < 0]
ax[0].scatter(neg["absB"], neg["T"], s=10, facecolors="none", edgecolors="k", lw=0.3, label="B < 0 (open)")
h = a[a["type"] == "gate-sweep hold"]
ax[0].plot(h["absB"], h["T"], "s", color="C3", ms=6, mfc="none", label="gate-sweep holds")
z = a[a["absB"] < 0.005]
ax[0].plot(z["absB"], z["T"], "D", color="C1", ms=5, label="B = 0 (cool-down end, 12_14)")
ax[0].axhline(T_BATH, color="k", ls="--", lw=0.8, label="LHe bath 4.2 K")
plt.colorbar(sc, ax=ax[0], label="run order (file-name time)")
ax[0].set_xlabel("|B| [T]"); ax[0].set_ylabel(r"Logged temperature $T_{read}$ [K]")
ax[0].set_title("Thermometer reading vs field, all runs"); ax[0].legend(loc="lower left")

T0 = mc.loc[mc["absB"] < 0.02, "T_mean"].mean()
f = pd.read_csv(tab_path("thermometer_fits.csv"))
par = lambda key: [float(s.split("+-")[0]) for s in f[f["model"].str.startswith(key)]["params"].iloc[0].split(", ")]
A, B0 = par("saturating")
(a2,) = par("quadratic  -a")
lo = mc[mc["absB"] <= 0.6]
ax[1].plot(lo["absB"], lo["T_mean"] - T0, "o", ms=3, color="0.3", label="master curve (20 mT bins)")
x = np.linspace(0, 0.6, 200)
ax[1].plot(x, -A * (1 - np.exp(-(x / B0) ** 2)), "-", color="C0",
           label=fr"$-A(1-e^{{-B^2/B_0^2}})$: A = {A:.3f} K, $B_0$ = {B0:.3f} T")
ax[1].plot(x, -a2 * x**2, "--", color="C3", label=r"$-aB^2$ (both fitted for |B| $\leq$ 0.45 T)")
ax[1].set_ylim(-0.6, 0.05); ax[1].set_xlabel("|B| [T]"); ax[1].set_ylabel(r"$T_{read}(B) - T_{read}(0)$ [K]")
ax[1].set_title("Low-field regime"); ax[1].legend()

hi = mc[mc["absB"] >= 1.2]
ax[2].plot(mc["absB"], mc["T_mean"], "o", ms=2, color="0.3", label="master curve")
c = [float(s.split("+-")[0]) for s in f[f["model"].str.startswith("quadratic T0")]["params"].iloc[0].split(", ")]
xb = np.linspace(1.2, 5.5, 100)
ax[2].plot(xb, np.polyval(c, xb), "-", color="C0", label=r"quadratic fit 1.2-5.5 T")
cl = [float(s.split("+-")[0]) for s in f[f["model"].str.startswith("linear")]["params"].iloc[0].split(", ")]
ax[2].plot(xb, np.polyval(cl, xb), "--", color="C3", label="linear fit 1.2-5.5 T")
ax[2].set_xlabel("|B| [T]"); ax[2].set_ylabel(r"$T_{read}$ [K]"); ax[2].set_title("High-field regime"); ax[2].legend()
plt.tight_layout(); plt.savefig(fig_path("fig03_thermometer_vs_field.pdf"))
