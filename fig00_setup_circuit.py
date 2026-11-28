"""fig00_setup_circuit.py -- Fig. 1: block diagram of the measurement (own drawing; the guide asks for a setup diagram).
No data inputs (settings from params.py). Output: figures/fig00_setup_circuit.pdf, one column wide (3.4 in).

Schematic only, not to scale. The Hall bar and its pads follow the contact map of the laboratory: current 5 -> 15
through the 1 MOhm series resistor, longitudinal pair 6-11 (lock-in 1), Hall pair 6-3 (lock-in 2), gate pad 1;
contacts 13, 14 and 16 exist on the chip but were not used (short unnumbered stubs). Every instrument is wired to the
element it measures or drives. The Hall pads 6 and 3 lie on opposite sides of the current path, so one crossing is
unavoidable; it is drawn as a hop (the wires are not connected)."""
import numpy as np
import matplotlib.patches as mp

from params import F_REF_HZ, I_EXC_NOMINAL, R_SERIES, V_SOURCE, fig_path, plot_style

plt = plot_style()
plt.rcParams.update({"figure.constrained_layout.use": False, "axes.grid": False, "savefig.bbox": "standard"})
W, H = 3.4, 3.22
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
LW = 0.8          # boxes
WL = 0.7          # wires
FS = 6.8          # instrument text


def box(x0, y0, x1, y1, text, fc="white", ls="-", fs=FS, z=3):
    ax.add_patch(mp.FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.012,rounding_size=0.035",
                                   fc=fc, ec="k", lw=LW, ls=ls, zorder=z))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15, zorder=4)


def wire(*pts, **kw):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "-", color=kw.get("color", "k"), lw=kw.get("lw", WL), zorder=kw.get("zorder", 2),
            solid_capstyle="round")


def pad(x, y, n):
    ax.add_patch(mp.Rectangle((x - 0.075, y - 0.062), 0.15, 0.124, fc="white", ec="k", lw=0.6, zorder=5))
    ax.text(x, y, n, ha="center", va="center", fontsize=6.8, zorder=6)


def dot(x, y):
    ax.plot([x], [y], "o", color="k", ms=2.2, zorder=6)


# ---- cryostat: sample in the LHe bath, in the bore of the superconducting solenoid (B perpendicular to the sample)
CX0, CY0, CX1, CY1 = 0.80, 0.86, 2.40, 2.86
box(CX0, CY0, CX1, CY1, "", fc="#eef4fb", ls="--", z=0.5)     # behind the wires
ax.text(CX0 + 0.06, CY1 - 0.05, "LHe bath,\n4.2 K", ha="left", va="top", fontsize=6.8, zorder=4, linespacing=1.1)

# Hall bar (vertical) with the local gate
bx, bw, by0, by1 = 1.55, 0.13, 1.24, 2.50
xc = bx + bw / 2
ax.add_patch(mp.Rectangle((bx, by0), bw, by1 - by0, fc="0.55", ec="k", lw=0.6, zorder=4))
gy = 1.92
ax.add_patch(mp.Rectangle((bx - 0.06, gy - 0.12), bw + 0.12, 0.24, fc="#e8c547", ec="k", lw=0.6, zorder=5))
ax.text(xc, gy, "gate", ha="center", va="center", fontsize=6.2, zorder=6)

# pads used: positions after the contact map (15 top, 5 bottom; 6 and 3 at the bottom on opposite sides; 11 on the
# left just below the gate; 1 = gate pad on the right)
P = {"15": (xc, 2.68), "5": (xc, 1.03), "6": (1.13, 1.03), "3": (2.10, 1.03), "11": (1.06, 1.64), "1": (2.20, gy)}
wire(P["15"], (xc, by1), lw=0.6, zorder=3); wire(P["5"], (xc, by0), lw=0.6, zorder=3)
wire(P["6"], (bx, by0 + 0.10), lw=0.6, zorder=3); wire(P["3"], (bx + bw, by0 + 0.10), lw=0.6, zorder=3)
wire(P["11"], (bx, gy - 0.17), lw=0.6, zorder=3)
wire((bx + bw + 0.06, gy), P["1"], color="#b8962a", lw=1.6, zorder=3)
for n, (x, y) in P.items():
    pad(x, y, n)
# contacts on the chip that were not used (13, 14, 16): short unnumbered stubs
for (xa, ya), (xb, yb) in (((bx, 2.40), (bx - 0.18, 2.58)), ((bx + bw, 2.40), (bx + bw + 0.18, 2.58)),
                           ((bx, 2.17), (bx - 0.26, 2.27))):
    wire((xa, ya), (xb, yb), color="0.45", lw=0.6, zorder=3)
ax.text(bx - 0.29, 2.26, "unused", ha="right", va="center", fontsize=5.8, color="0.35", style="italic", zorder=4)
ax.annotate("", (bx + bw + 0.10, 2.38), (bx + bw + 0.10, 2.10), arrowprops=dict(arrowstyle="->", lw=0.7), zorder=4)
ax.text(bx + bw + 0.14, 2.24, "$I$", ha="left", va="center", fontsize=7.5, zorder=4)

# thermometer next to the sample, and the solenoid (inductor symbol)
box(1.94, 1.34, 2.30, 1.50, "therm.", fs=6.0)
t = np.linspace(0, 4 * np.pi, 200)
cx0, cx1, cy = 1.98, 2.30, 2.60
ax.plot(np.linspace(cx0, cx1, 200), cy + 0.045 * np.abs(np.sin(t)), "-", color="k", lw=0.8, zorder=4)
ax.text((cx0 + cx1) / 2, cy - 0.04, "solenoid,\n$B \\perp$ sample", ha="center", va="top", fontsize=5.8, zorder=4)

# ---- instruments
RX0, RX1 = 2.50, 3.22                                       # right-hand column
XR = 3.31                                                   # current return riser, right of the column
box(1.95, 2.94, 3.37, 3.20, f"AC source, {V_SOURCE:.0f} V, {F_REF_HZ:.2f} Hz")
box(RX0, 2.40, RX1, 2.80, "Magnet supply\n(Oxford PS120)")
box(RX0, 1.74, RX1, 2.10, "SourceMeter\n(Keithley 2400)")
box(RX0, 1.24, RX1, 1.60, "Lake Shore 331")
box(0.05, 0.98, 0.64, 1.78, "Lock-in 1\n(SR510)\n\n$V_{xx}$:\npads\n6, 11")
box(0.95, 0.06, 2.30, 0.44, "Lock-in 2 (SR510)\n$V_{xy}$: pads 6, 3")

# lock-in 1: pads 11 and 6 (left)
wire((P["11"][0] - 0.075, P["11"][1]), (0.64, P["11"][1]))
wire((P["6"][0] - 0.075, P["6"][1]), (0.64, P["6"][1]))
# lock-in 2: pads 6 and 3 (down)
wire((P["6"][0], P["6"][1] - 0.062), (P["6"][0], 0.44))
YC = 0.66                                                   # current wire, below the cryostat
hop = 0.045
wire((P["3"][0], P["3"][1] - 0.062), (P["3"][0], YC + hop))
th = np.linspace(0, np.pi, 30)
ax.plot(P["3"][0] + hop * np.sin(th), YC + hop * np.cos(th), "-", color="k", lw=WL, zorder=2)   # hop over the current wire
wire((P["3"][0], YC - hop), (P["3"][0], 0.44))
# current: source -> 1 MOhm -> pad 5 ... pad 15 -> source
wire((P["5"][0], P["5"][1] - 0.062), (P["5"][0], YC), (XR, YC), (XR, 0.84))
zx = XR + 0.045 * np.array([0, 1, -1, 1, -1, 1, -1, 0])     # resistor zigzag on the riser
zy = np.linspace(0.84, 1.10, len(zx))
ax.plot(zx, zy, "-", color="k", lw=WL, zorder=2)
wire((XR, 1.10), (XR, 2.94))
ax.text(XR - 0.07, 0.97, f"{R_SERIES/1e6:.0f} M$\\Omega$", ha="right", va="center", fontsize=6.5, zorder=4)
ax.text(2.72, YC + 0.03, f"$I$ = {I_EXC_NOMINAL*1e6:.0f} $\\mu$A", ha="center", va="bottom", fontsize=6.5, zorder=4)
wire((P["15"][0], P["15"][1] + 0.062), (P["15"][0], 3.07), (1.95, 3.07))
# gate, thermometer, magnet
wire((P["1"][0] + 0.075, gy), (RX0, gy))
wire((2.30, 1.42), (RX0, 1.42))
wire((cx1, cy), (RX0, cy))
plt.savefig(fig_path("fig00_setup_circuit.pdf"))
