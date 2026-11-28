"""
params.py -- constants, paths and fixed analysis parameters for the 2DEG (Lab C) analysis.

Every number that is *assumed* rather than measured is listed here with its source,
so that the paper and the code use the same values.  Quantities that are produced by
the analysis (k, n_SdH, tau_q, ...) are NOT stored here; they are written to tables/*.csv
by the analysis scripts and read back by later scripts.
"""
import os
import scipy.constants as sc

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
TAB_DIR = os.path.join(ROOT, "tables")
for _d in (FIG_DIR, TAB_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- physical constants (CODATA via scipy)
e = sc.e
h = sc.h
hbar = sc.hbar
kB = sc.k
m_e = sc.m_e
eps0 = sc.epsilon_0
mu_B = sc.physical_constants["Bohr magneton"][0]
R_K = h / e**2                      # von Klitzing constant, 25812.807 Ohm

# ---------------------------------------------------------------- material parameters (GaAs/AlGaAs)
M_EFF = 0.067                       # m*/m_e, GaAs conduction band (standard value)
M_STAR = M_EFF * m_e
EPS_R = 12.9                        # GaAs static dielectric constant; Al0.3Ga0.7As ~12.2 -> used as systematic
EPS_R_ALT = 12.2
HBAR_OMEGA_LO_meV = 36.2            # GaAs LO-phonon energy (literature value; also left free in rt_model as a check)
V_SOUND_LA = 5.2e3                  # m/s, GaAs LA sound velocity (for T_BG estimate only)
G_BARE = -0.44                      # bulk GaAs electron g-factor (for comparison only)
GS, GV = 2, 1                       # spin and valley degeneracy

# ---------------------------------------------------------------- experimental settings (lab notebook / plan section 0.3)
T_BATH = 4.2                        # K. LHe at ambient pressure, bath not pumped (plan section 0.6)
R_RESOLUTION = 5.0                  # Ohm, digitisation step of the logged lock-in readings
V_SOURCE = 1.0                      # V, AC source amplitude (TA, 25 Nov)
R_SERIES = 1e6                      # Ohm, series resistor (TA, 25 Nov: 1 MOhm, not 1 kOhm)
I_EXC_NOMINAL = V_SOURCE / R_SERIES # A = 1.0 uA (confirmed by TA). The acquisition program was given R_SERIES and the
                                    # source value and logs R = V_lock-in / I directly, so the data columns are in Ohm
                                    # with I = 1 uA already applied; no further conversion is done in this code.
                                    # Consequence: the Hall scale factor k (hall.py) is NOT an excitation-current error.
LOCKIN_PHASE_DEG = 24.0             # reference phase shown on one SR510 (set by the TA; the other was not recorded)
LOCKIN_DELAY_S = 2.0                # mean delay of pre 1 s + post 1 s filters (plan 0.3)
LOCKIN_SETTLE_99_S = 6.6            # time to 99 % for the same filter chain
F_REF_HZ = 11.45

# ---------------------------------------------------------------- geometry
# N_sq = L/W between the longitudinal voltage pads 6-11: L/W = 2 (TA, 25 Nov; no uncertainty given).
# Sample layout (Nadav, 25 Nov; lab instructions Fig. 1): the gate (pad 1) covers only part of the bar;
# pad 11 sits across the bar from the gate contact, pads 6 and 3 (Hall pair) sit next to them.
# Distance gate -> 2DEG: not known (TA) -> only the measured d_eff = eps0 eps_r / C is reported.
N_SQ = 2.0
GATE_DEPTH_NOMINAL = None

# Thermometer: Lake Shore 331 controller (Nadav, 25 Nov); sensor per lab instructions: Cernox.
# Our field error (-25 % at 5.45 T, 4.2 K) is ~100x the typical Cernox CX-1050 error (+0.1 % at 2.5 T,
# -0.15 % at 8 T) and close to the typical Si-diode error with junction perpendicular to B
# (-8 % at 1 T ... -20 % at 5 T) -- Lake Shore sensor data, used here for orientation only (not citable).

# Phonon-limited mobility, now used only as a CONSISTENCY CHECK of N_sq (summary.py, geometry_estimates.csv):
# own estimate N_sq = mu_ph e n [R(77 K) - R0] (Matthiessen applied to R_2D(77 K) - R_0) vs the TA value.
#   * bulk GaAs lattice-limited mobility at 77 K: 2.4e5 cm^2/Vs
#     (Stillman, Wolfe & Dimmock, J. Appl. Phys. 41, 3088 (1970));
#   * calculations for the 2DEG give ~25 % lower values at n ~ 7e11 cm^-2
#     (Phonon-limited mobility in GaAlAs/GaAs heterostructures, Appl. Phys. Lett. 45, 581 (1984));
#   * measured 2DEG maximum at 77 K ~2e5 cm^2/Vs, limited by optical phonons
#     (Mendez, Price & Heiblum, Appl. Phys. Lett. 45, 294 (1984)).
# Central value 2.0e5 cm^2/Vs with a +-0.6e5 systematic covers this spread.
# (Hirakawa & Sakaki, Phys. Rev. B 33, 8291 (1986), named in the plan, not read here: paywalled.)
MU_PH = {77.0: (20.0, 6.0)}         # T [K] : (mu_ph [m^2/Vs], 1-sigma systematic)

# ---------------------------------------------------------------- analysis choices (see ledger/rulings.md)
FIELD_GRID_STEP = 0.005             # T, common |B| grid for symmetrisation
SDH_BMIN = 1.0                      # T, lower edge for SdH analysis (oscillations resolvable from ~1.2 T)
SDH_BG_DEG = 3                      # polynomial degree of the non-oscillatory background in B
VG_STEP = 0.010                     # V, gate step of the stepped sweeps

# ---------------------------------------------------------------- plotting style (writing guide 4.1)
# Figures are drawn at their final printed width, so the font sizes below are the sizes in the paper.
# REVTeX 4.2 reprint (two columns): \columnwidth = 246 pt = 3.40 in, \textwidth = 510 pt = 7.06 in.
# Include column figures with width=\columnwidth and figure* with width=\textwidth (scale 1.00-1.01).
COL_W = 3.4                         # in, one column (figure)
FULL_W = 7.0                        # in, full page width (figure*)


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # text: 8-9 pt in print
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        # layout: constrained layout fills figsize; the tight bbox only trims the outer white margin
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.02, "figure.constrained_layout.wspace": 0.02,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015, "figure.dpi": 150,
        # line and marker weights for a 3.4 in / 7.0 in figure
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "lines.markeredgewidth": 0.7,
        "axes.linewidth": 0.6, "axes.titlepad": 3.5, "axes.labelpad": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "legend.borderpad": 0.3, "legend.labelspacing": 0.25, "legend.handlelength": 1.8,
        "legend.handletextpad": 0.4, "legend.columnspacing": 0.9, "legend.borderaxespad": 0.3,
        "legend.framealpha": 0.85,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "errorbar.capsize": 2,
        "pdf.fonttype": 42,          # embed TrueType fonts (text stays selectable/searchable in the PDF)
    })
    return plt

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]


def panel_label(ax, letter, where=None):
    """Panel letter for the paper's (a), (b), ... references: prefixed to the panel title, or placed inside the
    axes (where = "upper left" / "upper right") for panels without a title."""
    if where is None and ax.get_title():
        ax.set_title(f"({letter}) " + ax.get_title())
    else:
        right = where == "upper right"
        ax.text(0.98 if right else 0.02, 0.95, f"({letter})", transform=ax.transAxes, ha="right" if right else "left",
                va="top", fontsize=8.5)


# Legends outside the axes: a soft frame (thin light-grey rounded border) keeps them visually apart from the x label.
LEGEND_BOX = dict(frameon=True, fancybox=True, edgecolor="0.72", facecolor="white", framealpha=1.0)


def soft_frame(leg):
    """Thin light-grey border for a legend that sits outside the axes."""
    leg.get_frame().set_linewidth(0.5)
    return leg


def legend_below(ax, *args, gap=None, **kw):
    """Legend under the axes, outside the data: centred, in a soft frame. gap = distance in inches from the bottom of
    the axes to the top of the legend; the default clears the tick labels and the x label with some extra space (or
    leaves a small margin when the axes has no tick labels, e.g. the upper panel of a shared-x pair). Constrained
    layout makes room for it."""
    import matplotlib.transforms as mt
    if gap is None:
        has_ticklabels = ax.xaxis.get_tick_params().get("labelbottom", True)
        gap = (0.43 if ax.get_xlabel() else 0.26) if has_ticklabels else 0.07
    tr = mt.offset_copy(ax.transAxes, fig=ax.get_figure(root=True), y=-gap, units="inches")
    for k, v in LEGEND_BOX.items():
        kw.setdefault(k, v)
    kw.setdefault("loc", "upper center")
    kw.setdefault("borderaxespad", 0.0)
    kw.setdefault("labelspacing", 0.3)
    return soft_frame(ax.legend(*args, bbox_to_anchor=(0.5, 0.0), bbox_transform=tr, **kw))


def fig_path(name):
    return os.path.join(FIG_DIR, name)


def tab_path(name):
    return os.path.join(TAB_DIR, name)
