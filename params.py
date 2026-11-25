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
I_EXC_NOMINAL = 1e-6                # A, AC source 1 V through 1 MOhm (inferred; TA asked to confirm)
LOCKIN_PHASE_DEG = 24.0             # reference phase set on both SR510
LOCKIN_DELAY_S = 2.0                # mean delay of pre 1 s + post 1 s filters (plan 0.3)
LOCKIN_SETTLE_99_S = 6.6            # time to 99 % for the same filter chain
F_REF_HZ = 11.45

# ---------------------------------------------------------------- geometry
# N_sq = L/W between the voltage pads 6-11 is NOT known (TA asked).  Geometry-dependent
# quantities are reported per unit N_sq and, for orientation only, at the own-data estimate
# written by summary.py to tables/geometry_estimates.csv.
N_SQ_KNOWN = None

# Phonon-limited mobility used for the N_sq calibration of plan section 0.5 (method 2), applied
# through Matthiessen's rule to the *phonon part* of the 2DEG resistance, R_2D(77 K) - R_0.
#   * bulk GaAs lattice-limited mobility at 77 K: 2.4e5 cm^2/Vs
#     (Stillman, Wolfe & Dimmock, J. Appl. Phys. 41, 3088 (1970));
#   * calculations for the 2DEG give ~25 % lower values at n ~ 7e11 cm^-2
#     (Phonon-limited mobility in GaAlAs/GaAs heterostructures, Appl. Phys. Lett. 45, 581 (1984));
#   * the cleanest 2DEGs reach ~4e5 cm^2/Vs at 77 K (upper bound).
# Central value 2.0e5 cm^2/Vs with a +-0.6e5 systematic covers this spread.
# (Hirakawa & Sakaki 1986, named in the plan, could not be checked here -> verify if cited.)
MU_PH = {77.0: (20.0, 6.0)}         # T [K] : (mu_ph [m^2/Vs], 1-sigma systematic)

# ---------------------------------------------------------------- analysis choices (see ledger/rulings.md)
FIELD_GRID_STEP = 0.005             # T, common |B| grid for symmetrisation
SDH_BMIN = 1.0                      # T, lower edge for SdH analysis (oscillations resolvable from ~1.2 T)
SDH_BG_DEG = 3                      # polynomial degree of the non-oscillatory background in B
VG_STEP = 0.010                     # V, gate step of the stepped sweeps

# ---------------------------------------------------------------- plotting style (writing guide 4.1)
def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "figure.dpi": 100, "savefig.bbox": "tight", "lines.markersize": 4,
        "axes.grid": True, "grid.alpha": 0.3,
    })
    return plt

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]


def fig_path(name):
    return os.path.join(FIG_DIR, name)


def tab_path(name):
    return os.path.join(TAB_DIR, name)
