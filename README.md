# Quantum Hall Effect in a Partially Gated GaAs/AlGaAs Two-Dimensional Electron Gas

## Abstract

This repository contains the raw data and the complete Python analysis for the B.Sc. lab "Transport in a 2D electron gas" (Lab C).
We measured a gated GaAs/AlGaAs Hall bar with two SR510 lock-in amplifiers: a cool-down R(T) from 300 K to 4.2 K, magnetic-field sweeps up to ±5.5 T, and stepped gate sweeps (−0.5 to +0.5 V) at six fields of both signs.
The scripts symmetrise the signals in ±B, extract the Hall and Shubnikov–de Haas densities, calibrate the Hall channel with the quantum Hall plateaus, determine the gate capacitance, the quantum lifetime (Dingle plot) and the parallel-channel activation energy, and analyse the field dependence of the thermometer reading.
Running `python run_all.py` regenerates every table (`tables/*.csv`) and every figure of the paper (`figures/*.pdf`) and finishes with numerical self-checks.

## How to reproduce

```
pip install numpy scipy pandas matplotlib
python run_all.py        # ~25 s; writes tables/ and figures/, then runs checks.py
```
Python ≥ 3.10. Each script can also be run on its own in the order listed below (later scripts read the tables written by earlier ones).

## Data (`data/`)

40 tab-separated LabVIEW files, named `<type>-IV-<hh_mm_ss>-11_11_2026.txt` (time = start of the run). Columns: Time [s] (valid only inside a file), Temperature [K], Field 1 [T], Field 2/3 (unused), Gate Voltage [V], SR510 No.1 [Ω] (pads 6–11), SR510 No.2 [Ω] (pads 6–3, Hall), two empty columns, gate leakage current [A].

| Type | Files | Content |
|---|---|---|
| `Rt` | 1 | cool-down R(T), B = 0, Vg = 0 |
| `RVg` | 25 | stepped gate sweeps (10 mV) at B = ±0.5 … ±5.5 T, down and up; one short B = 0 sweep (12_14_49) |
| `RHLabC` | 14 | field sweeps/ramps at Vg = +0.5 V |

Known file issues (handled in `load_data.py`, listed in `tables/inventory.csv`): `RHLabC-IV-12_46_02` is an exact prefix of `12_49_45` (dropped); the first 101 rows of `RHLabC-IV-14_37_02` are a copy of `RVg-IV-14_30_57` (trimmed).

## Files

### Shared modules

| File | Description | Inputs | Outputs |
|---|---|---|---|
| `params.py` | Physical constants, material parameters (m* = 0.067 mₑ, ε_r), experimental settings (lock-in delay, phase, resolution), literature mobility used for the geometry estimate, analysis choices, plot style. Every assumed number is listed here with its source. | – | – |
| `load_data.py` | Reads a raw file into a DataFrame, applies the duplicate-row fixes, run ordering by file-name time, sweep-direction helper, Hampel spike filter. | `data/*.txt` | – |

### Analysis scripts (run order)

| File | Plan § | Description | Inputs | Outputs |
|---|---|---|---|---|
| `inventory.py` | 0.2 | One row per file: rows kept, B/Vg/T ranges, sweep direction, maximum leakage, role, notes. | `data/` | `tables/inventory.csv` |
| `symmetrize.py` | 0.1, 0.4 | Onsager symmetrisation R_xx^S = [R₁(B)+R₁(−B)]/2, R_xy^A = [R₂(B)−R₂(−B)]/2 and the odd admixture α = R₁^A/R_xy^A. ±B are paired only within sweeps with the same sign of dB/dt (lock-in lag cancels to first order). | `data/` | `tables/field_symmetrized_vg+0.5.csv`, `gate_symmetrized.csv`, `alpha_summary.csv` |
| `sdh.py` | 4 | SdH analysis at Vg = +0.5 V for both voltage pairs: background subtraction, FFT in 1/B, Landau fan of the R_xx minima and of the R_xy plateau centres → densities n_xx and n_QH. | `field_symmetrized_vg+0.5.csv` | `tables/sdh_results.csv`, `sdh_fan_rxx.csv`, `sdh_plateaus_rxy.csv`, `sdh_fft.csv` |
| `hall.py` | 2a, 2b | Hall density from the low-field slope (Vg = +0.5 V) and from the gate sweeps (Vg = 0); calibration factor k = n_QH/n_H of the Hall channel (early and late epochs); plateau table. | `field_symmetrized…`, `gate_symmetrized.csv`, `sdh_results.csv` | `tables/hall_density.csv`, `hall_calibration.csv`, `hall_plateaus.csv` |
| `gate.py` | 3, 5, A9 | n(Vg) (k-corrected Hall), capacitance, d_eff with centroid and quantum-capacitance corrections, lock-in lag from up/down SdH minima, SdH minima vs Vg (second capacitance, spacing vs B), alignment of R_xx minima with R_xy plateaus, appendix A9 exponential fits. | `gate_symmetrized.csv`, `hall_calibration.csv`, `data/RVg-IV-12_14_49` | `tables/gate_density.csv`, `gate_capacitance.csv`, `gate_hysteresis.csv`, `gate_sdh_minima.csv`, `gate_sdh_spacing.csv`, `gate_alignment.csv`, `appendix_A9_fits.csv` |
| `temperature_vs_field.py` | 0.6 | Master curve T_read(\|B\|) from all runs; tests of evenness, hysteresis, ramp vs hold and drift; low- and high-field shape fits; equivalent sensor change for a Si diode and a resistive sensor. | `data/` | `tables/thermometer_master_curve.csv`, `thermometer_all_points.csv`, `thermometer_tests.csv`, `thermometer_fits.csv`, `thermometer_sensor.csv` |
| `rt_model.py` | 1 | Two-channel fit of the cool-down: 2DEG (impurity + Bloch–Grüneisen acoustic + LO phonons) in parallel with an activated channel in the doped AlGaAs; Arrhenius analysis of the parallel conductance. | `data/Rt-IV-11_59_06`, `hall_calibration.csv` | `tables/rt_clean.csv`, `rt_fit_params.csv`, `rt_arrhenius.csv`, `rt_values.csv`, `rt_summary.csv` |
| `dingle.py` | 6, 2c | Dingle plot → quantum lifetime τ_q at T = 4.2 K, with the logged T(B), the background normalisation and the amplitude cut as systematics; τ_q–T degeneracy profile; upper bound on g*. | `field_symmetrized…`, `thermometer_master_curve.csv`, `sdh_results.csv` | `tables/dingle_points.csv`, `dingle_results.csv`, `dingle_T_profile.csv`, `spin_splitting.csv` |
| `summary.py` | 0.5, 7, 8 | B = 0 resistances without a B = 0 gate sweep; mobility, transport time and mean free path with L/W = 2; consistency checks of L/W against own estimates (including a scan of a possible thermometer–sample temperature offset during the cool-down); μ(T) below 100 K; summary table at Vg = 0; uncertainty budget; check of every number quoted in the analysis plan. | all tables above | `tables/b0_values.csv`, `mobility_two_point.csv`, `mobility_vs_T.csv`, `rt_temperature_scale.csv`, `geometry_estimates.csv`, `summary_vg0.csv`, `uncertainty_budget.csv`, `plan_claims_check.csv` |
| `checks.py` | – | 22 numerical sanity checks (bookkeeping, symmetry, fan intercepts, k consistency, ν assignment, thermometer, fit quality, figure files). Exits with an error if any fails. | `tables/`, `figures/` | console |
| `run_all.py` | – | Runs every analysis script, every figure script and the checks in order. | `data/` | everything |

### Figure scripts (each writes one PDF to `figures/`)

| File | Figure | Shows | Inputs |
|---|---|---|---|
| `fig00_setup_circuit.py` | `fig00_setup_circuit.pdf` | Fig. 1 of the paper, one column: Hall bar with its pads (unused contacts as unnumbered stubs) in the LHe bath, wired to the AC source and 1 MΩ series resistor, both lock-ins, the SourceMeter, the Lake Shore 331 and the magnet supply (Oxford PS120) | settings in `params.py` |
| `fig01_cooldown_RT.py` | `fig01_cooldown_RT.pdf` | R(T) with two-channel fit and components; log–log phonon part R − R₀; Arrhenius plot of the parallel channel with local E_a (every 4th data point drawn) | `rt_*.csv` |
| `fig02_symmetrisation.py` | `fig02_symmetrisation.pdf` | raw R₁(±B), R₂(±B) vs symmetrised curves, with the even Hall part R₂^S in its own sub-panel (Ω); α vs \|B\| (100 mT bins) and vs Vg (50 mV bins) | `field_symmetrized…`, `gate_symmetrized.csv` |
| `fig03_thermometer_vs_field.py` | `fig03_thermometer_vs_field.pdf` | (a) logged T vs \|B\| for all runs; (b) low-field and (c) high-field fits of the master curve; (d) each point's deviation from the master curve in mK, one colour per group of measurements, with the mean of each gate-sweep hold | `thermometer_*.csv` |
| `fig04_hall_plateaus.py` | `fig04_hall_plateaus.pdf` (one column); with `FIG04_WIDE=1`: `fig04_hall_plateaus_wide.pdf` (panels side by side, full width) | R_xy^A (as read and /k) with R_K/ν lines and plateau centres; R_xx^S with even-ν positions for both densities | `field_symmetrized…`, `hall_plateaus.csv`, `sdh_results.csv` |
| `fig05_sdh_fft_fan.py` | `fig05_sdh_fft_fan.pdf` | oscillatory parts vs 1/B, FFT spectra, Landau fans of both voltage pairs | `sdh_*.csv` |
| `fig06_density_vs_gate.py` | `fig06_density_vs_gate.pdf` | three panels on one Vg axis with one legend: n(Vg) from Hall (k_late) and SdH minima with linear fits; minus the Hall fit (Hall averaged over down/up, SdH down/up pairs joined); down − up difference vs lock-in lag | `gate_density.csv`, `gate_capacitance.csv`, `gate_sdh_minima.csv`, `gate_hysteresis.csv` |
| `fig07_gate_sweeps.py` | `fig07_gate_sweeps.pdf` | R_xy^A/k and R_xx^S vs Vg at each \|B\| with ν labels | `gate_symmetrized.csv`, `gate_sdh_minima.csv` |
| `fig08_landau_fan.py` | `fig08_landau_fan.pdf` | Landau fan: R_xx^S minima of the gate sweeps in the (Vg, \|B\|) plane on lines of constant even ν (replaces the earlier colour map, which stretched six measured fields into bands) | `gate_capacitance.csv`, `gate_sdh_minima.csv` |
| `fig09_sdh_spacing_capacitance.py` | `fig09_sdh_spacing_capacitance.pdf` | ΔVg between consecutive SdH minima vs B with the C fit | `gate_sdh_spacing.csv`, `gate_capacitance.csv` |
| `fig10_dingle.py` | `fig10_dingle.pdf` | Dingle plot (T = 4.2 K and logged T); τ_q and residuals vs assumed T | `dingle_*.csv` |
| `figA9_exponential_fit.py` | `figA9_exponential_fit.pdf` | appendix A9: exponential and Drude fits of R_xx(Vg) and the failed extrapolation to Vg = 0 | `appendix_A9_fits.csv`, `gate_symmetrized.csv`, `data/RVg-IV-12_14_49` |

### Other

| File | Description |
|---|---|
| `ledger/rulings.md` | Every analysis decision that deviates from or adds to the agreed plan, with reason and cost if wrong. |

## Conventions

- R values are in Ω as logged by the acquisition program, which converts the lock-in voltage with the excitation current I = 1.0 µA (1 V source, 1 MΩ series resistor); resolution 5 Ω. Hall values marked "true" are divided by the empirical Hall-channel scale factor k (`hall.py`).
- Densities in 10¹⁵ m⁻² (= 10¹¹ cm⁻²). Geometry-dependent quantities (μ, ρ_xx, τ_tr, ℓ) use L/W = 2 between pads 6 and 11 (lab instructor); per-unit-L/W values are also tabulated.
- Figures: where data are dense, figures draw every n-th point or bin-averages (stated in each script's docstring and in the figure); all fits and numbers use every point.
- Figure size: every figure is drawn at its printed width in the REVTeX two-column paper, 3.4 in for one column (`COL_W`: figs 4, 8, 9) and 7.0 in for a full-width `figure*` (`FULL_W`: all others), with 8–8.5 pt text and 7.5 pt in-plot labels (`params.plot_style`). Include them with `width=\columnwidth` or `width=\textwidth`; they are then printed at 100 % and the fonts are the stated sizes. Multi-panel figures carry the panel letters (a), (b), … used in the paper. Legends sit below the panels in a soft light-grey frame (`params.legend_below`, `LEGEND_BOX`) or in an empty corner, so they never cover data; line style or marker differs between data sets, so every figure reads in greyscale.
- Dingle analysis: amplitudes are normalised by R_xx(B = 0); normalising by the field-dependent background instead roughly doubles τ_q and is quoted as a systematic.
