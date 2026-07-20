## Overview
This repository contains the full data and code pipeline behind a physics-informed neural network (PINN) reconstruction of sediment bed-level change in a nature-based stormwater detention pond, plus the conventional baselines it is benchmarked against.

A pair of cheap temperature sensors (one in the water column, one buried at ~0.02 m in the bed) record half-hourly temperature. Fitting a single diurnal harmonic to each day gives amplitudes and phases, from which the Luce–Tonina (LT) amplitude-ratio / phase-shift formulation yields a daily bed-level change proxy Δz_LT. Two GNSS levelling surveys (104 points, 2024-12-17 and 2025-10-30) anchor the total displacement. A PINN then reconciles the two: it learns a temperature field T(z, t) over a 0.30 m 1-D sediment column that simultaneously (a) satisfies the 1-D heat equation, (b) matches the LT-derived surface forcing, and (c) reproduces the GNSS displacement endpoint through a thermoelastic kernel — while jointly inferring two physical scalars, thermal diffusivity κ and an effective thermoelastic coefficient α.

The framing is deliberate about what the method can and cannot do:

•	κ is an upper bound, not a measurement. The loss surface is flat below ≈10⁻⁷ m² s⁻¹, so the inferred value (1.07e-07 m² s⁻¹) should be read as a loosely-bounded calibration value.

•	α is negative (≈ −1.0×10⁻² K⁻¹). The column warms across the window while the bed settles, so a positive expansion coefficient cannot reproduce the GNSS endpoint. The negative sign encodes net thermally-driven consolidation of organic, water-saturated sediment (Delage et al., 2000) and is a lumped calibration scalar.

•	The PINN is spatially blind. It reconstructs the temporal trajectory at the probe location only. Per-point spatial heterogeneity is a GNSS-kriging product, and GP kriging beats the PINN on leave-one-out spatial skill (skill score ≈ −1.20). This is reported, not hidden — it is the honest boundary of the contribution.

## Reuse and extension
This pipeline is deliberately small, CPU-only, and built from two cheap temperature sensors plus two survey epochs — the point is transferability to other low-instrumentation nature-based assets. Common adaptations:

•	Apply to a different pond. Replace temperature_sensors.csv (same four columns, day-first timestamps) and GNSS.xlsx (same six columns). Then check, in order: Z_MAX (column depth) and Z_SENSOR (burial depth) in common.py; the T_BASE / DELTA_T envelope anchor in load_data(), which is calibrated to British pond conditions and will not transfer unexamined; and ke in compute_delta_z_LuceTonina().

•	Change the physics. pde_residual() in common.py is the single place the 1-D heat equation lives. thermo_u() is the single place the thermoelastic kernel lives. Both take model and are used identically by the PINN, the MC members, and (for the kernel) the B2 FD baseline — so a change propagates consistently across the comparison.

•	Reweight the loss. LAMBDA_PDE / LT / GNSS / BC / IC are module-level constants. The current GNSS weight (20) is the highest because the survey endpoint is the only absolute anchor in the system; lowering it lets the surface fit dominate and α drifts.

•	The obvious open extension. The PINN currently has no spatial dimension, which is precisely why GP kriging beats it per-point. Extending the network input from (z, t) to (x, y, z, t) and fitting against per-point dz_gnss rather than the spatial mean is the natural next step — the GNSS file already carries the eastings and northings.

## Data description and availability
The raw data used for this project is available upon request: temperature_sensors.csv — raw sensor record (11,673 rows); revised_daily_temperature_harmonics_and_delta.csv — daily harmonics (308 rows);daily_delta_z_from_LT.csv — LT solver output (308 rows); cleaned_lt_daily_delta_z.xlsx — model input (308 rows);GNSS.xlsx — levelling survey (104 points)  
## Licence
If you use this pipeline, please cite the accompanying manuscript (TBC).
## Acknowledgements
Fieldwork and analysis conducted at UWE Bristol as part of doctoral research on condition-based sediment management of nature-based stormwater assets in the River Chew catchment, North Somerset.
