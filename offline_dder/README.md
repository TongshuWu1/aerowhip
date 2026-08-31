# Offline DDER EI/Cb identification

This folder is a historical, self-contained copy of the predecessor offline
cable-identification workflow retained for scientific provenance.

It is **not** imported by the active simulator, Identification GUI, current
PhysicalEpisode fitter, validation path, or active-model manifest. Its
geometries, one-pivot/free-tangent boundary, and one-second reinitialization
problem differ from the current aerial 12-node rigid-clamp system. Do not use
its values as current runtime parameters.

It identifies a one-attachment, mechanically free-tip cable from OptiTrack
measurements using the existing three-dimensional DDER implementation. The only
estimated material parameters are:

- homogeneous bending stiffness `EI` (`N m^2`); and
- objective Kelvin-Voigt bending damping `Cb` (`N m^2 s`).

The physical model, optimizer, validation protocols, fitted artifacts, and
focused tests were copied without changing the identification method. The live
geometry default was subsequently updated on 2026-08-27 from the superseded
78-mm attachment-to-c1 interval to the measured 60-mm interval. Existing fitted
artifacts retain their original 0.961-m geometry as provenance and are not
valid parameter results for the new 0.943-m geometry until the cable is refit.

## Launch

From the repository root:

```powershell
.\.venv\Scripts\python.exe offline_dder\run_offline_fitting.py
```

The detailed measurement contract and workflow are documented in
`optitrack_offline/README.md`.

## Layout

- `optitrack_offline/` — Motive CSV parsing, configuration, fitting, held-out
  validation, GUI, fitted model artifacts, and prior validation outputs.
- `cable_twin/shared/` — the exact shared DDER and accelerated numerical backend
  required by the fitter.
- `tests/` — focused CSV, one-attached DDER, and validation regression tests.
- `run_offline_fitting.py` — GUI entry point.

## Provenance

- Source snapshot: `legacy/current_baseline_2026-08-27/`
- Source Git branch: `twin-rewrite`
- Source Git HEAD: `cbdb59b`

The older planar/PIDNet offline workflow was intentionally not promoted. It
remains available in the legacy snapshot but represents a different sensing
pipeline from the authoritative OptiTrack one-pivot/free-tip EI/Cb fit.
