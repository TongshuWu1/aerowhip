# Online OptiTrack-identified rod particle filter

This ZED/PIDNet particle filter is a retained legacy prototype, not the public
online application. For reproducibility, launch it with:

```powershell
.\.venv\Scripts\python.exe -m research_tools.legacy_zed_online
```

For a test that can be inspected and replayed afterward, record the raw
synchronized ZED stream at the same time:

```powershell
.\.venv\Scripts\python.exe -m research_tools.legacy_zed_online --record-svo data\online_recordings\pf_test_01.svo2
```

The repository-root `run_online.py` instead launches the simulated OptiTrack
hidden-plant testbed. That testbed currently has no live hardware connection or
online adaptation.

The destination must not already exist. Closing the 3D viewer cleanly stops and
finalizes the lossless SVO2 recording.

The default identification artifact is
`optitrack_offline/models/cable_model.json`. Its homogeneous material
parameters `EI` and `Cb` were fitted from the 0.961 m OptiTrack specimen. The
online cable is the earlier 0.518 m cable of the same material and 3.5 mm
diameter. Deployment therefore transfers only `EI`, `Cb`, and measured bare
linear density. It constructs a 12-node, 0.518 m rod with uniform 47.1 mm edges
and a 3.77 g bare mass. It does **not** reuse the long specimen's unequal marker
spacing or 17 g instrumented mass.

Six numerical substeps are used because the shorter online edges are stiffer
numerically at the same physical `EI`. This gives the current model a 1.61x
margin below the conservative explicit stability limit at the configured 34 ms
maximum PF step; it changes numerical integration, not the identified material
model. These online physical settings are explicit in `config.toml`.

The online particle batch solves the same implicit Kelvin--Voigt damping
system with fixed-iteration conjugate gradient so that the complete transition
can be captured as a CUDA graph; offline fitting and validation retain the
direct Cholesky solve. Four mass-weighted length-projection iterations keep the
measured maximum edge error near 2 micrometres while avoiding redundant work.

Each live frame follows one direct observation path:

1. PIDNet extracts the cable-body and endpoint masks in the left ZED image.
2. The body mask is skeletonized into an ordered visible 2D route.
3. Registered depth initializes the metric cable and measures visible endpoint
   translations. It is not used to construct a recurring 3D body centerline.
4. Every 3D particle advances with the fitted torsion-free rod dynamics and
   exact edge-length constraints.
5. Particles are projected into the left image. A robust one-way distance from
   observed skeleton pixels to the projected particle curve updates their
   weights; this does not penalize cable portions hidden by occlusion. Pixel
   order is not used after initialization, so the geodesic graph is not rebuilt
   when the likelihood does not require it.

The 3D viewport colors PIDNet-selected point-cloud samples orange and draws the
PF MAP centerline green. The side panels show the RGB segmentation and ordered
skeleton. The filter also maintains weights, ESS, node covariance and velocity.

Initialization currently requires one frame in which the complete cable and
both endpoints have valid registered depth. Hold the cable approximately still
and fully visible until the green centerline appears. Afterwards, partial 2D
routes can update the filter and hidden endpoints use a no-motion prior.

This is a centerline-only phase. Torsion, contact and force are not implemented.
