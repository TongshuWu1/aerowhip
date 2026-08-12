# Cable Twin

One shared differentiable torsion-free constrained elastic-rod model with
separate offline and online workflows.

## Offline identification

```powershell
.\.venv\Scripts\python.exe run_offline.py
```

The offline workflow uses 2D PIDNet curves, not depth or a point cloud. Keep the
camera fixed, move the fully visible cable approximately parallel to the image
plane, and record one or more endpoint-driven motions. One constant metric scale
per recording is obtained from the known cable length; the resulting trajectories
identify the shared bending parameters `EI` and `Cb`.

One model is published per cable:

```text
data/offline_dder/models/cable1_dder.json
```

See [PIPELINE.md](PIPELINE.md) and
[cable_twin/offline/CABLE_MODEL.md](cable_twin/offline/CABLE_MODEL.md).

The compact internal parameter-recovery check is:

```powershell
.\.venv\Scripts\python.exe run_model_check.py
```

It saves `data/offline_dder/synthetic_recovery.json`. This uses known synthetic
parameters to check the implementation; it is not experimental validation.

## Online tracking

```powershell
.\.venv\Scripts\python.exe run_online.py
```

The online PF uses the fitted shared dynamics and the existing live 3D RGB-D
observation. Its observation path is intentionally independent of the planar
offline identification.

## Layout

```text
cable_twin/
  shared/   PIDNet, route geometry, constrained rod dynamics, model artifact
  offline/  raw capture, image-plane 2D extraction, EI/Cb fitting, review UI
  online/   live 3D particle filter and replay
```
