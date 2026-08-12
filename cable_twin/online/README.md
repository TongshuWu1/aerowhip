# Online constrained-rod particle filter

Run live tracking with the canonical accepted cable model:

```powershell
.\.venv\Scripts\python.exe run_online.py
```

The live process uses the shared PIDNet, route extraction, registered ZED depth,
torsion-free constrained-rod equations, and 3D viewport. The particle filter
outputs the 24-node MAP centerline, posterior covariance, weights, ESS, velocity,
and prediction-only state. The viewport displays only the information needed
during tracking.

The online registered-depth archive is a different observation schema from the
offline planar archive. Both use the same fitted physical model; planar
recordings must not be passed directly to `cable_twin.online.replay`.
