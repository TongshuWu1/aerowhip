# Offline planar identification

Run:

```powershell
.\.venv\Scripts\python.exe run_offline.py
```

## Physical setup

1. Measure and save the active cable length, mass, and diameter.
2. Keep the camera fixed and level.
3. Hold the two endpoints and keep the cable approximately in one plane parallel
   to the image sensor, at roughly constant distance from the camera.
4. Keep the full cable and both endpoint markers visible and avoid object contact.

No wall, calibration target, depth, stereo matching, or point cloud is used.
The planar assumption is an intentional Phase 1 approximation: appreciable
out-of-plane motion appears as foreshortening and cannot be recovered from RGB.

The Camera panel shows ZED exposure and gain readback. Use **Apply manual** to
shorten exposure when motion blur is visible, then increase gain only enough to
recover brightness. Settings are locked while recording and the verified values
are stored in the capture manifest. **Auto** returns to automatic exposure/gain.

## Recordings

Every recording has the same role. Hold only the two endpoints and include:

1. slow endpoint motion through several cable shapes;
2. brief pauses;
3. several faster, controlled endpoint changes; and
4. one or two seconds with both endpoints stationary after each faster change,
   so the cable's settling motion is visible.

You may fit one recording or select several recordings together. More
recordings are useful when they add different shapes and motion speeds.

Capture saves raw lossless SVO2. **View 2D selected** runs PIDNet offline when
needed and shows the RGB centerline beside the 24 PIDNet observation nodes used
by the fit. These node positions are neither smoothed nor forced to satisfy the
rod length constraint. The metric conversion uses one constant scale for the
entire recording, computed from the known cable length and the median complete
PIDNet curve length. It never rescales individual frames.

**Fit EI + Cb** automatically extracts any missing 2D trajectories and jointly
fits the same two homogeneous cable parameters to every selected recording. The
result panel reports `EI`, `Cb`, the metric size of one image pixel, and the rod
rollout residual against PIDNet.

The implemented dynamics are a differentiable torsion-free constrained elastic
rod: an inextensible spring-joint chain using nonlinear DER bending geometry.
They are not a full twist-aware DDER. See [CABLE_MODEL.md](CABLE_MODEL.md).

