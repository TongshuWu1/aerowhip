# Measured UAV Attachment and First Cable Interval Update

Date: 2026-08-27  
Scope: geometry and geometry-derived mass discretization only

## Outcome

The active UAV–DDER simulator now represents the newly measured attachment
geometry directly:

- UAV rigid-body reference to connector: **50 mm along body -Z**;
- connector to first OptiTrack cable marker `c1`: **60 mm**;
- clamped cable tangent: **body -Z**;
- DDER refinement: two rod edges per measured marker interval, so the first
  two edge rest lengths are **30 mm each**.

No DDER equation, UAV response equation, material parameter, time step,
substep count, constraint iteration count, OptiTrack marker mapping, MPPI
method, observer method, or fitting objective was changed.

## Coordinate convention and signed attachment axis

The implementation uses active `xyzw` quaternions that rotate body-frame
vectors into the world frame. Gravity is along world -Z, and the already
established clamped exit tangent was `[0, 0, -1]` in the body frame. Therefore
the measured statement "50 mm physically downward" is represented as:

```json
"attachment_offset_body_m": [0.0, 0.0, -0.05]
```

The tangent remains:

```json
"attachment_tangent_body": [0.0, 0.0, -1.0]
```

Both vectors are transformed by the same UAV orientation. At identity attitude
the connector lies 50 mm below the UAV reference. Under a +90-degree roll,
both the offset and tangent rotate from body -Z to world +Y, confirming that
the attachment is rigidly body-fixed rather than a world-frame graphics
offset.

## Cable geometry

The authoritative measured connector/marker interval list is now:

```text
[0.060, 0.085, 0.100, 0.100, 0.098,
 0.100, 0.100, 0.100, 0.100, 0.100] m
```

The first interval alone changed, from 78 mm to 60 mm. No length was moved to
another interval. The sum of the measured intervals is:

```text
L = 0.943 m
```

The previous 0.961-m value was the sum of the old interval list, not a second
independent active length constraint. Its 18-mm difference is exactly the
first-interval correction. The current simulator derives cable length from the
interval list and does not contain an independent `length_m` or
`total_cable_length` field that could silently force the old total.

With `segments_per_marker_interval = 2`, the 60-mm connector-to-c1 interval is
represented by:

```text
connector/node 0 -> node 1: 0.030 m
node 1 -> c1/node 2:         0.030 m
```

The OptiTrack material-site mapping is unchanged:

```text
c1...c10 -> DDER nodes [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
```

## Mass discretization

The physical masses were held fixed:

- bare cable mass: `0.007 kg`;
- ten moving marker masses: `10 x 0.000909091 kg`;
- total dynamic cable-plus-marker mass: `0.01609091 kg`.

Because total bare mass is fixed while the measured length changed, the derived
bare linear density is now:

```text
rho = 0.007 / 0.943 = 0.007423117709437964 kg/m
```

The existing length-weighted vertex-mass construction then redistributes that
bare mass over the new nonuniform edge lengths. For example:

```text
bare node 0 mass = 0.000111346765641569 kg
bare node 1 mass = 0.000222693531283139 kg
```

Marker point masses remain added at their unchanged even-numbered DDER nodes.
No mass was invented or removed.

## Code and data provenance

Updated active sources:

- `config/default.json` — measured connector offset and interval list;
- `README.md` — current measured geometry and derived total;
- `tests/test_measured_geometry.py` — geometry, rotation, mass, mapping, and
  rollout invariants;
- `tests/test_dder_regression.py` — new intentionally frozen trajectory hashes
  for the measured geometry;
- `offline_dder/optitrack_offline/config.json` and `config.py` — live offline
  EI/Cb fitting defaults now use the same 60-mm first interval;
- offline DDER documentation and tests — current total is 0.943 m.

Existing fitted model artifacts that record a 78-mm first interval and
0.961-m total were **not rewritten**. They preserve the geometry under which
they were produced and are explicitly documented as historical, superseded
fits. They must not be presented as fits of the newly measured cable. The
archived legacy project was likewise not modified.

## Verification

The following invariants were directly tested:

1. identity-attitude UAV-to-connector distance is exactly 0.050 m;
2. connector-to-c1 distance is exactly 0.060 m;
3. the first two DDER edges are exactly 0.030 m each;
4. the first edge is aligned with the prescribed clamped tangent;
5. attachment offset and tangent rotate together with UAV attitude;
6. measured intervals sum to 0.943 m and are the sole cable-length source;
7. bare, marker, and total masses remain conserved;
8. OptiTrack marker mapping remains unchanged;
9. CPU rollout remains finite and satisfies edge constraints;
10. production CUDA, differentiability, clamped-boundary, GUI, and observation
    contracts remain valid through the repository regression suite.

Verification results:

```text
current simulator repository tests: 33 passed
offline_dder tests:                  20 passed
```

The scientific 3D viewer was also inspected. It renders the connector and
cable from the simulator state, so it automatically shows the measured 50-mm
offset and shortened first interval; no separate visual-only geometry was
added.

## Scientific interpretation

This is a measured-geometry correction, not a new physical or control method.
Old trajectory hashes necessarily changed because the initial and prescribed
cable geometry changed. The production/reference runtime equivalence,
gradient, clamped-boundary, and observation-boundary tests continue to validate
the same algorithms under the new physical dimensions.

The active simulator and future EI/Cb fitting should now use the 0.943-m
geometry. Previously fitted 0.961-m artifacts require refitting before they can
be interpreted as physical parameters for this cable.
