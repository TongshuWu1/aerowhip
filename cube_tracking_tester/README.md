# RGB-D cube tracking tester

This launcher remains intentionally independent of the cable application. It
tests whether the live ZED RGB-D observation is sufficient to recover a known
150 mm bright-yellow cube without a neural network or fiducial markers. The
tracking implementation itself is canonical in
`ZED_segmentation_viewer/source/cube_tracker.py` and is shared with the main
pipeline.

## Measurement

Each frame performs exactly one observation path:

1. Threshold bright yellow in the rectified left image.
2. Keep the largest yellow connected component.
3. Unproject its registered depth pixels.
4. Robustly extract planar subsets.
5. Use three mutually perpendicular faces when they are visible.
6. With two perpendicular faces, measure their shared-edge span and require it
   to agree with the known 150 mm cube width.
7. Infer the cube centre from the visible outward face planes and known size.

Two faces determine the three cube axes. Their measured shared-edge midpoint
supplies the centre coordinate not constrained by the two face planes. An
incomplete span is rejected rather than extrapolated. With fewer than two
reliable perpendicular faces, the frame is invalid. The tester does not hold,
predict, or synthesize a pose.

Cube orientation is geometrically equivalent under the 24 rotational
symmetries of a cube. The displayed quaternion selects the equivalent
representation closest to the preceding valid frame. This changes no measured
surface and prevents representation-only 90-degree jumps.

## Run

Close every other application using the ZED, then run from the project root:

```powershell
.\.venv\Scripts\python.exe .\cube_tracking_tester\run_cube_tracking.py
```

Controls:

- `Q` or `Esc`: finish the run.
- `S`: save the current annotated frame.

The window shows the yellow component, the visible face samples, the fitted cube
wireframe, surface RMS, depth coverage, and processing time.

Every run saves:

- `poses.csv`: exact ZED image timestamp, raw cube pose, surface RMS, valid
  depth coverage, and processing time.
- `last_frame.png`: the final annotated image.
- `config.toml`: the exact tester settings.

Files are written beneath
`diagnostics/experiments/cube_tracking/<timestamp>/`, which is already outside
the tracked research source.

## Interpret the first test

Place the cube so the camera sees at least two complete adjacent faces. A
useful result has:

- a stable green wireframe aligned with all cube edges;
- two or three differently coloured face-point groups;
- high depth coverage;
- surface RMS clearly below the cable radius of 4.5 mm.

If the yellow contour is wrong, adjust only the `[yellow]` HSV limits in
`config.toml`. If the contour is correct but the fit is invalid, inspect depth
coverage and the visible surface in the ZED depth view before changing the
metric plane threshold.

## Synthetic verification

```powershell
.\.venv\Scripts\python.exe -m unittest cube_tracking_tester.test_cube_tracker
```
