# Preliminary1 intake review

Five paired takes imported with exact raw-file hashes. User confirms no contact, intervention or settings changes. OptiTrack exports are intentionally hand-trimmed airborne intervals; original files are unchanged. Controller XYZ is OptiTrack-derived, not an independent sensor.

| Take | Exported motion (s) | Pose complete | Missing marker frames C1-C10 |
|---|---:|---|---|
| figure8_001 | 51.12 | True | [3, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| figure8_002 | 52.99 | True | [0, 0, 2, 1, 1, 2, 3, 6, 13, 14] |
| osci_001 | 46.59 | True | [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] |
| osci_002 | 24.49 | True | [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] |
| vertifig8_001 | 29.61 | True | [342, 342, 342, 342, 342, 342, 342, 342, 342, 342] |

All Motive exports use cf_3, 100 Hz global metres and quaternion orientation. Controller files have a ~100 Hz row rate with ~10 Hz changes in cached OptiTrack XYZ. Use native pose for motion derivatives.

The vertical pair is experiment_vertifig8_001.csv / veritfig8_001.csv, explicitly confirmed by the user. All-marker dropout intervals in original OptiTrack time:
- 47.57–50.98 s: 342 frames.

Missing data remain missing. No interpolation over gaps, normalization, time shift, fitting or model publication has been applied. Intake is not approval of every sample: clock alignment, finite/geometry masks and data roles still need review. No fixed one-second whip crop applies to these longer preliminary movements.
