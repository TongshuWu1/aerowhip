# adp0 hover height and paired-log audit

The user observed measured drone hover above the saved prediction and asked whether this was a CSV-pairing mistake. The user confirmed that **both measured XYZ streams originate from OptiTrack**. The controller-side logger calls `cf.get_position()`; it does not provide independent onboard-estimator telemetry.

The audit supports the existing file pairings and finds that the hover height discrepancy exists within each controller log, without any cross-file time matching. No raw data, prediction, model, policy or flight CSV was changed. No fitting, training, spatial shift or flight was performed.

## Pairing and columns

- Direct CSV parsing identified `Rigid Body / cf_7 / Position / X,Y,Z` at zero-based columns 6,7,8 in all five OptiTrack files. These raw values exactly equal the existing loader's drone-origin arrays. They are not cable-attachment or unlabeled-marker coordinates.
- Independently searched all 25 OptiTrack/controller file combinations with only one time offset and no spatial transform. Every same-numbered pair has the smallest XYZ RMS in its row. Correct pairs: 1.816, 1.899, 1.858, 1.822, 1.906 cm over the whole moving recording. Incorrect pairs: 2.924–5.392 cm. The independent XYZ alignment agrees with the production alignment numerically.
- Each corresponding controller log contains the exact 245 distinct dynamic packets from the flown CSV. The packet-derived onset spread is 0.37–1.27 ms. Packet reception and mocap/log alignment are distinct operations; these figures are not proof of synchronized physical acquisition clocks.
- During the last 0.21 s before the whip, matched Z differs by only 0.071–0.231 mm RMS. In the common final-hover window, 9.2–10.9 s into the CSV, it differs by 0.141–0.234 mm RMS. Thus the several-centimetre hover discrepancy is not a discrepancy between the two measured Z streams.
- Dynamic agreement is not exact: the full-motion XYZ RMS remains about 1.8–1.9 cm. Axis-only alignment sometimes chooses a neighboring logging interval about 9 ms away. Sampling, reception latency and hold behavior remain limitations. This does not explain a sustained hover offset while commands and height are nearly constant.

## An example requiring no cross-file matching

In `flight_take/experiment_whip_adp_0_001.csv`, line 1757:

| Field | Value |
|---|---:|
| `time_s` | 17.550185680389404 |
| Measured `z` | 1.327895164489746 m |
| Desired `cmd_z` | 1.255 m |
| `cmd_vz`, `cmd_az` | 0, 0 |
| `cmd_valid` | 1 |
| `cmd_age` | 0.032276153564453125 s |

The measured height is **7.2895 cm above its command in this same log row**. The entire checked pre-hover interval is commanded to [-2,0,1.255] with zero V/A. The checked final hold has that same command.

## Height differences across the five flights

These are signed mean Z differences, not RMS trajectory errors. Before-whip values compare with the saved nominal start at 1.255 m; the rehearsal has no predicted samples before CSV time zero. Final hold compares native measured samples with the exact original saved forecast, whose mean Z is approximately 1.252 m.

| Flight | Before whip: measured minus nominal start | Final hold: measured minus prediction |
|---|---:|---:|
| 001 | +7.284 cm | +6.659 cm |
| 002 | +5.680 cm | +4.767 cm |
| 003 | +4.606 cm | +3.927 cm |
| 004 | +7.621 cm | +6.313 cm |
| 005 | +4.935 cm | +4.481 cm |

Both the saved predicted origin and measured cf_7 position refer to the tracked drone origin. The separate rotated 55 mm origin-to-cable attachment offset is applied to cable geometry. There is no basis here for subtracting 55 mm from measured drone positions. Variation between flights and pre/post hold also argues against treating this as a single fixed geometric translation.

## Model initialization consequence

`experimental_data/current_adaptation.py` uses 21 causal pre-hover samples to initialize fitting. `simulator/drone_pose_response.py:initialize_from_hover` supplies measured position/velocity/orientation and an inferred, frozen effective compensation. Under the fitted M1 parameters, the five initial Z compensations are approximately 1.047, 0.813, 0.646, 1.030 and 0.841 m/s². These are effective model terms, not measured thrust, firmware integrator values or an identified physical cause.

`simulator/research_pose.py:settled_initial`, used for nominal training/rehearsal, instead assumes the requested starting position, level orientation and zero compensation. The current recorded-initialization fit and ideal-hover deployment therefore have different initial-state assumptions. A good conditional model fit does not establish that a nominal launch at the commanded height reproduces these real launches.

Before choosing a remedy, distinguish an onboard-estimator discrepancy from a true controller hover-tracking error by recording the firmware's own estimated position alongside the OptiTrack measurement and desired command. The present two measured streams cannot make that distinction because they share the OptiTrack source. Do not infer a mass, voltage, thrust or offset correction from this audit alone. If measured-hover initialization is added to future generation, keep actual initial state, preceding hover command and world target as distinct inputs; simply raising the desired start to the measured height is not the same correction.

## Reproduction

Run `check_hover.py` and `check_pairing.py` here with the project Python. `hover_height.json` and `pairing_check.json` contain exact windows, hashes and numerical checks. `hover-height.png` / `.pdf` show the signed hover differences. All source hashes passed unchanged checks. The native ghost and reported real-flight RMS remain the original preflight comparison.
