# M6 horizontal command correction

This is the former M7 package, renamed M6 at the user's request. The original intermediate model and command are archived. The command preserves the original M5 horizontal desired trajectory and timing; renaming does not change the command, forecast, or model weights.

- Required starting tracked-origin hover: (0.0, 0.0, 1.7917713383723264) m.
- Fixed planned strike time: 1.293333333 s.
- Complete 30 Hz CSV: 306 rows, 10.167 s, including recovery and final hold.
- Use the existing FullState controller and established lab procedure; first row is the required hover, not takeoff.
- Recordings are stored under `flight_take/M6`. Active flight selection is unchanged.

Predicted tip-reference RMSE under this M6 model: 4.399 -> 1.224 cm. These are simulation results. Eight measured whips in two recordings are now available; they initialize the next full-model update, called M7, from this M6 predictor.

The naming and original model ancestry are recorded in `archive/horizontal-renumber-20260916/manifest.json` at the project root. The preflight prediction remains frozen and is not replaced by the post-flight refit.

CSV SHA-256: `16b45834deb1be3f8de377117c29af08b52bb875af2f3711067e67f0ed6ee7ec`
