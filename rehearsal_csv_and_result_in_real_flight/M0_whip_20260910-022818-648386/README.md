# New M0 whip recordings

Put each paired original recording in `flight_take/`:

- `whip_001.csv`: native global OptiTrack export, metres/quaternion.
- `experiment_whip_001.csv`: matching controller command/timing log.

Repeat with 002/003 if practical. Protocol roles are 001/002 adaptation and
003 validation, fixed before viewing outcomes. With one take, only development
adaptation can be assessed; there is no independent M1 test.

`simulation_csv/fullstate_30hz.csv` is the exact selected command, including
recovery. Its original forecast is linked by `protocol.json`; preserve both.
Record the initial hold, whip and recovery. Keep full original recordings and
note physical contact timing, intervention or hardware/controller changes.

`time_alignment.json` is deliberately empty until real timing evidence is
available. Do not enter zero merely to make comparison run. It requires the
source-bound offset between the two recorded clocks.

See [the current adaptation guide](../../docs/M0_TO_M1_ADAPTATION.md).
Nothing in this folder starts a fit automatically.
