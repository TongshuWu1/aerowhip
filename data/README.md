# Measurements

| Folder | Contents |
|---|---|
| `raw_takes/` | Imported preliminary takes, original OptiTrack/controller logs and metadata |
| `flight_batches/` | Paired recordings and exact flown commands, grouped by collection batch |
| `processed_takes/` | Derived take data, created only by explicit processing |

The retained `flight_batches/` collection contains `preliminary1` and the
10 September M0, M1 and M2 whip batches. Keep each batch's `flight_take/`,
`simulation_csv/`, alignment and protocol files together. Preserve filenames,
timestamps, exclusions and roles; renaming a folder does not change model identity.

The Python processing implementation is in `experimental_data/`; actual
measurements belong here. Read the [data lifecycle guide](../docs/methods/DATA_LIFECYCLE_AND_RECORDING_GUIDE.md)
and [experiment protocol](../docs/paper/PAPER_EXPERIMENT_PROTOCOL.md).
