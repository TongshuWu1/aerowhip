# Flight-program exports

Save complete command CSVs and trajectory/model bundles here when handing them
to the separate flight program. The main research UI now defaults to this folder.
These are offline exports; saving a file does not send commands to an aircraft.

Original planned commands and predictions remain with their immutable run in
`runs/rehearsals_pva/`. Pair the command actually flown with recordings in
`data/flight_batches/`. On the `deployment` branch, exports are grouped further
as `exports/<study>/<generation>/`.
