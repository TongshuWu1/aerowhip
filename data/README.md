# Experimental data contract

`raw_takes/` contains immutable source files, one logger/Motive pair per take.
`processed_takes/` contains deterministic `aerial_cable_take_v1` products.
`dataset_manifest.json` contains mutable scientific decisions (whole-take
Training/Validation/Ignore roles and manual Use/Exclude intervals) without
modifying either raw or processed evidence. `fit_results/` is reserved for
immutable fit snapshots and results once all fit gates pass.

The GUI and fitting code read processed takes only. Motive labeled markers
`cable:c1` through `cable:c10` are the cable observations. Live/unlabeled
logger cable IDs are intentionally absent from the processed fitting schema.

Run:

```powershell
.\.venv\Scripts\python.exe process_all_takes.py
```

Do not manually edit a processed NPZ or its metadata. Change processing code
or configuration and regenerate it; the source/config/processor fingerprint
records provenance and unchanged reruns are skipped.
