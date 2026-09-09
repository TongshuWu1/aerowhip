# Completed per-take normalization

User selected separate constants; normalized data are now saved and active. Upward shifts001:8.7518cm,002:9.08975cm,003:6.3258cm,004:8.589025cm. All four cover the whip. Drift remains; original failed screening is preserved with explicit user review.15 tests passed,3 historical skipped. No raw input changed.

# adp1 timing and pending normalization

Four cf3 pairs001â€“004. User explicitly excludes005 (not currently present); exclusion manifest prevents later inclusion without changing raw files. Exact CSV SHA08d779ffeb33356111e6f60557fb1d74dad3146abfc323d00fd9f939c73e6408 matches saved rehearsal20260909-030657-671710 and fresh normalized M1 policy checkpoint424b5be3c20b8ba1b7d067eb4f53ee1a10045a860ff832d7853c02b565dd7fbb.

User authorized matching logs. Controller XYZ used for estimating time offsets ONLY; all evaluation/normalization state remains native OptiTrack cf3/cable. Four offsets(controller_time=OptiTrack_time+offset): -0.30479081,+0.20545351,+0.66547139,+0.11518737s. Independent XY and Z estimates agree within0.25ms; residual measured-stream RMS1.75â€“1.88cm includes asynchronous/stale logging. Agreement is not proof of hardware synchronization.

Original pre/post batch normalization estimates bias-0.086704125m, so proposed upward shift+8.6704125cm. Per-take equal pre/post biases: -0.087518,-0.0908975,-0.063258,-0.08589025m. Screening FAILS:001/002 pre/post disagreement, hover drift/spread,003 between-take deviation. Final1s pre-hover check does not resolve all drift. Do not mark checks_passed true or claim normalization complete. User asked shared correction with flags versus per-take constant; awaiting response. Current scoring correctly blocks invalid calibration.

No refit, training, ghost regeneration, command change or raw-log edit.11 focused tests pass, including persistent005 exclusion.
