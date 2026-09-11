# Selected M2 prospective flight

User selected rehearsal **20260910-211435-608306-M2-frozen-refit-v1-whip** from the displayed list.
Use **fullstate_30hz.csv**: the complete **10.3 s** desired P/V/A command, including
recovery and final hold. Its whip portion is 1.166667 s. Model: M2-frozen-refit-v1;
MPPI impact weight 1600. Commands and prediction are copied byte-for-byte.

Start tracked origin: **[0, 0, 1.255] m**. Target: **[1.25, 0, 1.0] m**.
Use the established execution workflow: take off, hold 10 s at the saved start,
execute the complete 30 Hz CSV, then land. Takeoff, initial hold and landing are
outside this CSV. No sender, hardware or controller setting is changed here.
Predicted contact speed is 4.90513 m/s; physical performance remains to be measured.

Record paired controller and native OptiTrack logs including the initial hold,
whip and recovery. Keep original full recordings, raw global coordinates and
missing-marker gaps. Note any contact, intervention or settings/hardware change.
Place the pairs in:

`C:\Users\wts28\Documents\PHD\particle_filter_cable_project\rehearsal_csv_and_result_in_real_flight\M2_whip_20260910-211435-608306\flight_take`

Use names `experiment_whip_m2_001.csv` (controller) and `whip_m2_001.csv` (OptiTrack),
continuing 002–005. For five repeats, roles are predeclared before viewing outcomes:
001/002/004 adaptation, 003/005 validation. Data quality and valid free-motion
intervals still require review; no future flight condition has been assumed.

Compare measured M2 flights with this exact original **rehearsal.npz** first.
Any later M3 fitting follows the frozen staged-identification method after data
review and explicit authorization. No fitting or physical flight is started here.
The original M0/M1/M2 development artifacts and former selection are preserved.

Command SHA-256: `b34ddfe74a0292905c2173082f4d2a76e0b798c8c2e28dc8b7ce1be9b9c2d6b3`.
Forecast SHA-256: `452300b7067cc5cd324dde2954d231ed15ff6ebbcb49911b5fc9f80a321a7656`.
The ZIP contains the frozen model assets, source, commands and forecast.
