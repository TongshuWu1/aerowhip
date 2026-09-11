# AeroWhip export and compatibility modules

For the current operator workflow, use the **`deployment` branch** and its
`README.md`, `setup_lab.py` and `docs/LAB_RUNBOOK.md`. The main research app is
introduced in the [repository README](../README.md). It generates complete
30 Hz desired-PVA CSVs for the laboratory's separate flight program.

This source package contains saved rehearsal, recovery and CSV utilities.
`pva_rehearsal.py` and `curved_recovery.py` support the current PVA workflow.
It does not contain a validated aircraft sender or controller handoff.

## Historical compatibility API

`planner.py` remains available to interpret separately held legacy force-policy
packages. `create_plan(package, initial_path, output)` constructs a simulated
plan from the package's frozen model/policy and explicit initial state;
`force_at_elapsed` reads its recorded force schedule. These APIs describe that
legacy artifact format, not the current PVA operator workflow.

Keep historical force units, packet holds and cutoff behavior unchanged. Do not
relabel a force command as acceleration/PVA or infer a vehicle interface from
those helpers. The optional `controller_acceleration` helper is only a numerical
conversion under its stated mass/gravity assumptions; it is not an aircraft
integration implementation.

The reference controller scripts are retained for compatibility and review.
They are not the executor for the current study. Use the actual lab program and
verified vehicle/interface details for hardware execution. No platform or
physical-flight validation follows from importing these modules.
