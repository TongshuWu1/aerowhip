# Deployment verification

This file records software validation for the deployment branch, not new
experimental or aircraft results.

## Environment observed

Windows 11, Python 3.12.10, NVIDIA RTX 4080, PyTorch 2.11.0+cu128. Direct dependency
versions are in requirements/deployment-constraints.txt.

The environment checker passed dependency inspection, imported-baseline integrity
and a small CUDA float64 tensor operation. It did not run model identification,
MPPI search, PPO training or a flight.

## Completed software checks

On 11 September 2026, the integrated run passed **102 tests**, with **one skipped**
because its separately held research dataset is absent. This comprises 52
deployment tests and 50 passing existing regressions. The existing regression
suite includes tiny isolated CPU training smoke fixtures; no retained PPO/SAC
job was resumed and no new M0/M1/M2 experiment fit was run.

The suite covers baseline identity and import, portable source/private packaging,
setup arguments, fixed take roles, immutable raw imports, timing estimation,
review gates, failed-attempt retention, explicit retries, diagnostic/fitting
separation, continuous distances with missing-coverage flags, and GUI actions.
After fixing new source-index paths to use forward slashes, the affected 32
seed/backend tests passed again. The final 52 deployment tests also passed after
normalizing source text to the same line endings used by fresh Git checkouts.

A private application archive was extracted into a different folder containing
spaces. From that copy, verification passed for the complete release manifest,
retained baseline, study creation, exact M0 CSV export, and planner input
preparation using the original model, reference and proposal arrays. The
optimizer was not started.

Five existing M0 development recording pairs exercised import and review in
that isolated copy. Take 003 was explicitly excluded for insufficient causal
history. Native full-update preparation and CPU model construction passed with
001/002/004 as adaptation and 005 as validation. No fit was started. These are
software fixtures, not newly collected experimental results.

The guided UI was inspected at 1366×768. The retained M0 forecast and an imported
development flight loaded and closed cleanly. Offscreen flight inspection used
error plots; native VTK 3D rendering requires an interactive OpenGL surface and
was not established by this check.

The checks used the existing development environment; a fresh dependency install
was not performed. The Windows/Ubuntu CI workflow is included but has not run on
GitHub. The original 495 copied source/config files were rehashed and unchanged
in the research checkout.

Retained identities:

- M0 model signature: `fc854eae6a37bd8cd3457f1eb7eeb3ba52ae8e0b1315b8fdcc6534a5f94b2858`
- Command CSV SHA-256: `ea2e20bba92eb55cad12559ea97fb817eed76cd9ef6b21c191bf7b504368dde3`
- Original forecast SHA-256: `8ed2d231f333d567d71fd8871322e6a2d77b625a29ea2135c434553290b25e13`

## Git consolidation checks (11 September 2026)

The combined main/lab source passed 104 targeted software tests on Windows 11
with Python 3.12.10. These cover deployment, PVA export, model evaluation,
adaptation, flight comparison, GUI workflows, and source packaging using
isolated fixtures. Research configurations and retained evidence were kept at
their original Git versions. No experiment fitting or training was started.

The retired Isaac prototype separately passed its 12 CPU math tests. Its source
and history are retained without activating it as the paper simulator.

After synchronizing deployment and correcting release resources, the same
104-test subset passed again in the deployment checkout. The dependency,
retained-baseline and CUDA float64 checks also passed on Windows/RTX 4080.
All 10,874 retained research evidence files matched their saved hashes, and
documentation links were checked in both checkouts. Source/private release
builders now include the default processing configuration and label references
to separately held research assets instead of emitting broken links.

## Destination checks

Ubuntu/RTX 5080 has not been exercised by this Windows run. On the colleague's
machine, rebuild .venv, run tools/check_lab.py with --compute --require-cuda
--require-baseline, open the GUI, and verify the exported M0 package against its
manifest before the experiment. Local simulated fixtures cannot verify the
external sender, tracking clock or vehicle response.
