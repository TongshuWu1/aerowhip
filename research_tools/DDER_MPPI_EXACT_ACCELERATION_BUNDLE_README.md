# DDER–MPPI exact acceleration delivery bundle

Start with `DDER_MPPI_EXACT_ACCELERATION_REPORT.md`.

The bundle contains:

* `report/` — the complete research/engineering report and original forward
  profile report;
* `profiles/` — staged 20-update timing JSON, final exclusive kernel profile,
  original profile, and unsuccessful-trial measurements;
* `validation/` — frozen reference bundle, final numerical/ranking validation,
  and paired closed-loop reference/optimized results;
* `source/` — the exact implementation and reproduction tools;
* `tests/TEST_RESULTS.txt` — final compile and unit-test result.

Important outcome: mean warmed planning improved from 589.90 ms to 82.84 ms
(7.12×). The strict 70–80 ms practical planning target was not reached; final
p95 is 85.55 ms.

