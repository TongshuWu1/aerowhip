# Incomplete artifact export

The deterministic Stage A optimization and metrics completed, but the first
artifact write stopped while exporting the mixed Sobol/Adam CSV trace because
the two row types did not share one optional column.  The frozen optimization
was rerun unchanged after correcting only that CSV schema mismatch.

Authoritative result:

`../2026-08-28T035156.958739+0000_a0319d7f/`

The authoritative rerun reproduced the fitted parameters, best objective, and
all training/validation metrics exactly and contains the complete 32-row Sobol
plus 100-row Adam trace.
