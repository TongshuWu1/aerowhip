# Retired learning source

This directory preserves inactive research implementations for provenance.
They are deliberately outside the active `learning` package and pytest path.

- `runners/`: SAC, deterministic amortization, diffusion/scorer, and residual
  policy milestone entry points.
- `modules/`: their implementation modules.
- `config/`: exact historical experiment configurations.
- `tests/`: historical branch tests.

These files are source snapshots, not supported entry points. Restoring a
historical experiment requires recreating its original relative layout and
using the immutable artifact/source-hash manifest from that milestone. Do not
import these modules into production code.
