These two bounded integration jobs were moved out of the user policy library after their workers exited. Frozen source and logs are retained as audit evidence, including their original absolute paths. Do not resume these relocated test runs.

seed901: first configuration incorrectly disabled validation while plateau stopping remained enabled; failed with the expected validation requirement. Its stalled shutdown was terminated.

seed902: disabled plateau stopping, used four-case validation and disabled the update guard for this bounded test. Completed two batches / eight attempts. Launch config was amended after preparation for this test; the original preparation metadata hashes therefore do not represent those test overrides. Final production launcher snapshots are covered separately by immutable-snapshot tests.
