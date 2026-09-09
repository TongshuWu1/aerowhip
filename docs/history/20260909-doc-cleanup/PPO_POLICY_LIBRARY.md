> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# PPO runs and policies

The SAC page has been removed. Existing SAC artifacts remain saved.

## Start a new policy

Open **PPO → Training**, enter **New run name**, set the seed, total target attempts, batch and device, and click **Train new policy**. This starts a fresh policy using the active project model/task configuration. The name is saved with the run; its unique timestamp directory prevents name collisions.

## Continue an existing policy

Open **PPO → Policies**, select the exact checkpoint, and click **Continue training…**. This stages the selection in Training; it does not start a job. Enter the new run name and a total attempt target greater than the checkpoint’s completed count, then click **Continue selected checkpoint**. The source run and checkpoint stay intact. Continuation inherits saved model/task and the existing trainer’s configured optimizer-resume behavior. It does not automatically apply a newly fitted baseline to an old policy.

## Use a policy for full-state export

Select a checkpoint in Policies and click **Use in Full-state 30 Hz**. The Full-state page opens with that exact checkpoint selected. Rehearsal and export still require their existing buttons. A running rehearsal must finish or stop before its selection can change.

## Delete or restore

**Delete from library** hides the selected checkpoint from Policies and the Full-state list. Files are preserved so references and experiment history remain valid. Enable **Show deleted**, select the checkpoint, and click **Restore** to bring it back. Checkpoints currently selected for Full-state/continuation, or belonging to an active training run, cannot be removed from the library.

This changes library visibility, not disk usage. No original checkpoints, measurements, training snapshots, physics settings or CSV execution semantics are changed by library management.
