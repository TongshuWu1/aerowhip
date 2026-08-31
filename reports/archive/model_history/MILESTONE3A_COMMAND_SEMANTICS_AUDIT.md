# Milestone 3A — FullState Command-Semantics Audit

> **SUPERSEDED BLOCKER NOTICE (2026-08-27):** The flight stack is on an
> external computer and was never expected to be present in this repository.
> The user has confirmed stock/unmodified firmware, default configuration, and
> a medium vehicle.  More importantly, all four physical takes directly show
> yaw-only command quaternions, zero angular-rate commands, nonzero horizontal
> acceleration commands, and large measured roll/pitch.  The effective model
> has therefore been corrected using the recorded input-output evidence and
> Stage A is no longer blocked.  This document is retained as provenance for
> the earlier conservative stop; current results are in
> `reports/MILESTONE3A_STAGE_A_REPORT.md`.
>
> **2026-08-28 amendment:** The standard `cmdFullState` contract is now
> recorded explicitly: position/velocity/acceleration are world-frame SI,
> yaw is radians, the command quaternion is yaw-only, and omega is body-frame
> rad/s. The current four takes have identically zero omega commands, so this
> clarification changes metadata but none of the completed Stage A numerical
> results. The exact stock firmware version and external default configuration
> files still need to be archived when the flight computer becomes available.

Date: 2026-08-27 (America/New_York)  
Repository: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`

## Executive result

Milestone 3A is **blocked before model modification and fitting**. The supplied
source is the experiment logger, which subscribes to FullState messages; it is
not the publisher that constructs those messages. No FullState publisher,
flight script, launch configuration, Crazyflie firmware/controller selection,
or estimator configuration is present in the repository or other supplied
local experiment files.

The available evidence is sufficient to establish that fitting the current
`FullStateUAVModel` literally against `cmd_q` would be scientifically invalid:
all four takes have exactly zero command-quaternion x/y components and zero
recorded angular-rate command, while the measured vehicle executes substantial
roll/pitch motion under nonzero horizontal acceleration commands. It is not
sufficient to derive the exact controller transformation from acceleration and
yaw into desired attitude.

Accordingly:

- raw processing, processed schemas, synchronization, quality masks, markers,
  annotations, windows, DDER, cable geometry, EI, Cb, and dataset GUI were not
  modified;
- `FullStateUAVModel` was not changed;
- no take roles were assigned;
- no Sobol or gradient optimization was run;
- no Stage A parameters or validation claims were produced;
- Stage B and Stage C remain locked.

This is the required safe outcome under the instruction not to fit `K_R` and
`K_omega` against an unverified attitude-command interpretation.

## 1. Source discovery

The audit searched the repository and locally supplied project/experiment
locations for:

```text
FullState
cmd_full_state
cmdFullState
send_full_state_setpoint
sendFullState
fullState
stabilizer.controller
Mellinger
PID
Lee
estimator
Kalman
```

File types included Python, C/C++, ROS message/launch files, YAML, TOML, and
XML. Locations included the current repository, Documents projects, Downloads,
Desktop, OneDrive, temporary experiment files, and local PyCharm projects. No
active WSL/ROS environment or Docker CLI containing the `/workspace` runtime
was available on this host.

The only experiment-specific FullState source found was:

```text
experimental_data/source_audit/experiment_logger.py
```

SHA-256:

```text
f27cdcd9b34c15f5ef5b63a4dfb6ecb4586af4ff9e77be92cb776cc5e82c1b95
```

This file is the subscriber/logger. It contains no command-generation or
firmware-controller logic.

## 2. What the logger proves

The logger imports:

```python
from crazyflie_interfaces.msg import FullState
```

and subscribes to:

```text
/<tracking.drone_name>/cmd_full_state
```

unless another topic is passed by command-line option.

It records the FullState message as follows:

| CSV field | Source in FullState message | Logger transformation |
|---|---|---|
| `cmd_x/y/z` | `pose.position` | float copy |
| `cmd_vx/vy/vz` | `twist.linear` | float copy |
| `cmd_ax/ay/az` | `acc` | float copy |
| `cmd_qx/qy/qz/qw` | `pose.orientation` | float copy, xyzw |
| `cmd_omega_x/y/z` | `twist.angular` | float copy |
| `cmd_yaw` | `pose.orientation` | logger-computed quaternion yaw |

`cmd_yaw` is therefore not an independent command. The logger calculates:

```text
yaw = atan2(
    2 (qw qz + qx qy),
    1 - 2 (qy^2 + qz^2)
)
```

and therefore stores radians.

The logger selects the newest cached FullState message whose header timestamp
is no later than the consumed NatNet frame timestamp. Its default command-age
timeout is 0.2 s. This agrees with the Milestone 3 causal zero-order-hold
reconstruction.

The logger copies `twist.angular` without conversion. ROS convention suggests
radians per second, but the actual publisher source is required to verify that
the producer obeyed this convention. The logger does not save the command
header frame, so the angular-rate frame cannot be reconstructed from the CSV.

## 3. What remains unknown

The following required semantics are not contained in the logger or datasets:

- source file and method that publishes `/cf_7/cmd_full_state`;
- exact Crazyswarm/Crazyswarm2 version or commit;
- exact Crazyflie ROS API callback and conversion path;
- Crazyflie firmware version or commit;
- controller type actually selected in firmware;
- estimator type and configuration;
- whether `pose.position` and `twist.linear` are references, feedforward
  quantities, or combined differently by the selected controller;
- whether `acc` is translational acceleration, acceleration feedforward,
  gravity-inclusive specific force, or transformed before control;
- whether `pose.orientation` is full desired attitude or a yaw-only container;
- whether `twist.angular` is body-frame or world-frame desired rate;
- publisher-confirmed angular-rate units.

No source-backed formula is therefore available for converting:

```text
p_cmd, v_cmd, a_cmd, yaw_cmd
```

into desired thrust direction and full desired attitude.

## 4. Four-take data consistency diagnostic

No physical model was optimized. The diagnostic below only summarizes the
recorded commands and measured Motive attitude on command-valid frames.

| Take | q_cmd xy norm max | omega abs max | horizontal a_cmd RMS | horizontal a_cmd max | measured roll range | measured pitch range |
|---|---:|---:|---:|---:|---:|---:|
| fig8_001 | 0 | 0 | 0.740 m/s^2 | 1.997 m/s^2 | -11.00 to +11.88 deg | -12.10 to +1.88 deg |
| fig8_002 | 0 | 0 | 1.641 m/s^2 | 4.494 m/s^2 | -22.12 to +24.31 deg | -15.16 to +9.01 deg |
| fig8_003 | 0 | 0 | 3.640 m/s^2 | 9.024 m/s^2 | -50.97 to +47.69 deg | -22.45 to +16.51 deg |
| osc_001 | 0 | 0 | 2.498 m/s^2 | 4.285 m/s^2 | -1.24 to +2.61 deg | -30.55 to +18.04 deg |

The recorded quaternion contains only a small constant z component plus w in
each take. Its logger-derived yaw is also constant within each take:

| Take | logged/derived yaw |
|---|---:|
| fig8_001 | -0.002219 rad |
| fig8_002 | -0.002565 rad |
| fig8_003 | -0.000745 rad |
| osc_001 | -0.002040 rad |

This is inconsistent with interpreting `cmd_q` as the complete desired
roll/pitch/yaw trajectory supplied to the current attitude tracker. It is
consistent with—but does not prove—a stack in which horizontal acceleration
and yaw are converted internally into desired thrust/body-z direction and
attitude.

The requested desired-roll/pitch RMSE, correlation, and systematic-delay study
cannot be computed honestly yet. Doing so requires the actual controller
formula and frame conventions. Choosing a common acceleration-plus-gravity
formula without the publisher/controller source would violate the milestone's
instruction not to assume that formula.

## 5. Current model conflict

The current minimal model uses:

```text
a = k_a a_cmd + K_p (p_cmd - p) + K_v (v_cmd - v)

orientation_error = LogLike(q_cmd * inverse(q))

omega_dot = K_R orientation_error
            + K_omega (omega_cmd - omega)
```

where `q_cmd` is interpreted as complete desired body-to-world orientation and
`omega_cmd` as desired world-frame angular velocity.

Applied literally to these recordings, the attitude portion would drive the
vehicle toward nearly level constant-yaw attitude while the real vehicle
tilts substantially. Fitting `K_R` and `K_omega` cannot repair this structural
semantic mismatch without making their values misleading.

The likely minimal correction is an adapter that derives desired thrust/body-z
from the same translational terms used by the real stack and combines it with
commanded yaw before applying effective attitude dynamics. That correction has
not been implemented because the exact source-derived formula, frames, gravity
handling, saturation, and controller semantics are absent.

## 6. Stage A status

Stage A was not enabled.

The following outputs therefore do not exist:

- fitted `K_p`, `K_v`, `k_a`, `K_R`, or `K_omega`;
- Sobol candidate results;
- gradient-refinement results;
- training position/orientation RMSE;
- held-out validation metrics;
- roll/pitch/yaw prediction errors;
- lead-time errors;
- bound-hit diagnostics;
- delay estimate based on a verified input-output model.

No EI or Cb value was changed or optimized. No cable error entered any UAV
objective.

## 7. Dataset status

The four complete takes remain disabled/Ignore. A tentative split such as
`osc_001`, `fig8_001`, `fig8_002` for training and `fig8_003` for validation
was not committed because model structure is not yet resolved. In particular,
`fig8_003` contains automatically masked geometry/marker-quality regions that
must remain excluded, although it otherwise provides the strongest motion.

When the semantic gate is opened, the whole-take split can be frozen before
any optimization. The validation take must not be used to select model form,
bounds, optimizer settings, or gains.

## 8. GUI status

The existing three-tab GUI is unchanged. Fit remains disabled through the
scientific gate. No cable-fitting option was enabled. Stage B and Stage C
remain unavailable.

After the publisher audit is resolved, the Fit & Validate tab should be changed
to show Stage A — UAV only and measured/simulated UAV overlays. That work was
not performed prematurely because there is no valid Stage A command adapter to
execute.

## 9. Exact artifacts required to continue

Provide the files/configuration from the environment that ran the experiment:

1. the trajectory or flight script that calls/publishes FullState;
2. any wrapper implementing `cmdFullState` or publishing `cmd_full_state`;
3. ROS launch files and parameter YAML used for `cf_7`;
4. Crazyswarm/Crazyswarm2 repository commit, release, or container image tag;
5. Crazyflie firmware commit/version;
6. controller selection and controller parameter dump;
7. estimator selection and estimator parameter dump;
8. confirmation of coordinate frames and units used by the publisher.

The most important missing artifact is the FullState **publisher/flight
script**, not another copy of the logger.

Once supplied, the next valid order is:

```text
publisher -> ROS/Crazyflie callback -> firmware controller semantics
    -> source-derived desired-attitude diagnostic on all four takes
    -> minimal command-adapter correction
    -> freeze whole-take split
    -> Stage A only
    -> held-out validation
    -> STOP
```

Until then, proceeding to Stage A would create attractive but scientifically
uninterpretable UAV gains.
