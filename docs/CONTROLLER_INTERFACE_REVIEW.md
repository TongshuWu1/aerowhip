# Colleague's force-controller script review

**Historical force-only experiment. Do not apply its gain-disabling or force-to-
acceleration procedure to the current learned PVA workflow.** Current identification
models the loaded drone with its existing closed-loop controller. Read the
[current drone-command audit](DRONE_COMMAND_CHAIN_AUDIT.md) for the selected CSV
and remaining sender verification. The original reviewed script below is not
the verified sender for that experiment.

Reviewed September 6, 2026: `force_controller.py` supplied from Downloads.
The original file was not modified or run against ROS/hardware. This is a
Crazyswarm2/ROS 2 experiment script selecting onboard Mellinger controller ID 2,
not an implementation of the previously discussed custom Lee controller.
The flashed firmware revision and vehicle parameters still need confirmation.

## What it does

Arm `cf_7`, take off to 0.40 m, stream normal full-state hover, zero six position
P/D/I gains and two integral ranges, stream zero feedforward acceleration for
0.2 s, restore gains, recover hover, then request high-level landing. Attitude
feedback remains active. It does not load PPO, observe ten cable markers,
generate a force sequence, or log the flight for adaptation.

## Force conversion

The inspected upstream [Mellinger source](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/modules/src/controller/controller_mellinger.c)
constructs a desired world thrust vector from mass times acceleration plus
gravity and position feedback. With position feedback disabled, the required
mapping from our total world-force command is:

```text
a_ff = F_policy / m_controller - [0, 0, g_firmware]
```

Use the actual configured `ctrlMel.mass`, the flashed firmware's gravity
constant and a verified shared Z-up frame. This is a force-to-setpoint conversion,
not the measured aircraft acceleration. Do not subtract cable reaction from
the policy force or add hover support again. `cmdFullState` acceleration is in
m/s², not newtons. These conclusions are conditional on the flashed controller
matching the inspected implementation.

Our model's hanging support is approximately 1.717 N, including cable/marker
weight. With controller mass 0.159 kg, that corresponds to approximately
0.99 m/s² upward feedforward. Zero acceleration supplies only the controller's
configured mass times gravity once position feedback is disabled.

The firmware projects desired thrust onto the current body thrust axis and
uses a force-to-motor-command scale. Desired force therefore is not instantaneously
realized. Verify `ctrlMel.massThrust`, actuator response and achievable forces;
printing `ctrlMel.mass` alone does not establish correct force tracking.

## Findings in the supplied script

1. **Setpoint gaps during switching.** Each `set_params` sleeps 0.4 s without
   streaming. There are two consecutive calls both when disabling and restoring
   gains. An offline mock reproduced a maximum 0.82 s gap between full-state
   setpoints. The nominal 0.2 s test consequently does not bound time with
   position feedback disabled. Mode switching needs continuous setpoints and
   verified completion, not sleeps on the command-streaming path.
2. **Incomplete gain restoration.** Ctrl-C inside feedforward sends emergency
   stop and returns from `main`, leaving the gains at zero. An offline mock
   reproduced this for all six gains. `gains_modified` is also set only after
   both write batches, so a partial-write exception can evade restoration.
   A failure while restoring inside `except` can prevent the subsequent
   emergency call. Cleanup needs explicit partial-mutation tracking and
   protected restoration paths, without delaying an emergency motor stop.
3. **No verified parameter transition.** The script accepts nonfinite saved
   gains and only prints the mass. Upstream Crazyswarm2 parameter writes are
   asynchronous; reading its ROS parameter cache alone does not prove onboard
   acknowledgement. Validate values and verify the actual firmware transition.
4. **Height incompatible with the hanging cable.** The modeled attachment-to-tip
   length is 0.9525 m. A 0.40 m launch height cannot reproduce a freely hanging
   cable above a flat floor. It could be an unloaded controller test, but cannot
   serve as our cable initialization unchanged. The trained attachment height
   is 1.5 m and the tracking origin has a separate attachment offset.
5. **Timing is not our policy clock.** `RATE=50` controls stream publication;
   the policy requires one held action every 50 ms (20 Hz), with a frozen cutoff.
   Do not advance the policy index on every 50 Hz publish. A 50 Hz tick does not
   divide 50 ms exactly. Use deadlines that preserve command boundaries and log
   actual send times. Attitude control remains onboard throughout the maneuver.
6. **Landing handoff is not confirmed.** The script sends
   `notifySetpointsStop(remainValidMillisecs=200)` and `land` immediately through
   separate asynchronous services. Upstream documents the low-level validity
   interval; this file does not establish ordering, acknowledgement or actual
   transition completion. Verify the flashed stack rather than assuming its
   `No delay here` comment establishes correct behavior.

API reference: [Crazyswarm2 Python implementation](https://github.com/IMRCLab/crazyswarm2/blob/main/crazyflie_py/crazyflie_py/crazyflie.py).
Upstream source is a reference, not evidence of the version installed in the lab.

## Integration decision

The architecture can support our intended maneuver: estimate drone/cable state
during hover, plan once, freeze forces/cutoff, disable outer position feedback,
execute the force-derived acceleration setpoints once with attitude stabilization,
then restore position control. Actual cable/hit measurements can be logged during
execution without feeding the strike policy. Controller tracking errors belong
in the simulator/actuator validation before attributing flight mismatch to DDER.

First confirm the vehicle identity/mass, firmware revision, `ctrlMel.mass`,
`ctrlMel.massThrust`, coordinate frame and telemetry. Correct the transition and
cleanup behavior, then verify the bridge with offline mocks and appropriate
controller tests before connecting a learned strike. No PPO retraining is
required merely to convert force to acceleration; a meaningful measured plant
mismatch may require later simulation and policy adaptation.

Validation performed: Python syntax parsed; upstream controller/API inspected;
only extracted Python control flow executed with in-memory mock drone/time/ROS
objects. No ROS packages were imported by the mock, no vehicle connection was
made, and no arming, parameter or flight command was transmitted.
