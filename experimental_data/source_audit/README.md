# Supplied experiment-source evidence

`experiment_logger.py` is the exact logger supplied on 2026-08-27, preserved
for command and provenance audit. SHA-256:

`f27cdcd9b34c15f5ef5b63a4dfb6ecb4586af4ff9e77be92cb776cc5e82c1b95`

It is evidence, not an imported runtime module. It establishes that the CSV
logger subscribes to `crazyflie_interfaces.msg.FullState`, copies
`pose.orientation` as xyzw, copies `twist.angular` without conversion, and
derives the CSV `cmd_yaw` from the quaternion using `atan2` (radians).

It does not contain the publisher/controller implementation. Consequently it
does not establish whether the publisher intended the quaternion as full
attitude or yaw-only, the angular-rate frame, the firmware controller, or the
estimator configuration. Those unresolved items continue to block parameter
fitting.
