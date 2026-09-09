# Hover calibration integration — unfinished

User confirmed the available signal is **four individual motor PWM values**, with battery voltage also available. The ROS topic/message fields and whether those PWM values are before or after motor mapping, saturation and voltage compensation have not been supplied. Do not average these values and assume they are equivalent to Mellinger's collective `control.thrust`.

`hover_calibration.py` currently implements and tests the algorithm for a verified **collective legacy command** whose mapping is `output = massThrust * projected nominal thrust`. It does not yet implement a four-motor PWM adapter. Stable-window rejection, a frozen scale applied to total XYZ thrust including gravity, and an absolute-deadline strike scheduler have three passing offline tests. These do not validate the motor mapping or ROS behavior.

`force_controller.py` is a preserved working copy of the uploaded version (2), with an unconditional pre-arm draft guard. Its original 157 g constant, old force sequence and gain-switch gaps are not corrected yet; it must not be used for flight. The user-provided Downloads file is untouched.

Needed to finish: actual four PWM field names, ROS topic/message or logging configuration, and the relevant firmware/motor mapping version. Inspect the mapping before deriving a correction from PWM. Battery voltage may be recorded; no continuously adapting battery model is required by the requested per-maneuver workflow. The intended correction is re-estimated during each stable hover and frozen throughout the next strike.
