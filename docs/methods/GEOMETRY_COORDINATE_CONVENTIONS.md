# Geometry and coordinate conventions

Geometry-only revision, 8 September 2026 UTC. No model fitting, residual training, PPO training, rate change, or firmware modification. The user confirmed that no position offset is applied to the OptiTrack input on the flight computer. The controller therefore receives the top rigid-body origin as its external position measurement; its estimator still fuses measurements and does not equal the raw measurement at every instant.

## Points and frames

| Symbol | Meaning | Status |
|---|---|---|
| W | Recording/simulation world, meters, right-handed Z-up | Existing data use the same global XYZ convention; no new axis transform applied |
| T | Local frame of OptiTrack rigid body `cf_7` | Pose quaternion is XYZW and maps T vectors into W |
| O | `cf_7` rigid-body origin on the top marker plane | Position reference supplied to the controller, by user confirmation |
| A | Cable attachment underneath the battery | Rigidly displaced from O |
| C1 | First marker on the flexible cable | Separate from A; not a rigid drone point |
| COM | Physical center of mass | Location is unknown; do not identify it with O or A |
| B | Firmware/IMU body frame | Do not infer a T-to-B rotation from the attachment height; inspect actual setup before using firmware angular data |

Reported ruler dimensions are approximately **55 mm vertically from the top reference plane to A**, then **63 mm of cable arc length from A to C1**. These dimensions were reported during the September 2026 calibration work; the superseded calibration note is no longer distributed in this checkout. They were not independently remeasured in this revision. A 63 mm arc can have a shorter straight-line chord when bent. Neither distance includes the other. Marker-center versus cable-centerline placement remains a measurement limitation.

Preserved active offset O-to-A expressed in T: `[0.0066549972854827175, -0.01287427254333901, -0.055]` m.

Preserved historical complete-model candidate offset: `[0.007084866324004131, -0.014136315175398022, -0.055]` m. The lateral values are fitted estimates; the 55 mm vertical distance was fixed. This revision selects or applies neither geometry version. Each saved model supplies its own offset.

The legacy JSON key `optitrack_to_attachment_offset_body_m` is retained for snapshot compatibility. Its vector is interpreted in **T, the tracked rigid-body frame**, not automatically the firmware body frame.

The cable has 12 simulation nodes: A at node 0, an interpolated midpoint at node 1, C1 at node 2, through C10 at node 11. The first flexible span is subdivided into two 31.5 mm rest-length edges. The midpoint is not another measured marker. No cable-clamping or dynamics assumption was changed.

## Coordinate operations

Let R be the active T-to-W rotation and r the constant O-to-A vector in T:

```
p_A = p_O + R r
v_A = v_O + R (omega_T × r)
a_A = a_O + R (alpha_T × r + omega_T × (omega_T × r))
```

Angular velocity and angular acceleration must be expressed in T for these equations. R must belong to the same timestamp and trajectory as p_O. Future measured orientation is legitimate when reconstructing measured attachment motion, but cannot be inserted into a claimed forward prediction from initial state alone.

`simulator/geometry.py` provides normalized XYZW rotations, attachment positions, complete rigid-offset P/V/A, and an explicitly initial-only world offset. Invalid quaternions remain invalid/NaN, including all-zero quaternions; they are never interpreted as identity. The physics fit adapter and preprocessing quality checks share the normalized rotation operation.

No NN is used for these geometric operations. How the future drone orientation should be modeled is deliberately left for the next model-design discussion.

## Full-state export boundary

The current virtual planner has no attitude trajectory. Existing full-model exports therefore remain an explicitly named **fixed initial translation**:

```
p_command = p_virtual_attachment - R_initial r
v_command = v_virtual_attachment
a_command = a_virtual_attachment
```

This generates a reference for O. It does not claim that the actual attachment will trace the virtual attachment path during rotation. The unchanged V/A values are mathematically consistent with a constant translation; a dynamic rigid-offset conversion would require R(t), omega(t), and alpha(t).

New export calls can provide r in T and an initial XYZW orientation; the offset is rotated before translation. The desktop rehearsal explicitly records its nominal identity initial orientation, since it does not measure a real flight pose. Export metadata labels this assumption and states that dynamic rigid-body conversion is false. Calls supplying a world offset directly remain supported and are no longer mislabeled as necessarily level hover. Existing CSVs were not rewritten.

The UI now labels start coordinates **Attachment hover XYZ**. For example, an attachment at world Z=1.500 m with the nominal level orientation and r_z=-0.055 m corresponds to tracked-origin Z=1.555 m. Lateral components also require the corresponding shift. Target XYZ remains a world point, so the drone attachment offset is never subtracted from the target.

Historical exports labelled `simulated_cable_attachment` retain that explicit reference point; they must not be confused with exports labelled `OptiTrack_cf7_origin`. Converting those historical artifacts or redesigning the planner reference is outside this geometry revision.

## Firmware and tracking path checked

The reviewed upstream revisions are Crazyflie firmware `fe0f5b0ee1c5b9c5e21cb47ad3ae593ddaa86cf7`, Crazyswarm2 `fcf51e27fe675bbdb564a40f4ca1ee4336c062ea`, and the ROS 2 motion-capture package `64d3af2456e534cd5e587b98ef2f15dd8ad35e8c`. These are source-review revisions, not a claim about the installed flight firmware.

- External position/pose handlers copy incoming XYZ to estimator measurements. The Kalman position/pose updates compare these directly with position state; this path has no top-marker-to-COM or top-marker-to-cable offset. [Localization handler](https://github.com/bitcraze/crazyflie-firmware/blob/fe0f5b0ee1c5b9c5e21cb47ad3ae593ddaa86cf7/src/modules/src/crtp_localization_service.c), [pose measurement update](https://github.com/bitcraze/crazyflie-firmware/blob/fe0f5b0ee1c5b9c5e21cb47ad3ae593ddaa86cf7/src/modules/src/kalman_core/mm_pose.c).
- Full-state packets supply absolute XYZ and corresponding V/A. The Mellinger implementation compares reference and estimated P/V, adds acceleration feedforward and gravity, and constructs the desired thrust direction. It has no knowledge of the cable attachment geometry. [Full-state decoder](https://github.com/bitcraze/crazyflie-firmware/blob/fe0f5b0ee1c5b9c5e21cb47ad3ae593ddaa86cf7/src/modules/src/crtp_commander_generic.c), [Mellinger implementation](https://github.com/bitcraze/crazyflie-firmware/blob/fe0f5b0ee1c5b9c5e21cb47ad3ae593ddaa86cf7/src/modules/src/controller/controller_mellinger.c).
- The Crazyswarm2 servers forward received mocap position XYZ and quaternion XYZW directly to external-position/pose packets. The motion-capture ROS 2 vendor path forwards the vendor rigid-body pose; configurable marker-layout offsets belong to the separately constructed rigid-body tracker and are not a universal offset applied by firmware. [Server](https://github.com/IMRCLab/crazyswarm2/blob/fcf51e27fe675bbdb564a40f4ca1ee4336c062ea/crazyflie_server_cpp/src/crazyflie_server.cpp), [ROS 2 tracking node](https://github.com/IMRCLab/motion_capture_tracking/blob/64d3af2456e534cd5e587b98ef2f15dd8ad35e8c/motion_capture_tracking/src/motion_capture_tracking_node.cpp).

The supplied `Downloads/full_state_pva.py` passes its PVA to cmdFullState without an offset. The user has now also confirmed no input offset on the flight side. Firmware filtering, aircraft body axes, and the actual installed revision remain separate questions for model design; none justifies inventing a geometric correction here.

## Verification and data audit

`runs/audits/20260908-034535-geometry-conventions/` contains the descriptive audit of all 11 recordings, input hashes, and archived upstream sources. No optimization or reprocessing was performed. Both geometry versions were evaluated without selecting new parameters.

Reconstructing each recorded drone marker in T using R transpose gives less constellation spread than the inverse interpretation in every take. This supports the existing XYZW rotation direction. It does not establish the physical location of COM or the firmware axes. Historical candidate first-span median chords are 58.25–61.53 mm across takes, consistent with a flexible 63 mm arc at the level of median measurements. Some frames still exceed arc length plus 2 mm; those remain visible in the audit and were neither erased nor used to change geometry.

The preprocessing quality rule now flags excess chord length rather than short chords caused by bending. Invalid pose has its own mask. Future processing fingerprints include the geometry values and shared geometry implementation, preventing a geometry change from silently reusing stale quality masks. Existing processed datasets and historical fits remain unchanged.

Thirty targeted tests passed on Windows 11 using CPU calculations, covering rotation direction/sign/normalization, invalid quaternion handling, independently differentiated rotating-offset P/V/A, bent-span quality handling, geometry-sensitive cache identity, export conversion and metadata, historical geometry adapters, and Full-state UI behavior. An outdated test was corrected to test an explicitly protected temporary manifest instead of assuming the currently authorized fig8vertical_002 is still protected. No hardware, flight controller, GPU runtime, model fit, or policy training was tested or started by this revision.
