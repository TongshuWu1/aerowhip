# M1 brief physical performance check

Five clean M1 takes; operator confirmed no cable contact, abort, or intervention.
Every take contains all 124 dynamic packets from the exact corrected CSV.
Tip tracking is complete over the strike interval. Raw logs were copied without
changing bytes; occasional gaps outside the strike interval remain preserved.

| Physical metric | Original M0 flights | Corrected M1 flights |
|---|---:|---:|
| Target error at original strike time | 24.12 cm | 11.18 cm |
| Closest target error, 0–1.5 s | 14.73 cm | 7.83 cm |
| Tip RMSE to fixed original M0 motion | 18.09 cm | 16.06 cm |
| Quadrotor RMSE to fixed original M0 motion | 15.66 cm | 11.17 cm |

Values are equal-take means. Fixed-time target error mean ± sample SD:
M0 24.12 ± 6.94 cm; M1 11.18 ± 4.04 cm. Reference tracking uses [0, 34/30] s;
the original strike time remains 1.1172482457473654 s. The comparison does not
shift or scale trajectories. Clock offsets were estimated from measured
streams, so timestamp uncertainty remains. These are separate flight batches,
not paired identical disturbances or proof of generalization across targets.

For M2, takes 001/002/004 are adaptation and 003/005 are validation. The fixed
original M0 reference is retained. M1 target accuracy improved substantially;
whole tip-trajectory improvement is smaller, as the table shows.
