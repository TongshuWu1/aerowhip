# M5 curved side whip — 1.3 m release

Use `fullstate_30hz.csv` from this folder. In the rehearsal UI, select **M5-Curved-Side-Whip-1p3m**.

The original maneuver has been lowered by 0.208229 m. The starting hover is **[0, 0, 1.791771] m**, and the simulated release point is **[-1.361136, 1.148685, 1.300000] m**. The target marker represents the selected release point; it is not a measured hit.

Independent replay confirms that the original command timing, velocity, acceleration, and whip shape are preserved. The 45.1-degree side turn, 80 ms release interval, and 5.94 m/s simulated tip speed are retained. This was a verified vertical translation of the saved MPPI plan, not a new optimization.

The complete 9.57 s command includes braking, return, and hold. Full drone-and-cable recovery passed the existing checks. The lowest predicted cable position is **0.784 m** above the floor. The horizontal footprint is unchanged: the cable reaches Y **-2.63 m** behind the starting position during recovery. Physical performance remains untested.

The strict comparison of the entire cable replay against a rigid vertical translation did **not** pass: the largest coordinate difference is 1.38 cm at 8.59 s, during late settling. Differences first exceed 0.1 micrometre at 6.33 s. The cause of that late divergence has not been established. The complete CSV translation agrees within 1.2e-15, and the whip cable prefix within 8e-13 m; the release height and motion requirements are verified. Both complete recovery trajectories pass the existing limits. The failed comparison and its original tolerance are retained in the verification JSON.

`height_translation.json` records the source plan. `height_translation_verification.json` records comparisons of the entire command and cable replay against the original. `side_whip_review.json` contains the motion metrics. The short GIF shows the whip; the rehearsal UI shows the complete recovery.
