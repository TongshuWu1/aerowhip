# M5 forward-left horizontal whip

Simulation rehearsal: accelerate forward (+X), then release the cable tip toward the left (+Y). The release is selected from the motion; target accuracy and physical contact have not been measured.

At release (1.707 s), the whole-cable height RMS relative to the attachment plane is 3.04 cm, and the cable spans 13.36 cm vertically. The final segment still inclines downward by 42.4 degrees. Tip speed is 5.17 m/s, with velocity elevation -0.72 degrees. This describes a mostly horizontal cable with a curved tip, not a perfectly horizontal cable.

Initial tracked-origin hover: [0, 0, 1.2884] m. Selected release: [3.0947, 0.9717, 1.4500] m. Forward loading, 103.9-degree loading-to-release turn, backward pullback, and an 80 ms aligned release were independently checked. Ordered cable-group velocity peaks are a kinematic proxy, not a measurement of energy transfer.

The search used M5, 512 samples and 80 iterations with a soft whole-cable horizontal penalty. The saved command was then translated downward by 0.911624 m without changing jerk or timing. Independent replay matched the translated whip geometry within 2.3e-12 m and velocity within 3.3e-11 m/s. Full recovery was independently simulated and passed the existing limits and frozen-prefix checks.

The complete CSV lasts 14.667 s including braking, return, and settling. Simulated drone bounds: X [-0.113, 4.924], Y [-1.372, 0.641], Z [1.026, 1.620] m. Cable bounds: X [-0.436, 5.800], Y [-2.091, 1.292], Z [0.177, 1.587] m. The large recovery footprint needs inspection against the actual available flight space. This rehearsal does not establish physical-flight safety.

Release height was kept at 1.45 m because lowering the same motion another 0.15 m would reduce minimum drone height to about 0.876 m, below the configured 0.96 m limit. Existing limits were not relaxed.

Files: fullstate_30hz.csv, rehearsal.npz, motion-review.png, side-whip-review.png, motion_review.json, side_whip_review.json, horizontal_review.json. The active flight selection was not changed.
