# M5 curved side whip

Open **M5-Curved-Side-Whip** in the rehearsal UI. `fullstate_30hz.csv` contains the complete 30 Hz command, including braking, return, and hold. Start at the saved hover position **[0, 0, 2.0] m**. The maneuver lasts 1.53 s and the complete CSV lasts 9.57 s. This is an offline simulation result; physical performance has not been measured.

The simulated release occurs at 1.293 s, after a 45.1-degree change from the measured loading direction. Tip speed is 5.94 m/s, with its velocity 7.2 degrees above horizontal. The tip remains inside the configured strike cone at the required speed for 80 ms. Proximal, middle, and distal velocity-band peaks occur at 0.913, 1.107, and 1.267 s. These are kinematic checks of a loading-and-release pulse, not a measurement of energy transfer. The cable is allowed to remain inclined.

The displayed release point, **[-1.361, 1.149, 1.508] m**, was selected from the optimized motion. It is not a measured hit or evidence of target accuracy.

Full recovery was simulated and passed the saved limits. Over the complete sequence, the predicted cable occupies X **[-1.37, 0.62] m**, Y **[-2.63, 1.46] m**, and Z **[0.99, 2.54] m** in the saved coordinate frame. In particular, the return extends substantially behind the starting Y position. Use the complete rehearsal to inspect that motion; the short GIF shows only the whip.

The final search used 512 samples over 80 iterations. Independent replay reproduced the saved score and release, and moving the display target to the selected release point did not change them. The exported command also passed the existing drone and cable prefix-consistency checks. Related tests: 85 passed, 2 skipped; the final rehearsal was additionally loaded successfully in the UI offscreen.

Review files: `side-whip-preview.gif`, `side-whip-review.png`, and `side_whip_review.json`. Full predicted positions and velocities are in `rehearsal.npz`. Model assets and the exact planner job are recorded in `rehearsal.json`.
