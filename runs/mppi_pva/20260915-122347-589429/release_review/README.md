# M5 horizontal-whip search review

The search completed, but this candidate does not yet meet the requested horizontal cable sweep. It remains a simulation diagnostic, not a flight release.

The target position was free. The final refinement used 512 samples for 40 iterations, initialized from a single M0-derived pulse with controlled descent. The M5 physical model and existing flight limits were retained. The optional objective rewards horizontal cable geometry while requiring loading, backward motion, and delayed velocity peaks along the cable.

At the selected release (1.1067 s), simulated tip speed is 6.981 m/s and its velocity is 0.85 degrees below horizontal. However, the last three cable segments are inclined 56.4, 61.6, and 62.1 degrees from horizontal. Cable height RMS relative to the attachment plane is 0.385 m, with 0.705 m total vertical span. Horizontal tip velocity therefore does not establish the requested sideways cable sweep.

The proximal, middle, and distal velocity-band peaks occur at 0.6933, 0.9067, and 1.0933 s. This retains the delayed velocity-peak pattern seen in the M0 reference. It is a kinematic proxy, not a measurement or proof of energy transfer. A separate flatter circular candidate was rejected because its motion was predominantly circling, with nearly simultaneous band peaks.

Independent replay with frozen job code reproduced the saved score and release. Replacing the placeholder target with the selected release point left the score, release time, and release position unchanged. No target accuracy or flight performance was measured. Full cable recovery was not evaluated for this rejected candidate, and no flight CSV or rehearsal was published. The active M4 flight export was left unchanged.

Related implementation checks: 82 passed, 2 skipped. Test command: `.venv/Scripts/python.exe -m pytest tests/training/test_free_whip.py tests/ui/test_free_whip_display.py tests/training/test_horizontal_cable_objective.py tests/training/test_strike_objective.py tests/flight/test_position_spline.py tests/flight/test_fullstate_recovery.py tests/flight/test_curved_recovery.py tests/flight/test_gentle_recovery.py tests/ui/test_ppo_policy_rehearsal.py -q -p no:cacheprovider`.

Review files: `side-whip-preview.gif` (motion), `motion-review.png` (geometry), `release-comparison.png` (M0 and candidate velocity histories), and `motion.npz` (replayed cable positions and velocities).
