Take off; hold 10 s at saved start; execute complete CSV at saved timestamps; land. Offline artifact, no flight sender.

Desired OptiTrack tracked-origin P/V/A, kinematic acceleration, 30 Hz zero-order hold, no force or mass compensation

The complete CSV includes whip, recovery and final hold. Jerk CSV documents reference generation; it is not sent to the aircraft.
The saved NPZ is the original prediction for comparison with measured flights. Do not replace it with a later fitted model.
This package contains commands, frozen model assets and the originating run source. Use a compatible Python 3.12/PyTorch CUDA environment.
Replay offline: python tools/rehearse_pva.py --job . --output new_rehearsal
