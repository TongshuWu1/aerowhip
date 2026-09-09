Native 30 Hz PPO / FullState offline package

fullstate_30hz.csv is the exact complete reference: whip, braking, return, final hold.
The CSV uses unshifted OptiTrack tracked-origin coordinates, metres and seconds.
virtual_force_30hz.csv is simulator input INCLUDING gravity, not a drone force command.
checkpoints/policy.pt and assets/ contain the frozen PPO and both model residuals.
rehearsal.json records the desired start, target, feasibility and prediction limits.

To generate another OFFLINE plan with this code and a compatible Python/PyTorch CUDA environment:
python tools/rehearse_research.py --checkpoint checkpoints/policy.pt --output new_plan --origin X Y Z --target X Y Z --device cuda

Use the saved initial_tracking_origin_m values for X Y Z. Initial velocity is zero and the cable is assumed hanging.
Tested on Windows 11 / RTX 4080 / Python 3.12 / PyTorch 2.11.0+cu128.
No ROS sender or aircraft interface is implemented in this package. Recovery is not empirically validated.
