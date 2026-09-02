# Simple sequential SAC 10-second baseline

This is one conventional sequential SAC run. The 6-D action is 3-D yaw-local acceleration plus body roll/pitch/yaw rates. The reward has only endpoint progress, near-target speed/direction, UAV displacement, and a first-success bonus. An episode lasts 10.0 seconds unless a row becomes numerically non-finite. `training_log.csv` reports episode throughput and endpoint success rate. No CEM, demonstrations, protected data, or hardware are used.
