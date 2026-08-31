# Shared cable pipeline

This package contains the code used by both offline identification and online
tracking:

- ZED acquisition and camera calibration
- canonical PIDNet runtime
- complete image-plane routes for offline identification
- partial ordered routes and registered-depth lifting for online tracking
- cable observation and trajectory schemas
- differentiable constrained rod equations with physical `EI` and `Cb`
- fitted model loader
- asynchronous 3D point-cloud viewport

The shared package contains no offline optimizer and no particle filter.
