# Two-Cable 3D Reconstruction Pipeline

## Scope

This document describes the code as it is currently implemented. It separates
active tracking measurements from diagnostics and from planned work.

The system reconstructs two visually identical deformable cables using one ZED
RGB-D camera. The neural network observes a shared cable body, two endpoint
groups. Cable identity, projected junction topology, and 3D shape are estimated
outside the neural network.

The canonical live and collection mode is native ZED HD1080 at 30 FPS. PIDNet
receives the full 1920 x 1080 RGB frame, and its training default is the same
resolution; the observation path does not silently downsample frames to 720p.
Pixel distances are scaled by 1.5 and pixel-area rejection thresholds by 2.25
from their former HD720 values. All metric geometry and dynamics remain
resolution-independent and unchanged.

## Runtime data flow

```text
ZED RGB + registered depth
        |
        +--> PIDNet: cable body + endpoint group 1 + endpoint group 2
        |           |
        |           v
        |    2D thinning and compressed skeleton graph
        |           |
        |           +--> endpoint-identified complete graph routes
        |           |
        |           +--> all directly observed ordered 3D graph edges
        |           |
        |           v
        |    Two batched but statistically independent CUDA particle filters
        |           |
        |           +--> complete-route initialization / route-centred retracking
        |           |
        |           +--> prior-based ordered graph-edge attribution
        |           |
        |           +--> visible-edge motion transport and particle likelihood
        |
        +--> yellow mask + registered depth + known 150 mm cube fit
                    |
                    +--> raw plane pose (preserved)
                    +--> robust joint pose refinement and covariance
                    +--> temporal rigid pose and velocity state
        |                       |
        +-----------------------+
                    |
                    v
        Passive CUDA cable--cube contact observer
        physical gap + normal motion + optional co-motion
        binary temporal posterior for each cable
        |
        v
Cable estimates, cube state, passive contact posteriors, diagnostics, and viewer
```

There is one PIDNet inference and one batched PF computation per tracked frame.
The two PF populations have separate weights and posterior estimates. The cable
batch dimension lets CUDA evaluate them together without coupling their
probabilities.

The raw cube observation uses the exact same immutable RGB-depth frame as the
cable tracker. Its serial CPU worker runs concurrently with the CUDA cable
path. After the results rejoin, a small causal SE(3) filter updates on the
tracking thread using the exact capture timestamp. Cube evidence does not yet
alter cable particles, weights, proposals, constraints, or resampling.

After both visual estimators finish, a read-only CUDA contact observer evaluates
the cable particle populations against the temporal cube state. This observer
does not modify either estimator. Contact feedback and force estimation remain
separate later stages.

## Information provided by the neural network

PIDNet returns three independent sigmoid channels:

1. **Cable body:** all visible cable pixels, without cable identity.
2. **Endpoint group 1:** both endpoints belonging to physical cable 1.
3. **Endpoint group 2:** both endpoints belonging to physical cable 2.

The endpoint groups are the direct cable-identity measurements. Endpoint
components are associated temporally within their own group so the two ends do
not exchange start/end labels merely because their image ordering changes.

The deployed checkpoint was migrated from the former four-channel model by
removing only the obsolete output row. The cable and two endpoint logits are
bit-for-bit unchanged by that migration. Future training runs use the
three-channel annotation and checkpoint schemas directly.

The collection GUI and main tracker share one runtime mask contract in
`NN_collection_training/source/config.toml`. The current thresholds are
0.85/0.90/0.75 for body/endpoint-1/endpoint-2. The body mask then receives a
7-pixel close and removal of components below 180 pixels; endpoint masks are
left exactly as thresholded. One shared implementation produces these masks
for GUI preview/evaluation, editable prediction drafts, the cable observation,
cube exclusion, and visualization. Loading a checkpoint never changes the
active thresholds. `Calibrate on Validation` searches the verified validation
split, and `Save Runtime to config.toml` explicitly publishes those displayed
thresholds to the shared runtime file. The main process reads that file at
startup and prints its path, checkpoint SHA-256, thresholds, and cleanup values.

## RGB-D geometry

The binary semantic masks needed for connected components and skeletonization
are transferred to the CPU. Only the registered-depth samples selected by the
current semantic masks are gathered and uploaded to CUDA; the observation path
does not scan or upload an otherwise unused full HD1080 depth tensor.

The result is a compact 3D array and pixel-to-point index image. Full
XYZRGBA point-cloud construction is not part of tracking. The viewer constructs
its point cloud independently and can use a display-only stride.

At native HD1080, observation preprocessing reuses the cleaned `uint8` masks
instead of creating three full-frame Boolean copies. Connected-component
statistics provide each exact cable bounding box, so thinning materializes only
the small component crop rather than rescanning the full label image for every
component. When two separate body components are present, their independent
skeletons are thinned by two persistent CPU workers. Graph construction and 3D
lifting still use the original full-resolution component labels, so these
changes preserve the observation support and route geometry.

## Passive cube observation

The known rigid object is a bright-yellow cube with side length \(0.150\) m.
Its current observation is deliberately geometric rather than learned:

1. threshold yellow pixels in the rectified left image;
2. retain all cleaned yellow fragments when their combined area is sufficient;
3. unproject its registered depth pixels;
4. robustly extract planar subsets;
5. select mutually perpendicular cube faces;
6. recover the raw cube pose from the visible planes and known side length;
7. jointly refine centre and rotation against the selected face points;
8. estimate a six-dimensional tangent-space pose covariance.

Three visible faces directly constrain the cube centre. With two visible
adjacent faces, their normals determine the three cube axes and the robust
shared-edge point span supplies the remaining centre coordinate. The span must
cover 90--110% of the known width; incomplete observations are rejected rather
than extrapolated. Fewer than two perpendicular faces also produce an invalid
measurement.

Image connectivity is not used as object identity: an occluding cable may
split one physical face into multiple yellow components. Those original
fragments are pooled for the robust 3D fit without filling the occluded pixels
or introducing cable depth into the cube cloud.

The synchronized PIDNet cable and endpoint masks are dilated by
`cable_exclusion_radius_px` and removed from the yellow evidence before depth
unprojection. This rejects cable-coloured pixels and RGB/depth boundary bleed;
setting the radius to zero disables the exclusion for ablation. Cube fitting
starts immediately after PIDNet inference and overlaps cable observation
construction and the PF update, preventing the cube fit from becoming a serial
frame-rate limiter. PIDNet mask union, dilation, and exclusion execute inside
the cube worker rather than on the cable tracking thread.

A geometrically uniform cube has 24 equivalent proper rotations. The raw
tracker selects the representation closest to the preceding valid observation
solely to prevent representation-only 90-degree changes. The raw centre,
rotation, quaternion, and surface RMS retain their original per-frame meaning.
Refined centre, rotation, surface RMS, and pose covariance are separate fields
on the same measurement, with their own validity and reason. A refinement
failure does not invalidate or overwrite a valid raw plane measurement. The
six-dimensional covariance is ordered as camera-frame centre translation
\((x,y,z)\), then camera-frame left rotation-vector perturbation
\((r_x,r_y,r_z)\).

The separate `RigidCubeStateFilter` consumes only the refined fields. Centre
and linear velocity use a constant-velocity error-state Kalman filter. Each
accepted visual orientation is first mapped to the cube-symmetric equivalent
nearest the previous accepted visual orientation, then updates the nominal
rotation on SO(3). Prediction-only frames hold that rotation while its
covariance grows; noisy angular velocity therefore cannot make a stationary
cube spin. Angular velocity is a smoothed finite difference of consecutive
accepted, symmetry-aligned visual orientations and remains available for
interaction inference without driving the nominal pose.

Invalid raw measurements produce causal centre prediction with increasing
covariance for at most the configured prediction age; an expired state is
marked invalid rather than held indefinitely. A new measurement after expiry
reinitializes the state. The raw measurement is never changed by the temporal
filter. Its published 12-dimensional error covariance contains pose, linear
velocity, and the separately observed angular velocity. Before each update,
the six-dimensional pose innovation is tested with its full covariance. An
innovation outside the configured chi-square gate is rejected and the filter
remains prediction-only for that frame, without orientation drift.

The camera panel shows the measured yellow boundary and raw cube wireframe.
The 3D viewer shows the valid temporal cube state as a magenta wireframe and
uses the current raw pose before the temporal state initializes. It
reports whether the state used a measurement or prediction, measurement age,
linear speed, face count, refined surface residual, centre, and CPU time.
`[cube_tracking].enabled` isolates the raw observation and
`[cube_state_filter].enabled` isolates the temporal state. The standalone
`cube_tracking_tester/run_cube_tracking.py` uses this same canonical module.

Cube tracking and the first contact stage are currently observational only.
The system now reports a passive binary temporal contact probability for each
cable, but contact feedback to the cable PF, force estimation, and
sticking/sliding classification are not implemented.

The canonical cube geometry also exposes an analytical signed oriented-box
query. For arbitrary camera-frame points it returns signed distance, closest
surface point, and outward surface normal. The CUDA contact observer implements
the same oriented-box geometry for batched particles and does not construct a
cube mesh.

## Passive cable--cube contact observer

For every cable particle, the observer densely samples the ordered PF
centerline and queries the analytical oriented cube. Its physical surface gap
is

\[
g = d_{\rm cube}(\mathbf x)-r_{\rm cable},
\]

so a centreline lying one cable radius from the cube has zero gap. The closest
dense sample supplies its surface point, outward normal, cable arc coordinate,
and a contiguous near-surface arc interval. Cube pose covariance is projected
onto that surface normal and combined with the current refined surface RMS and
a fixed sensor floor. Cable uncertainty is integrated directly through the PF
weights rather than reduced to a second Gaussian approximation.

The per-particle contact compatibility combines near-zero physical gap with
near-zero relative normal velocity. Tangential agreement is not required,
because binary contact must allow sliding. When both bodies are moving,
compatible full 3D co-motion can provide a bounded positive Bayes factor; it is
inactive while stationary and its absence never argues against contact.

Each cable has a two-state continuous-time Markov prior and a scalar contact
posterior. The observation likelihood includes a bounded outlier mixture, so
one frame cannot become decisive. A measurement update is permitted only when
both the cable PF and cube state accepted evidence from the current synchronized
frame. In addition, each particle's closest cube sample is looked up in the
PF's current dense arc-support mask. A supported sample uses its contact/free
likelihood; a missing or unknown sample contributes the same neutral likelihood
to both states. The posterior-weighted supported mass therefore scales the
evidence continuously without a global visibility threshold. Remote visible
fragments cannot validate a hidden contact arc. If either estimator is
prediction-only or local support mass is zero, the contact state only predicts
forward; missing observations cannot create contact confidence.

The reported point, normal, gap, and arc interval come from the particle with
the largest product of PF weight, local observed-arc support, and contact
compatibility. Thus the displayed contact geometry cannot be selected from a
high-weight particle whose proposed contact arc is hidden.

`[contact].enabled` isolates this observer. It reads PF tensors only after the
visual update and never changes particles, velocities, weights, proposals,
constraints, ESS, or resampling. The 3D viewer shows a near-surface contact
point and normal plus one compact line with probability, gap, normal velocity,
local support mass, arc interval, and whether the frame used measurements (`M`)
or prediction (`P`).

## Skeleton graph

The cable-body mask is thinned to a one-pixel centerline. The pixel graph uses
8-neighbour connectivity with diagonal corner pruning. It is compressed into:

- graph nodes at endpoints, endpoint anchors, and junction regions;
- graph edges containing ordered image pixels;
- an ordered 3D polyline for every edge;
- measured edge length;
- endpoint anchors attached to either end of an edge.

Each directly observed graph edge is exported as a `GraphEdgeObservation`.
This is raw observation structure. It contains no PF-derived cable identity.

### Complete routes

When both endpoints of one cable are present in the same graph component, the
observation layer enumerates bounded edge-simple trails between them. A trail
is converted into an ordered 3D route and resampled for PF use.

The search is bounded by:

- candidate count;
- expanded search states;
- graph edges per trail;
- cable-length margin.

Complete routes initialize a previously uninitialized cable and provide
route-centred broad proposals. Once a filter is initialized, tracking uses all
confidently attributed visible graph edges, so a complete route is no longer
required on every frame.

### Removed endpoint-fragment path

The previous partial-observation implementation retained only one incident
edge at each endpoint and compared it with a cable prefix or suffix. It could
ignore other visible graph components and therefore reward an incorrect
population during disconnection. That implementation, its feature switches,
and its scoring code have been removed.

When no complete route is available, an initialized filter can still update
from all confidently attributed visible graph edges. It performs a
prediction-only update only if there is no usable attributed edge.

No straight line is created across an occlusion, and no latent gap is rendered
as measured cable geometry.

## Cable particle model

One particle is a complete ordered 3D cable:

\[
X^{(p)} =
(\mathbf{x}_0,\mathbf{x}_1,\ldots,\mathbf{x}_{N-1}).
\]

The configured implementation uses 16 nodes and 800 particles per cable.
Consecutive nodes are constrained to equal segment lengths:

\[
\|\mathbf{x}_{i+1}-\mathbf{x}_i\|
\approx
\frac{L}{N-1}.
\]

The segments are a discretization, not an assumption that the cable is
globally straight. Dense samples are interpolated along every link for
measurement scoring and visualization.

### Physical cable volume

Both cables have a configured measured diameter of \(0.009\) m and radius
\(0.0045\) m. The PF state remains the ordered medial centerline because this
is the minimal state needed for efficient batched prediction and scoring.
Every selected centerline also defines a physical capsule tube:

\[
\mathcal K(X,r)
=
\left\{
\mathbf y:
\min_i d\!\left(
\mathbf y,
[\mathbf x_i,\mathbf x_{i+1}]
\right)
\le r
\right\}.
\]

The viewer renders this volume as a closed swept tube with hemispherical end
caps. Its cross-section frames use parallel transport, avoiding the twisting
and undefined straight-section behavior of Frenet frames. Press `B` to isolate
the physical body visualization.

Only the two selected estimates are meshed. Particle prediction, constraints,
and likelihood still operate on centerlines, so adding the physical volume
does not increase PF state size or mesh all 1,600 particles. The configured
radius is now carried in every `ParticleFilterFrame` for subsequent analytic
cable-object, cable-cable, and self-contact calculations.

Each particle also has ordered per-node velocity. Prediction uses damped
constant velocity plus smooth process noise. Ten percent of the population is
route-centred when a complete route is present. During a disconnected
observation it is instead centred on the visible-edge-transported prior, with
the same configured broad deformation noise.

## Active measurement scoring

A complete graph route \(R\) initializes an uninitialized population by
comparing ordered dense particle samples with ordered route samples:

\[
E_{\mathrm{route}}^{(p)}
=
\frac{1}{K}
\sum_k
\rho_H
\left(
\frac{
\|\mathbf{x}^{(p)}(s_k)-\mathbf{r}(s_k)\|
}
{\sigma_{\mathrm{trace}}}
\right).
\]

The Huber loss is quadratic near zero and linear for larger residuals. This
limits the influence of one unusual depth sample without making large
residuals free.

For an initialized cable, each attributed visible edge \(e\) supplies an
identity, orientation, confidence \(\gamma_e\), and predicted cable interval
\([s_a,s_b]\). Every particle is sampled at the same ordered arc coordinates:

\[
E_{\mathrm{visible}}^{(p)}
=
\frac{
\sum_e \gamma_e\ell_e
\operatorname{mean}_k
\rho_H
\left(
\frac{
\|\mathbf q_{e,k}-\mathbf X^{(p)}(s_{e,k})\|
}
{\sigma_{\mathrm{trace}}}
\right)
}{
\sum_e \gamma_e\ell_e
}.
\]

The edge-length factor prevents a tiny graph spur from carrying the same
measurement mass as a long visible cable section. Only directly observed
intervals are scored. Hidden intervals receive neither a reward nor a missing
data penalty.

The active energy also contains:

- fixed-link and endpoint feasibility;
- an unsupported-run penalty only during complete-route initialization;
- a configurable smoothness term;
- the previous particle weight when resampling is not required.

Each cable chooses and normalizes its weights independently.

## Prediction and resampling

The PF uses effective sample size:

\[
N_{\mathrm{eff}} =
\frac{1}{\sum_p (w^{(p)})^2}.
\]

With adaptive ESS enabled, systematic resampling runs only when a real route
or visible-edge measurement is available and the configured ESS threshold is
crossed. With it disabled, every accepted measurement is resampled. The
resampling implementation is batched on CUDA.

When there is no usable route or attributed-edge measurement, initialized
filters can commit their prediction if `prediction_without_measurement` is
enabled. Missing data therefore does not invent a cable-shape measurement.
With `hold_occluded_endpoints` enabled, each endpoint that disappears after a
valid observation remains a zero-velocity boundary at its last observed 3D
position. This applies during visible-edge updates and prediction-only frames.
Actual endpoint visibility remains unchanged in diagnostics, and a newly
observed endpoint immediately replaces the held boundary. The assumption is
independently switchable for ablation.

## Prior-based soft graph-edge attribution

Before the current frame measurement update, the PF computes a predicted
reference curve from its prior particle population:

\[
\bar X_c^- =
\sum_p w_{c,p}X_{c,p}^-.
\]

The weighted particle spread is retained as prediction uncertainty.

For each observed graph edge \(e\), the attributor estimates:

\[
A_e =
\left(
P(c=1),
P(c=2),
P(\mathrm{unassigned}),
o,
[s_a,s_b],
\gamma
\right).
\]

Here:

- \(c\) is cable identity;
- \(o\) is edge orientation;
- \([s_a,s_b]\) is the predicted cable arc interval;
- \(\gamma\) is identity confidence.

### Ordered interval fitting

The attributor enumerates contiguous intervals on each predicted cable in both
directions. The observed edge samples are aligned in order with each interval.
The cost is:

\[
E_{\mathrm{attr}}
=
\operatorname{mean}_k
\rho_H
\left(
\frac{\|\mathbf q_{e,k}-\bar X_c^-(s_k)\|}
{\sqrt{\sigma_0^2+\sigma_{\mathrm{PF}}^2}}
\right)
+
\lambda_L
\rho_H
\left(
\frac{|\,|s_b-s_a|-\ell_e\,|}
{\sigma_L}
\right).
\]

The first term measures ordered 3D agreement. The second provides a modest
edge-length consistency check. PF spread weakens prediction confidence rather
than turning an uncertain prediction into a hard gate.

Endpoint anchors constrain identity and arc direction:

- endpoint 0 corresponds to arc coordinate zero;
- endpoint 1 corresponds to the configured cable length;
- an edge anchored to one endpoint group cannot be assigned to the other cable.

Unanchored edges are allowed to remain ambiguous or unassigned. They do not
participate in motion transport or likelihood evaluation until one cable has
enough probability. This prevents a visually shared branch from being forced
into a false hard identity.

### Temporal fragment identity

The skeleton graph is rebuilt every frame, so component and edge numbers are
not persistent identities. Each current 3D edge is therefore matched against a
short history of observed edges using directed point-to-polyline distance.
Directed distance allows a new fragment to match a subset of an older edge
when occlusion splits the graph.

The matched edge's previous cable probabilities form a decaying soft prior:

\[
E_{\mathrm{identity}}(c)
=
E_{\mathrm{attr}}(c)
-
\lambda_T\log p_{t-1}(c).
\]

Spatial match quality and history age reduce the prior toward a uniform
distribution. The prior is ignored beyond the configured 3D match gate.
Consequently, current geometric evidence can override history; identity is
never hard-locked. Endpoint anchors remain authoritative.

### Attribution UI

The live skeleton panel displays:

- cable-1 edges in endpoint-group-1 blue;
- cable-2 edges in endpoint-group-2 green;
- ambiguous edges in orange;
- unassigned edges in gray;
- identity probability;
- attributed arc interval;
- mean 3D residual;
- temporal-match distance and matched-fragment count;
- assigned, ambiguous, and unassigned counts;
- attribution CPU preparation and CUDA time.

The `Y` control toggles attribution visualization only. `I` independently
enables the temporal identity prior. The measurement
switches are independent, allowing tracking to remain active with a clean
skeleton view or allowing attribution to be inspected without applying its
measurements.

## Current occlusion behavior

The implemented state after removing the unsound endpoint-fragment scorer is:

- uninitialized cable: a complete endpoint-to-endpoint route is required;
- initialized cable: confidently attributed visible edges are measurements;
- graph disconnected: visible edges still update particle weights and motion;
- no attributed edge: prediction-only update;
- hidden geometry retains its own damped per-node motion prediction;
- uncertainty is reported from the particle posterior;
- no straight hidden connector is created.

## Local node-motion update

For a node near a confidently attributed visible edge, measurement
displacement is:

\[
\Delta\mathbf{x}_i
=
\mathbf{x}_i^{\mathrm{observed}}
-
\mathbf{x}_i^{\mathrm{predicted}}.
\]

Observed edge samples affect a node only when their cable-arc distance is
within the configured local support radius. Their confidence-weighted
displacement gives the local velocity innovation:

\[
\mathbf{v}_{i,t+1}
=
\mathbf{v}_{i,t}^{\mathrm{predicted}}
+
K\,\frac{\Delta\mathbf{x}_i}{\Delta t}.
\]

No displacement is interpolated across a hidden interval or extrapolated
beyond a visible fragment. Unsupported nodes keep their own damped predicted
velocity. The fixed-link velocity projection is then applied once; it removes
only adjacent radial relative velocity required by inextensibility and does not
impose a shared cable translation.

## Projected-junction policy

Projected crossing candidates come from junction regions in the thinned
cable-body graph, not from a neural semantic channel. A junction supplies a
location and incident graph edges. It does not determine which half-edges form
one physical cable, whether the junction is a self-crossing or two-cable
crossing, or whether the cables physically touch.

Endpoint identity, fixed cable length, temporal continuity, visible-edge
attribution, and particle likelihood resolve cable traversal hypotheses.
Physical contact and height ordering remain outside the skeleton observation
model. The separate passive contact observer uses the measured cable radius,
but this radius still does not alter the PF centreline measurement likelihood.

## Feature controls

Current live controls are:

| Key | Feature | Effect |
|---|---|---|
| `2` | `connected_trace` | Ordered route scoring instead of unordered proximity |
| `5` | `smoothness` | PF curvature regularization |
| `6` | `fixed_length` | Fixed-link position constraints |
| `7` | `temporal_prediction` | Damped velocity prediction |
| `M` | `endpoint_motion_transport` | Boundary-conditioned proposal transport |
| `O` | `hold_occluded_endpoints` | Hold a hidden endpoint at its last observed 3D position |
| `E` | `adaptive_ess_resampling` | Use the ESS threshold; off resamples every accepted measurement |
| `9` | `global_particles` | Ten-percent retracking proposals |
| `V` | `local_node_motion` | Evidence-gated velocity update for locally supported nodes |
| `F` | `single_endpoint_updates` | Permit one-endpoint anchoring |
| `G` | `prediction_without_measurement` | Commit temporal prediction when observation is unavailable |
| `H` | `posterior_uncertainty` | Posterior covariance diagnostics |
| `K` | `fused_constraint_kernels` | Native fused CUDA link constraints |
| `J` | `cuda_graph_replay` | CUDA graph replay for supported constraint paths |
| `Y` | `edge_attribution_diagnostics` | Read back and display ordered edge attribution |
| `I` | `temporal_edge_identity` | Soft graph-fragment identity memory across frames |
| `A` | `visible_edge_scoring` | Use attributed visible edges as PF likelihoods |
| `D` | `visible_edge_transport` | Transport predicted shape from visible-edge displacement |
| `S` | `visible_edge_exploration` | Centre the 10% broad population on the transported prior during disconnection |

The removed `partial_observation` and `fragment_score` switches no longer
exist because their implementation was removed rather than retained as legacy
code.

## Live evaluation

The main application opens one separate evaluation window with three signals
for each cable:

- visible trace error in millimetres;
- maximum posterior node uncertainty in millimetres;
- fraction of the estimated cable supported by the current observation.

These are complementary: measurement fit, estimator confidence, and how much
of the cable is actually visible. Endpoint and fixed-length errors are not
plotted because the active endpoint and link constraints make them poor
independent quality indicators.

Evaluation diagnostics are sampled at the configured 5 Hz rather than every
tracking frame. With the viewer active, samples are aligned to fresh viewer
diagnostic frames so the application performs no extra attribution readback.
In headless mode, the same configured cadence applies independently. The live
history is saved to `metrics.csv`, and the final displayed plot is saved to
`metrics.png` under `diagnostics/evaluation_runs/<timestamp>`. CSV output is
buffered and flushed once per second and at shutdown. These two files are the
intended compact inputs for later analysis.

The plot remains cable-only. Each CSV row also records the synchronized raw
cube measurement: validity, rejection reason, fitted face count, centre XYZ,
and cube-surface RMS. Invalid cube measurements leave centre and RMS empty
rather than substituting a held or predicted pose.

When contact inference is enabled, the same row also saves each cable's contact
probability, evidence/prediction flag, physical gap and uncertainty, relative
normal velocity, co-motion score, local contact-arc support mass, contact arc
interval, closest surface point, surface normal, and reason. These values are
intentionally logged rather than added as more live plots.

The CSV also records four compact observation diagnostics for each cable:
detected endpoint count, the two endpoint body-component labels, complete-route
count, and the observation reason. These values distinguish a segmentation
connectivity failure from a 3D graph/depth failure without changing the
observation or PF mathematics.

## Performance isolation

Capture, tracking, point-cloud preparation, and rendering run in separate
stages. Tracking buffers are allocated once and acquired before ZED image and
depth transfers; if tracking is saturated, the frame is skipped without those
copies. Viewer stride and rendering rate do not reduce the tracking
observation. The default viewer cloud is reconstructed from the already
registered depth only on display frames; full ZED XYZRGBA retrieval remains an
explicit diagnostic option.

Profiler output separates:

- NN inference;
- observation construction;
- skeleton and graph construction;
- graph-edge export;
- PF CUDA stages;
- PF readback;
- graph attribution CPU preparation/readback and CUDA time;
- visible-edge likelihood CUDA time;
- passive contact wall and CUDA time.

The initial NumPy implementation was removed after the live viewer exposed
approximately 39 ms diagnostic-frame latency for eight edges. The implemented
attributor evaluates the same exhaustive ordered interval set as one batched
CUDA operation. A warmed synthetic eight-edge test measured approximately
1.43 ms median wall time and 1.01 ms median CUDA time. Real-camera cost still
depends on the number of observed graph edges.

## Source layout

| File | Responsibility |
|---|---|
| `NN_collection_training/source/cable_pidnet.py` | Shared PIDNet model and runtime inference |
| `NN_collection_training/source/cable_detection.py` | Shared PIDNet runtime thresholds and mask cleanup |
| `NN_collection_training/source/config.toml` | Canonical checkpoint, inference, threshold, and cleanup settings |
| `ZED_segmentation_viewer/source/observation.py` | RGB-D geometry, endpoints, skeleton graph, routes, raw graph edges |
| `ZED_segmentation_viewer/source/graph_attribution.py` | Prior-based ordered edge-to-prediction association |
| `ZED_segmentation_viewer/source/particle_filter.py` | Batched PF prediction, scoring, constraints, posterior, attribution orchestration |
| `ZED_segmentation_viewer/source/cuda_constraints.py` | Fused CUDA fixed-link kernels |
| `ZED_segmentation_viewer/source/cable_geometry.py` | Capsule-tube geometry and parallel-transport mesh construction |
| `ZED_segmentation_viewer/source/cube_tracker.py` | Raw known-cube RGB-D plane fitting, joint pose refinement/covariance, and analytical cube surface queries |
| `ZED_segmentation_viewer/source/cube_state_filter.py` | Separate causal cube centre/pose state, symmetry-safe visual orientation, velocities, and covariance |
| `ZED_segmentation_viewer/source/contact_estimator.py` | Read-only uncertainty-aware binary cable--cube contact observer |
| `ZED_segmentation_viewer/source/live_evaluation.py` | Three-signal live plot and CSV logger |
| `ZED_segmentation_viewer/source/app.py` | Asynchronous capture/tracking/viewer preparation and console profiling |
| `ZED_segmentation_viewer/source/split_viewer.py` | OpenGL viewer, diagnostics, feature controls, recording |
| `ZED_segmentation_viewer/source/config.toml` | Runtime parameters and independent feature switches |

## Required invariants

The following are intentional research constraints:

- the NN never produces cable-1 and cable-2 body masks;
- endpoint groups are the direct identity anchors;
- graph attribution uses the predicted prior, not the updated posterior;
- attribution is computed before the current measurement changes the PF;
- only confidently attributed visible edges change tracking;
- missing geometry is never drawn as a measured straight cable;
- graph junctions never declare physical contact or over/under ordering;
- every future partial measurement must retain order and arc identity;
- hidden geometry must remain uncertain when it is not observable.
