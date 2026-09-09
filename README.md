# Aerial cable research - Simulator branch

The complete application and seven-page UI, starting with empty experiment
libraries. Source was ported from `twin-rewrite` commit `19dc9a2` into a fresh
`Simulator` branch. No collected flights, processed datasets, fitted M0/M1,
residual weights, learned policies, checkpoints, or past results are included.

## Run the full application

Install Python 3.12 and dependencies, then launch:

```sh
python -m pip install -r requirements.txt
python run_simulation.py
```

The original pages are retained: **Model**, **Recordings**, **PPO**,
**Diagnostics**, **Rehearsal & Export**, **MPPI Planner**, **Adaptation Check**.
Training settings, policy management, model/residual fitting, CSV processing,
rehearsal, export, visualization, and their implementation code remain present.
The source for removed legacy navigation is retained where shared compatibility
code needs it; the active navigation stays unchanged.

For a short headless dynamics check:

```sh
python run_simulation.py --headless --device cpu --duration 0.05
python -m pytest tests/test_clean_simulator.py tests/physics/test_point_mass.py -q
```

The tests are ported too. Historical integration tests still require their
original non-distributed artifacts; they are not all runnable in this empty
workspace. No full-suite success is claimed for a data-free checkout.

## Empty starting state

`config/` provides portable, explicitly **unfitted** example parameters so the
application opens without loading old weights. The geometry/mass layout is kept
as design parameters, and the nominal response/EI/Cb values are illustrative.
Both residuals are disabled until newly fitted weights are supplied. There is
no learned bootstrap action sequence or checkpoint selected for continuation.
This nominal profile is **not M0** and is not evidence of a fitted aircraft model.
The dataset manifest is empty; generated data/models/runs remain ignored by Git.

## Next step

Set up an independent **Isaac Lab / PhysX drone and passive cable**, collect new
preliminary data, and fit a new **M0**. Then train a new policy, execute its
FullState sequence in that plant, record new takes, and perform adaptation.

This port does not implement that new PhysX plant or collect data. The existing
Isaac Lab integration still hosts our external model, and must not be called an
independent PhysX experiment. The inherited five-take adaptation runner is kept;
it still needs the preliminary/synthetic-data workflow before being used to fit
new M0. No fit, policy training, or flight execution was launched during the port.

The original project, original data, and all trained artifacts remain preserved
on `twin-rewrite`. See `PORT_MANIFEST.json` for source provenance.
