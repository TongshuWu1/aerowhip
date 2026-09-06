# Windows development and Ubuntu deployment

Use `Whip-Development.zip` on the Windows 4080 workstation. It includes current source, data, fit, selected PPO, stopped SAC/comparison evidence and tests. Use `Whip-Deployment.zip` on the Ubuntu flight computer; it contains a headless offline PPO planner and no recordings/results/GUI. Both archives have checksums in `BUNDLES.json`, and each extracted folder has a per-file `TRANSFER_MANIFEST.json`. Transfer through your preferred file-transfer method; nothing here sends email.

## Windows development / PyCharm

Extract into a writable directory and open that directory as a PyCharm project. Install Python 3.12 if absent. In a terminal at the project root:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe tools/lab_preflight.py --device cpu
.venv/Scripts/python.exe tools/lab_preflight.py --device cuda --batch 1024
.venv/Scripts/python.exe run_simulation.py
```

In PyCharm Settings → Project → Python Interpreter, select the newly created `.venv/Scripts/python.exe`. Use the project interpreter and project-root working directory in Run configurations. The bundled `.run` configurations use `$PROJECT_DIR$`, not the original user's interpreter path. See [JetBrains interpreter instructions](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html).

The UI and existing selected policy are unchanged. New training defaults use batch 1,024 rather than the 5090 study's large batch. This is a conservative starting allocation, not a claim of maximal throughput or guaranteed full-training memory on 16 GB. Confirm CUDA preflight, then measure a short new run if training is needed. A resumed historical run may restore its historical batch; inspect it first. Do not retrain solely to deploy this checkpoint.

## Ubuntu

For the small package, follow its root README (`deployment/README.md` in the full project). The planner uses CPU and needs no ROS installation. Keep the colleague's ROS environment separate if its Python version differs; the initial adapter can exchange timestamped files. The small package has no desktop/training launcher.

The full development project is also intended to work on Ubuntu: create `.venv` with `python3.12 -m venv .venv`, replace `.venv/Scripts/python.exe` with `.venv/bin/python` in the commands above, and select `.venv/bin/python` in PyCharm. Desktop rendering also requires a functioning graphical session and OS Qt/OpenGL libraries. The headless package avoids those dependencies.

The [official PyTorch selector](https://pytorch.org/get-started/locally/) is the source for compatible CUDA wheels. The packaging workstation uses torch 2.11.0+cu128; the cu128 index was checked during preparation. A compatible NVIDIA driver is still required. Do not infer CUDA support just from a GPU being present. The wheel normally supplies NVRTC; the runtime searches Windows torch DLLs or Linux `nvidia/cuda_nvrtc/lib` and optionally `CUDA_PATH`. It compiles for the detected GPU rather than hard-coding the 5090. [NVIDIA lists GPU compute capabilities](https://developer.nvidia.com/cuda/gpus).

We can test the Windows 5090 locally; actual 4080/5080 and Ubuntu execution remain target-machine checks. `lab_preflight` performs a finite physics step and reports memory for that step, not full training peak memory. If CUDA fails, keep the report and diagnose the wheel/driver/NVRTC error; do not alter physical parameters or lower precision to make the check pass.

## Transfer versus publication

The development transfer is a private research handoff with raw measurements and results. It is not the source-only public-release candidate built by `tools/build_source_release.py`. Neither export chooses a publication license or publishes anything. Retired siblings, virtual environments and Git history are excluded. Historical absolute paths in immutable evidence are provenance; active selectors in the transfer are relative.

Read `HANDOFF.md` before changing fits/rewards or resuming training. `docs/CONTROLLER_INTERFACE_REVIEW.md` identifies the remaining real-controller work. No exported plan is declared flight-ready.
