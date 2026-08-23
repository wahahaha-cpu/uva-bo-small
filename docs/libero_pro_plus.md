# LIBERO Pro and LIBERO-Plus

This project keeps the two benchmarks separate from the UVA training environment.

## What the names mean

`LIBERO-Pro` is the VLA-Adapter released Pro checkpoint, currently
`VLA-Adapter/LIBERO-Long-Pro`. It evaluates the original `libero_10` suite: 10
tasks and 50 trials per task in the released protocol. The existing launcher is
`/home/jinboning/project/VLA-Adapter/scripts/eval_official_libero10_pro_baseline.sh`.

`LIBERO-Plus` is the official `sylvestf/LIBERO-plus` benchmark package. It keeps
the LIBERO API and expands the four suites to 10,030 generated tasks:

| suite | tasks | Plus default trials |
| --- | ---: | ---: |
| `libero_spatial` | 2402 | 1 |
| `libero_object` | 2518 | 1 |
| `libero_goal` | 2591 | 1 |
| `libero_10` | 2519 | 1 |

Together these are 10,030 tasks. Plus classifies each generated task into one
of seven dimensions: object layout, camera viewpoint, robot initial state,
language instruction, light condition, background texture, or sensor noise.

The official Plus evaluation change is `num_trials_per_task=1`. Run each suite
separately when a complete 10,030-task evaluation is required. `libero_10` in
Plus is therefore not the original 10-task `libero_10`.

## Installation and configuration

The source checkout and runtime config are placed at:

```text
/data1/local_userdata/jinboning/LIBERO-plus
/data1/local_userdata/jinboning/LIBERO-plus-runtime/config.yaml
```

The runtime config points `benchmark_root`, `bddl_files`, `init_states`,
`datasets`, and `assets` into the Plus checkout. It is selected only by
`LIBERO_CONFIG_PATH` in the Plus launcher, so `/home/jinboning/.libero/config.yaml`
and the original `/home/jinboning/project/LIBERO` remain unchanged.

To reproduce the isolated Python extras and user-local ImageMagick runtime
without sudo:

```bash
cd /home/jinboning/project/uva-bo-small
INSTALL_RUNTIME_DEPS=1 ./scripts/eval/configure_libero_plus.sh
```

The official assets archive is about 6.4 GB and contains the new object,
texture, and scene data. The Plus source checkout already carries the BDDL and
initial-state files. Fetch and extract the asset archive with:

```bash
cd /home/jinboning/project/uva-bo-small
DOWNLOAD_ASSETS=1 ./scripts/eval/configure_libero_plus.sh
```

The command is resumable and normalizes the archive's internal prefix, leaving
the assets below `/data1/local_userdata/jinboning/LIBERO-plus/libero/libero`.
The VLA-Adapter
Python environment now contains `wand` and `scikit-image`. A user-local
ImageMagick runtime is configured at
`/data1/local_userdata/jinboning/vla-adapter-assets/system-libs/imagemagick6`,
so root access is not required. Do not `pip install -e LIBERO-plus` into the UVA
environment, since that would replace the original editable `libero` package
globally.

## Preflight and evaluation commands

Check the selected suite and all generated files:

```bash
cd /home/jinboning/project/uva-bo-small
DRY_RUN=1 ./scripts/eval/libero_plus.sh
```

Run Plus with the released Long-Pro checkpoint (one rollout per task):

```bash
CUDA_DEVICE=0 SUITE=libero_10 ./scripts/eval/libero_plus.sh
```

Run another suite:

```bash
CUDA_DEVICE=0 SUITE=libero_spatial ./scripts/eval/libero_plus.sh
```

The launcher automatically pairs each suite with its matching released Pro
checkpoint: `LIBERO-Spatial-Pro`, `LIBERO-Object-Pro`, `LIBERO-Goal-Pro`, or
`LIBERO-Long-Pro`. Only `LIBERO-Long-Pro` is currently present locally, so the
configured `libero_10` command is immediately usable. For another suite,
download its matching checkpoint or pass `BASE_CHECKPOINT` to a compatible
checkpoint whose `dataset_statistics.json` contains that suite's normalization
key. Do not use Long-Pro normalization for Spatial/Object/Goal.

Useful overrides are `BASE_CHECKPOINT`, `PYTHON_BIN`, `LIBERO_PLUS_ROOT`,
`LIBERO_PLUS_CONFIG_PATH`, `IMAGEMAGICK_ROOT`, `NUM_TRIALS_PER_TASK`, `SEED`,
`LOCAL_LOG_DIR`, and `DRY_RUN`. The wrapper makes a temporary checkpoint view because the upstream
loader rewrites model files; the official checkpoint itself is not modified.

For the original Pro baseline use:

```bash
DRY_RUN=1 ./scripts/eval/libero_pro.sh
CUDA_DEVICE=0 ./scripts/eval/libero_pro.sh
```

That path intentionally uses the original LIBERO package and the released
50-trial protocol. The current UVA `LiberoImageRunner` is an HDF5-demo runner;
it is not silently treated as a Plus evaluator. The configured Pro/Plus entry
points use the official VLA-Adapter benchmark runner, which consumes Plus BDDL,
initial states, assets, and task language directly.

Official references:

- https://github.com/sylvestf/LIBERO-plus
- https://huggingface.co/datasets/Sylvest/LIBERO-plus
- https://github.com/OpenHelix-Team/VLA-Adapter
