# Climate modelling using heterogenous MOMARL

Reinforcement learning (RL) training module for the JUSTICE climate-economy model. Supports both single-objective Multi-Agent RL (MARL) and Multi-Objective MARL (MOMARL) using algorithms from [HARL](https://github.com/PKU-MARL/HARL).

The computational experiments in this project were run on the DelftBlue supercomputer, provided by Delft High Performance Computing Centre ([https://www.tudelft.nl/dhpc](https://www.tudelft.nl/dhpc)).

---

## Table of contents

- [Structure](#structure)
- [Environments](#environments)
  - [Adding a new environment](#adding-a-new-environment)
- [Training](#training)
  - [Setup](#setup)
  - [MARL (single-objective)](#marl-single-objective)
  - [MOMARL (multi-objective)](#momarl-multi-objective)
  - [A note on HARL](#a-note-on-harl)
  - [HPC / array jobs](#hpc--array-jobs)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Experiment tracking](#experiment-tracking)
- [Troubleshooting](#troubleshooting)
- [Tips](#tips)

---


## Structure

```
thesis_rl/
├── train.py               # Entry point for single-objective MARL training
├── train_momarl.py        # Entry point for MOMARL training (Pareto front coverage)
├── evaluation.py          # Entry point for evaluating a saved model (used by root-level evaluate_*.py)
├── tune_harl.py           # W&B Bayesian hyperparameter sweep
├── args.py                # Argument parsing and dataclasses (TrainArgs, EnvArgs, MOMARLArgs)
├── envs/
│   ├── justice_environment.py       # PettingZoo ParallelEnv wrapping the JUSTICE model
│   └── justice_environment_moma.py  # MOMAland MOParallelEnv with vector rewards
├── configs/
│   ├── algo_comparison/   # Config YAMLs comparing HAPPO, MAPPO, HASAC, HATD3 (temperature vs. economic output)
│   └── welfare/           # Config YAMLs for welfare-objective experiments
├── evaluation/
│   ├── evalute_model.ipynb          # RL policy evaluation notebook
│   ├── detailed_analysis.ipynb      # Climate/economic detailed analysis
│   └── momarl_rl_visualization.ipynb # MOMARL Pareto front visualization
├── agents/                # Agent implementations
├── algorithms/            # Algorithm extensions
├── HPC/                   # Slurm job scripts (may differ from HPC cluster copies)
└── HARL/                  # External HARL submodule
```

Root-level scripts (outside `thesis_rl/`) used for running and analysing experiments:

| Script | Purpose |
|---|---|
| `evaluate_happo.py`, `evaluate_hasac.py`, `evaluate_mappo.py` | Loop over all trained models of one algorithm and call `thesis_rl/evaluation.py` for each; outputs go to `eval_results/detailed_analysis/evaluation/<algo>/` |
| `evaluate_welfare_hasac.py` | Same, for the welfare-objective HASAC runs (outputs to `eval_results/welfare_analysis/`) |
| `eval_happo.sh`, `eval_hasac.sh`, `eval_mappo.sh` | Slurm wrappers that run the corresponding `evaluate_*.py` on the cluster |
| `sync_happo_results.sh`, `sync_hasac_results.sh`, `sync_mappo_results.sh` | `rsync` selected runs from DelftBlue (`/scratch/aandrasz/JUSTICE/results/...`) into local `results/` |



## Environments

Both environments wrap the **JUSTICE** integrated assessment model (implemented externally) and expose a [PettingZoo](https://pettingzoo.farama.org/) parallel API. Each agent corresponds to a world region.

> If you are unfamiliar with JUSTICE, start with [JUSTICE_example_run.ipynb](../JUSTICE_example_run.ipynb) in the repo root. It walks through model initialisation, running a simulation, evaluating welfare outcomes, and visualising results.

| Environment | Class | Reward |
|---|---|---|
| `justice_environment.py` | `JusticeEnvironment` | Scalar (configurable welfare/temperature/economic objective) |
| `justice_environment_moma.py` | `JusticeEnvironmentMOMA` | Vector (one value per objective) |

**Supported MOMARL objectives** (`rewards` in `EnvArgs`):
- `inverse_global_temperature`
- `global_economic_output`
- `consumption_per_capita`
- `gini_consumption`
- `temperature_threshold`
- `welfare`

Agents control the **emissions control rate** (and optionally the **savings rate**) at each timestep (2015–2300).

### Adding a new environment

To integrate a new environment with HARL, three things are needed:

**1. Implement the environment**

Create a new environment class in `thesis_rl/envs/`. For single-objective use, extend PettingZoo's `ParallelEnv` (see `justice_environment.py`). For multi-objective use, extend MOMAland's `MOParallelEnv` (see `justice_environment_moma.py`).

**2. Create a HARL wrapper**

Create a wrapper class in `thesis_rl/HARL/harl/envs/harl_justice/` that adapts your environment to the HARL interface (see `harl_justice.py` or `harl_justice_momarl.py`). The wrapper must implement `step()`, `reset()`, `get_avail_actions()`, and expose `observation_space`, `action_space`, and `share_observation_space`.

**3. Register in HARL**

Add the new environment name in `thesis_rl/HARL/harl/utils/envs_tools.py` — there are three `elif env_name == ...` blocks where environments are instantiated (for training, evaluation, and rendering). Also add an entry to the `LOGGER_REGISTRY` in `thesis_rl/HARL/harl/envs/__init__.py`.

See also the [Applying to New Environments](https://github.com/PKU-MARL/HARL#applying-to-new-environments) section of the HARL README for the minimal interface contract and logger requirements.


## Training

### Setup

**1. Install dependencies**

From the repo root (requires Python >=3.10):

```bash
pip install -e .
```

Key packages installed: `torch`, `pettingzoo`, `momaland`, `morl-baselines`, `wandb`, `harl` (via the `HARL/` submodule).

**2. Install HARL**

HARL is bundled as a submodule. Install it in editable mode:

```bash
cd thesis_rl/HARL
pip install -e .
cd ../..
```

**3. Configure Weights & Biases**

Log in to your [W&B](https://wandb.ai) account ([docs](https://docs.wandb.ai/quickstart)):

```bash
wandb login
```

Then set the project and entity in your training command (or edit the defaults in `train.py` / `train_momarl.py`):

- `entity`: your W&B username or team (default: `olaandrasz-tu-delft`)
- `--wandb_project`: the project name to log runs under

To disable W&B logging during development:

```bash
wandb disabled
```

### MARL (single-objective)

```bash
python -m thesis_rl.train --algo happo --env harl_justice
```

Algorithms available: `happo`, `hatrpo`, `haa2c`, `haddpg`, `hatd3`, `hasac`, `had3qn`, `maddpg`, `matd3`, `mappo`.

Config YAMLs under `configs/` are passed via `--algo_config` and `--env_config`.

### MOMARL (multi-objective)

```bash
python -m thesis_rl.train_momarl --algo happo --env harl_justice_momarl \
    --weights_generation 'uniform' --num_weights 100 \
    --ref_point 0.0 0.0
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--weights_generation` | `uniform` | `uniform` or `OLS` (see [Troubleshooting](#troubleshooting)) |
| `--num_weights` | `100` | Maximum number of weight vectors / training runs |
| `--start_uniform_weight` | `0` | Slice start for distributed uniform-weight runs |
| `--end_uniform_weight` | `10` | Slice end for distributed uniform-weight runs |
| `--ref_point` | `0.0 0.0` | Reference point for hypervolume calculation |
| `--base_save_path` | `results/momarl` | Directory for saving results |

Each weight vector triggers one full HARL training run. Results (weights + vector returns) are saved as `eval_results.json` and `final_results.json` under `base_save_path`.

**Uniform** generates an evenly spaced grid, suitable for parallelisation across array jobs (see [HPC / array jobs](#hpc--array-jobs)). **OLS** iteratively selects the next weight that maximises the hypervolume improvement.

See section [Configuration](#configuration) for running the training with a config file.

### A note on HARL

[HARL](https://github.com/PKU-MARL/HARL) provides the heterogeneous multi-agent RL algorithms used here. A few practical notes:

- The algorithms can be run with parallelised environments via the `n_rollout_threads` config option. Do not exceed ~10 parallel environments locally; on a remote cluster, allocate at least one CPU per parallel environment.
- All training in this project is CPU-only. The JUSTICE model is not GPU-compatible, so there is little benefit to running on GPU.

### HPC / array jobs

For large Pareto sweeps, distribute uniform-weight runs across cluster array jobs using the `--start_uniform_weight` and `--end_uniform_weight` arguments to slice the weight grid. Ready-made Slurm array scripts live in `thesis_rl/HPC/`:

| Script | Description |
|---|---|
| `HPC/train_momarl_array.sh` | Generic MOMARL array job |
| `HPC/train_momarl_happo_array.sh` | HAPPO-specific array job |
| `HPC/train_momarl_hasac_array.sh` | HASAC-specific array job |
| `HPC/train_momarl_mappo_array.sh` | MAPPO-specific array job |

Each script submits one job per weight slice. Adjust the `--array` range and `--start/end_uniform_weight` values to match the total number of weights you want to cover.

> **Note:** These scripts are sometimes edited directly on the HPC cluster and may be out of sync with this repo. Check for bugs before running them.


## Configuration

Each YAML in `configs/` bundles both the environment and algorithm configuration into a single file, covering `main_args` (algo/env selection), `env_args` (number of agents, objectives, action space, etc.) and `algo_args` (network architecture, PPO hyperparameters, rollout settings, logging directory). Passing one of these files to the training script reproduces the exact setup used in this project.

The configs are organised into two folders:

- **`algo_comparison/`** — experiments comparing HAPPO, MAPPO, HASAC and HATD3 on the temperature/economic-output objectives. Files ending in `_v1`/`_v2` are discrete-action variants; files containing `continous` (sic) use continuous actions (`discrete_actions: False`). The `_continous_v2` files are the ones used for the final algorithm comparison (results in `results/continous_3_fixed`).
- **`welfare/`** — experiments using welfare-based objectives (utilitarian, prioritarian) with HASAC (`hasac_temp_welfare_util.yaml`, `hasac_temp_welfare_prio.yaml`) and the generic `temp_welfare.yaml` / `economy_welfare.yaml`.

When running a new experiment, copy the closest existing YAML and change at least `algo_args.logger.log_dir` (where HARL writes models and logs), the objectives in `env_args.rewards`, and any algorithm hyperparameters. Check `env_args.discrete_actions`, `num_env_steps` and `n_rollout_threads` too; these differ between configs.

To use a config file, pass it via `--load_config`. For example, to run the HAPPO algorithm:

```bash
python thesis_rl/train_momarl.py \
    --weights_generation uniform \
    --num_weights 4 \
    --load_config 'thesis_rl/configs/algo_comparison/happo_temp_econ_continous_v2.yaml' \
    --wandb_project momarl_debug \
    --seed 3
```

## Evaluation

Open the notebooks in `evaluation/` after training:

- **`evalute_model.ipynb`** — load a saved policy and run rollouts, inspect scalar metrics.
- **`detailed_analysis.ipynb`** — climate and economic time-series analysis per region.
- **`momarl_rl_visualization.ipynb`** — plot the Pareto front from MOMARL results.


## Experiment tracking

All runs are logged to [Weights & Biases](https://wandb.ai) under the project specified by `--wandb_project`. MOMARL runs are grouped by experiment name so the full Pareto sweep appears as one group in the W&B UI.

## Troubleshooting

**OLS weight generation is broken**

`--weights_generation OLS` uses `morl-baselines` `LinearSupport`, which has a Python version incompatibility with the rest of the project. Use `uniform` instead. Fix by resolving the `morl-baselines` version constraint in `pyproject.toml` once a compatible release is available.

**HARL submodule not found / import errors**

Make sure you have installed HARL separately after cloning:

```bash
cd thesis_rl/HARL && pip install -e .
```

**W&B authentication errors**

Run `wandb login` before training, or set the `WANDB_API_KEY` environment variable. To skip W&B entirely during local development, run `wandb disabled`.



## Tips

A few lessons learned from working on this project that may save you some time.

* Avoid relying on AI tools at the beginning. It is useful to understand the environment implementation and (MO)MARL in detail. It might feel uncomfortable or overwhelming at first, but it will make you far less confused in the long run.
* Go step-by-step with your implementation.
* Perform local checks before running code on a remote cluster. Some experiments took several days to complete, and a small bug in a config can lead to significant wasted compute time.
* Don't panic when the results are not perfect. It is normal for things not to work on the first try.
* Feel free to reach out — you can find my socials on my [GitHub profile](https://github.com/aleksandra-and).