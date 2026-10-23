# Training scripts

Everything here is submitted to Slurm through `train.sh`. There are three layers:

```
launch_sweep.sh          queues one Slurm job per configuration
  └─ train.sh            the sbatch job: 1 node, 8 GPUs, 72h, loads the conda env
       └─ seacast_pretrain.sh / seacast_finetune.sh / ...
                         the actual `python -m neural_lam.train_model` command
```

Each layer forwards its arguments to the next one, and the training scripts append
those arguments **after** their own defaults. `argparse` keeps the last occurrence
of a flag, so anything in a training script can be overridden from the command
line without editing the file.

Run all commands from the repository root.

## Running a single job

```sh
# the default (seacast_pretrain.sh, unmodified)
sbatch training_scripts/train.sh

# a specific script
sbatch training_scripts/train.sh training_scripts/seacast/seacast_pretrain.sh

# a specific script with overrides
sbatch training_scripts/train.sh training_scripts/seacast/seacast_pretrain.sh \
    --hidden_dim 512 --graph global_cluster_1_deg_10_refinement_4_levels
```

Logs go to `slurm_logs/<jobid>_<jobname>.out`. The first line of the log echoes the
resolved command, so you can always see which settings a job actually ran with.

## Running a sweep

Edit the `RUNS` array at the top of [`launch_sweep.sh`](launch_sweep.sh) — one line
per job, containing the flags for that run:

```bash
RUNS=(
    "--hidden_dim 256 --graph global_cluster_1_deg_20_refinement_4_levels"
    "--hidden_dim 256 --graph global_cluster_1_deg_15_refinement_4_levels"
    "--hidden_dim 512 --graph global_cluster_1_deg_20_refinement_4_levels --batch_size 2"
)
```

Then queue them all:

```sh
bash training_scripts/launch_sweep.sh
```

Options:

| Command | Effect |
| --- | --- |
| `DRY_RUN=1 bash training_scripts/launch_sweep.sh` | Print the `sbatch` commands without submitting. Do this first. |
| `DEPEND=15131 bash training_scripts/launch_sweep.sh` | Start every job only after job 15131 finishes (`afterany`). |
| `bash training_scripts/launch_sweep.sh <script>` | Sweep a different training script than `seacast_pretrain.sh`. |

Each job is named `SeaCast_h<hidden_dim>_<short graph>` (e.g. `SeaCast_h256_1_deg_20`),
so the runs land in separate log files. `squeue -u $USER` shows the queue.

The `RUNS` lines are not limited to `--hidden_dim` and `--graph` — any flag of
`neural_lam.train_model` can be added, e.g. `--batch_size`, `--lr`,
`--processor_layers`, `--load`.

## SeaCast pretraining

[`seacast/seacast_pretrain.sh`](seacast/seacast_pretrain.sh) — `graph_det` on the 1°
data (`data/global_ocean_1_density.yaml`), `hidden_dim 256`, `processor_layers 3`,
`batch_size 3`, muon optimizer, `--scheduler deterministic`.

```sh
# single run with the defaults
sbatch training_scripts/train.sh training_scripts/seacast/seacast_pretrain.sh

# sweep hidden_dim / graph
DRY_RUN=1 bash training_scripts/launch_sweep.sh   # check
bash training_scripts/launch_sweep.sh             # submit
```

Graphs available at 1° (`data/graphs/`):

```
global_cluster_1_deg_20_refinement_4_levels
global_cluster_1_deg_15_refinement_4_levels
global_cluster_1_deg_10_refinement_4_levels
```

Checkpoints are written to `saved_models/train-graph_det-<layers>x<hidden_dim>-<date>-<id>/`
as `min_val_loss.ckpt` and `last.ckpt`.

## SeaCast finetuning

[`seacast/seacast_finetune.sh`](seacast/seacast_finetune.sh) — the same model at
**1/4°** (`data/global_ocean_1_4.yaml`), with `--scheduler finetune_deterministic`
and `--load` pointing at a pretrained checkpoint.

Two things differ from pretraining and are easy to get wrong:

1. The graph must be a **`1_4`** graph, matching the 1/4° config.
2. `--load` in the script is the literal placeholder `CHECKPOINT`. Always pass a real
   checkpoint path, either by editing the script or by overriding it on the command
   line (the override wins).

```sh
sbatch training_scripts/train.sh training_scripts/seacast/seacast_finetune.sh \
    --load saved_models/train-graph_det-3x256-08_04_23-0203/min_val_loss.ckpt
```

To sweep finetuning runs, put the matching graphs and checkpoints in `RUNS` and pass
the finetune script to the launcher:

```bash
RUNS=(
    "--hidden_dim 256 --graph global_cluster_1_4_deg_20_refinement_4_levels --load saved_models/<pretrain_run_a>/min_val_loss.ckpt"
    "--hidden_dim 256 --graph global_cluster_1_4_deg_15_refinement_4_levels --load saved_models/<pretrain_run_b>/min_val_loss.ckpt"
)
```

```sh
bash training_scripts/launch_sweep.sh training_scripts/seacast/seacast_finetune.sh
```

Use `DEPEND=<pretrain_jobid>` to queue the finetuning behind a pretraining job that is
still running.

Graphs available at 1/4°:

```
global_cluster_1_4_deg_20_refinement_4_levels
global_cluster_1_4_deg_15_refinement_4_levels
global_cluster_1_4_deg_10_refinement_4_levels
```

## Probabilistic fine-tuning

[`graph_crps/README.md`](graph_crps/README.md) — CRPS fine-tuning of the 1/4°
`graph_det` model into an ensemble model, with FGN noise, MC dropout and/or
per-level noise (`--stochasticity`), in distributed, checkpointed or single-GPU
mode, and how many epochs to run.

```sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_fgn.sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_mc.sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_sdl.sh
```

## Notes

- `train.sh` requests 8 GPUs on 1 node for 72 hours. Change the `#SBATCH` header there
  if a run needs different resources — it applies to every job in a sweep.
- W&B run names are `train-<model>-<processor_layers>x<hidden_dim>-<date>-<id>` and do
  **not** contain the graph. When comparing runs that share a `hidden_dim`, tell them
  apart by the Slurm job name or the graph recorded in the W&B config.
- Raising `hidden_dim` may need a smaller `batch_size` to fit in memory; add
  `--batch_size 2` to that run's line in `RUNS`.
- [`debug.sh`](debug.sh) runs training directly (no `srun`, `wandb off`) for quick
  interactive checks on an allocated node.
