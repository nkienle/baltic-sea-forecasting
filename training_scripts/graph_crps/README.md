# Probabilistic fine-tuning of SeaCast (`graph_crps`)

This turns the deterministic ¼° SeaCast model (`graph_det`) into an ensemble model.
It does so by fine-tuning with the (almost fair) CRPS loss, with one or more sources
of stochasticity switched on.

Run all commands from the repository root. Job submission works as for the other
training scripts, see [`../README.md`](../README.md).

## The three sources of stochasticity

Select them with `--stochasticity`. Any combination is allowed, e.g.
`--stochasticity mc`, `--stochasticity sdl`, `--stochasticity fgn mc sdl`.

| Source | What it does | Main flags | Reference |
| --- | --- | --- | --- |
| `fgn` | A random vector z ~ N(0, I), drawn per sample and step, is embedded and fed to *every* conditional layer norm of the backbone. This is what `graph_crps` always did. | `--noise_dim` (32) | FGN |
| `mc` | MC dropout inside the MLPs of the GNNs, on the hidden features before the last linear layer. It is kept on at inference, and new masks are drawn for every member and step. No new parameters. | `--mc_dropout_p` (0.1; the paper found 0.05 best calibrated), `--mc_dropout_scope processor\|mesh\|all` | U-Cast, [arXiv:2604.09041](https://arxiv.org/abs/2604.09041) |
| `sdl` | Stochastic decomposition layers. Gaussian noise is added to the mesh node representations on every level of the hierarchy: `F + α · ε ⊙ Linear(z_l) ⊙ m_l`. ε is i.i.d. per node and channel, z_l is an independent latent per level, m_l is learned and α is fixed. Noise on the coarse levels becomes spatially coherent through message passing. | `--sdl_alpha` (0.235), `--sdl_points down\|init`, `--sdl_levels`, `--sdl_noise_dim`, `--sdl_m_init` | Schreck et al., [arXiv:2512.18815](https://arxiv.org/abs/2512.18815) |

More on the flags:

- **`fgn` off.** z is then the constant `--deterministic_noise_value`, exactly as in
  `graph_det`. So with `--stochasticity mc` or `sdl` alone, the conditional layer norms
  see the same input as during the deterministic pretraining.
- **`--mc_dropout_scope`:**
  - `processor` (default): the processor GNNs, i.e. the down/up sweeps through the
    hierarchy.
  - `mesh`: also the mesh init and read-out GNNs.
  - `all`: the whole backbone.
- **`--sdl_points`:**
  - `down` (default): after the down-GNN of each level in every processor down sweep,
    plus at the top level. This is the graph analogue of the paper's "after each
    decoder upsampling".
  - `init`: once per level after mesh init.
- **`--sdl_levels`:** e.g. `--sdl_levels 1 2` skips the finest level (level 0).

The SDL paper uses a grid U-Net, not a graph, so the mapping onto mesh levels is our
own. The `sdl` parameters are the only new ones. They are initialised fresh when a
`graph_det` checkpoint is loaded; every other weight must match the checkpoint.

## The starting checkpoint

The scripts start from the ¼° deterministic model:

```
saved_models/train-graph_det-3x1024-10_07_08-7783/last.ckpt
```

It is `--model graph_det`, `--hidden_dim 1024`, graph
`global_cluster_1_4_deg_15_refinement_4_levels`, 235 epochs, and was trained by
`seacast/seacast_finetune_muon.sh`.

To use another checkpoint, pass `--load <path>`. The architecture flags (`--graph`,
`--hidden_dim*`, `--processor_layers`, ...) must match it, otherwise loading fails.

## The transition from deterministic to probabilistic training

The deterministic weights were trained without noise. If all noise were switched on at
full strength in the first step, two things would happen together:

- full-strength `fgn` changes every layer norm;
- `mc` drops features in every block.

Combined with a fresh optimizer, the large early gradients can damage the pretrained
weights before the CRPS loss has a useful signal. `--scheduler crps_finetune`
therefore does the following:

1. **Noise ramp.** A factor r rises linearly, per step, from 0 to 1 over
   `--stoch_warmup_epochs` (default 2). It scales all sources:

   | Source | Scaled as |
   | --- | --- |
   | `fgn` | z = c + r ε |
   | `mc` | p = r · `mc_dropout_p` |
   | `sdl` | amplitude r · α |

   At r = 0 the model *is* the deterministic model (this is tested). Validation and
   test always use r = 1, so the logged metrics are those of the target model. r is
   logged as `stoch_ramp`.
2. **LR warm-up.** With `--crps_lr 1e-4` (the default), the learning rate warms up
   linearly from `crps_lr/10` = 1e-5 to 1e-4 over the first `--warmup_epochs` (default:
   `--stoch_warmup_epochs`). It is then cosine-annealed back to 1e-5 by the end of the
   fine-tuning. The optimizer state starts fresh; the deterministic optimizer state is
   not reused.
3. **Phases,** counted from the epoch after the loaded checkpoint (epoch 235 for the
   checkpoint above):

   | Phase | Unroll | Epochs | Notes |
   | --- | --- | --- | --- |
   | (0) | ar=1 | `--freeze_backbone_epochs` (default 0) | optional, only the `sdl` parameters are trained |
   | 1 | ar=1 | `--crps_epochs` (default 25) | the noise ramp runs during its first `--stoch_warmup_epochs` |
   | 2 | ar=2 | `--crps_ar2_epochs` (default 5, 0 to skip) | gradient checkpointing switches on automatically |

   Phase 0 is only worth trying if `sdl` runs go wrong early. The SDL paper found
   training all weights jointly better.

The loss is the almost fair CRPS with `--crps_alpha 0.95` over 2 ensemble members.
U-Cast found M = 4 barely better than M = 2.

## The scripts

There is one script per source of stochasticity. All three use distributed CRPS:
one ensemble member per GPU. Each lists every option of its source and of the CRPS
fine-tuning explicitly, with comments, at the current default values.

| Script | Source |
| --- | --- |
| [`seacast_crps_fgn.sh`](seacast_crps_fgn.sh) | `--stochasticity fgn` |
| [`seacast_crps_mc.sh`](seacast_crps_mc.sh) | `--stochasticity mc` |
| [`seacast_crps_sdl.sh`](seacast_crps_sdl.sh) | `--stochasticity sdl` |

```sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_fgn.sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_mc.sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_sdl.sh

# override any value on the command line (the last occurrence of a flag wins)
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_mc.sh \
    --mc_dropout_p 0.05

# several sources at once: start from the script of one of them; the options of
# the others keep their defaults unless given
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_sdl.sh \
    --stochasticity fgn mc sdl
```

## Memory modes

The ensemble members can be computed in three ways:

| Mode | Members | Memory per GPU | Speed (8 GPUs) |
| --- | --- | --- | --- |
| distributed (`--train_distr_crps`), used by the three scripts above | one per GPU; GPUs paired (0,1), (2,3), …, both GPUs of a pair get the same sample | ≈ deterministic model | ~1.5 h per ar=1 epoch |
| [`seacast_crps_finetune_ckpt.sh`](seacast_crps_finetune_ckpt.sh) (`--grad_checkpointing`) | both on every GPU, each rollout step recomputed in backward | ≈ deterministic model | ~2.1 h per ar=1 epoch |
| [`seacast_crps_finetune_local.sh`](seacast_crps_finetune_local.sh) | both on every GPU, no recomputation | ≈ 2× deterministic; **OOMs at hidden 1024** | fastest per step, for small models only |

Measured peak memory of one ar=1 training step, at hidden 256 on the 1° data, bf16,
relative to `graph_det`:

| Mode | Peak memory |
| --- | --- |
| 2 local members with checkpointing | 1.01–1.06× (`mc` costs the most) |
| 2 local members without checkpointing | 1.9–2.0× |

**Use the distributed mode for the 1024 model.**

- It needs an even number of GPUs; with an odd count it stops at start-up.
- Every GPU pair sees one sample per step, so an epoch has twice as many steps as with
  the other scripts. The total number of training samples per epoch is the same.
- The logged `train_loss` is the actual CRPS. The factor 2 that the distributed
  gradient needs is applied to the gradient only.

The `_ckpt` and `_local` scripts default to `--stochasticity mc`. Use the other
sources there with `--stochasticity ...`, and take their options from the
per-source scripts.

If a run runs out of memory, add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
the environment in `train.sh`. Reserved-but-unallocated memory has caused OOMs before.

## How many epochs

The default is 25 epochs at ar=1 and 5 at ar=2, about 11% of the 235 deterministic
epochs. For comparison, U-Cast fine-tunes for 8% of its pretraining epochs. The SDL
paper uses about 1.6% of its pretraining compute.

**What the estimate is based on.** The deterministic ¼° fine-tune took, on 8 GPUs at
batch size 1:

- 45 min per ar=1 epoch;
- 2 h 26 min per ar=2 epoch;
- about 4 min per validation.

**Estimated wall time on 8 GPUs** (validation every 5 epochs with 4 members, ~20 min
each):

| Script | 25 epochs ar=1 | 5 epochs ar=2 | Validation | Total |
| --- | --- | --- | --- | --- |
| distributed | ~38 h | ~25 h | ~2 h | **~65 h**, fits one 72 h job |
| checkpointed | ~53 h | ~25 h | ~2 h | **~80 h**, needs one resume |

Check the real time of the first epoch in the Slurm log. The progress bar shows
`[mm:ss<mm:ss, s/it]`, and `[graph_crps] peak GPU memory` is printed after every
epoch.

**When to change the number of epochs:**

- If `val_mean_ens_crps` has flattened well before epoch 25, use a smaller
  `--crps_epochs` for the next runs.
- If it is still falling clearly when phase 1 ends, the next runs can use more.
- When comparing sources, give all of them the same budget.

## Resuming and extending

The run continues where it stopped, also within the noise ramp. The optimizer state
and LR schedule are restored. To resume, pass the run's own checkpoint with the same
flags:

```sh
sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_mc.sh \
    --load saved_models/<run name>/last.ckpt
```

To queue the resume behind the running job:

```sh
sbatch --dependency=afterany:<jobid> training_scripts/train.sh \
    training_scripts/graph_crps/seacast_crps_mc.sh \
    --load saved_models/<run name>/last.ckpt
```

To extend phase 1, pass a larger `--crps_epochs` together with `--load` of the run's
own checkpoint. This only works before phase 2 has started. The cosine LR is then
re-stretched over the new length.

> **Note:** the `--load` path is fixed when the job is submitted, so a resume job
> queued behind a running job must point to `last.ckpt`, which is updated every
> epoch.

## Comparing the sources

Submit the three scripts (above), plus the combination if wanted. That is 3–4 jobs ×
8 GPUs × ~65 h.

To sweep the options of one source, put the variations in `RUNS` in
[`../launch_sweep.sh`](../launch_sweep.sh) and pass that source's script. The job
name then shows the sources, e.g. `SeaCast_crps_mc`.

```bash
RUNS=(
    "--stochasticity mc --mc_dropout_p 0.05"
    "--stochasticity mc --mc_dropout_p 0.1"
)
```

Check the commands first, then submit:

```sh
DRY_RUN=1 bash training_scripts/launch_sweep.sh training_scripts/graph_crps/seacast_crps_mc.sh
bash training_scripts/launch_sweep.sh training_scripts/graph_crps/seacast_crps_mc.sh
```

Worthwhile variations:

- `--mc_dropout_p 0.05`;
- `--sdl_levels 1 2`, i.e. no noise on the finest level.

## What to watch (wandb)

| Metric | Meaning |
| --- | --- |
| `stoch_ramp` | the noise ramp r: 0 → 1 over the first `--stoch_warmup_epochs` |
| `train_loss` | the CRPS of the 2 training members. It rises at first, as the noise is ramped up. |
| `val_mean_ens_crps` | the ensemble CRPS on validation (`--ensemble_size` members, normalized units, mean over lead times and variables). This is what `min_val_crps.ckpt` keeps the best of. |
| `val_mean_spsk_ratio` | the spread/skill ratio. It should approach 1: below 1 means under-dispersed (too little spread), above 1 over-dispersed. Per variable and lead time it is in the `val_spsk_ratio` plot. |
| `val_ens_crps`, `val_ens_rmse`, `val_spread` | plots per variable and lead time |
| `val_mean_loss` | the loss of a *single* member. It is not a good measure of the ensemble, and its checkpoint `min_val_loss.ckpt` favours low spread. |

If the spread stays too small, raise `--mc_dropout_p` or `--sdl_alpha`, or add another
source. U-Cast reports slight under-dispersion from MC dropout alone. The SDL paper
shows that scaling the noise of a trained model at inference also changes the spread.

## Evaluation

Run the test set with more members. Pass the **same** `--stochasticity` (and
`--mc_dropout_*` / `--sdl_*`) flags as in training: they are read from the command
line, not from the checkpoint.

```sh
srun python -m neural_lam.train_model \
    <the model flags of the script> \
    --model graph_crps --stochasticity mc \
    --eval test --ensemble_size 10 --ar_steps_eval 10 \
    --load saved_models/<run name>/min_val_crps.ckpt
# add --save_ensemble_zarr / --save_forecasts to store the ensembles
```

For a baseline, the deterministic model evaluated with `--model graph_crps
--stochasticity` (nothing after it) is an ensemble of identical members. Its CRPS
equals its MAE.

## Troubleshooting

- **Spread exactly 0 with `--train_distr_crps`.** The two GPUs of a pair must draw
  different noise. This comes from the per-rank reseed in `ARProbModel.on_fit_start`;
  check that it has not been removed.
- **"Can not use train_distr_crps with odd number of GPUs".** Use an even number of
  GPUs, or the `_ckpt` script.
- **Out of memory.**
  - First, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Then switch to the distributed or checkpointed script.
  - At ar=2, checkpointing is already on.
- **Error loading the checkpoint, about missing or unexpected keys.**
  - The architecture flags do not match the checkpoint.
  - Or a run trained with `sdl` is loaded without `--stochasticity sdl`.
- **`--model crps` is not a model.** Use `--model graph_crps`. The old scripts in this
  folder have been fixed.

Tests: `python tests/test_graph_crps_stochasticity.py [unit]`. A small end-to-end run
of all modes on 2 devel GPUs: `bash training_scripts/graph_crps/smoke_test.sh`.
