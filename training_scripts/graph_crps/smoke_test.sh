#!/bin/bash
# Smoke test of the probabilistic fine-tuning (--scheduler crps_finetune) with a
# tiny model at 1 deg, on 2 GPUs (2 devel A100s are enough):
#   interactive --gpus=2 --reservation=devel -t 01:00:00
#   bash training_scripts/graph_crps/smoke_test.sh
# 1) trains a tiny graph_det for 1 epoch, 2) CRPS-fine-tunes it in every
# memory mode and with every source of stochasticity, resumes and extends a
# run, and 3) prints the epochs and peak memory of each run.
# Checkpoints go to saved_models/, wandb runs are offline.
set -euo pipefail
cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting
PY=${PY:-/proj/berzelius-2022-164/users/x_erila/.conda/envs/bsf/bin/python}
export WANDB_MODE=offline
export PYTHONPATH=$(pwd)

SMALL="--config_path data/global_ocean_1_density.yaml
    --graph global_cluster_1_deg_20_refinement_4_levels
    --hidden_dim 64 --hidden_dim_grid 32 --hidden_dim_mesh_nodes 32
    --hidden_dim_edge 16 --processor_layers 2 --input_steps 2 --batch_size 1
    --num_workers 2 --limit_train_batches 20 --limit_val_batches 4
    --val_interval 1 --ar_steps_eval 2 --val_steps_to_log 1 2
    --num_sanity_val_steps 0 --ensemble_size 4 --n_example_pred 1
    --optimizer muon_flat"
CRPS="--model graph_crps --scheduler crps_finetune --stoch_warmup_epochs 1"

latest() { ls -td saved_models/train-"$1"-* | head -1; }
LOGDIR=${LOGDIR:-saved_models/smoke_logs_crps}; mkdir -p "$LOGDIR"; N=0
run() {
    N=$((N + 1)); echo "=== [$N] $*"
    # train_model's main is wrapped in loguru's logger.catch, which logs the
    # exception and exits 0, so also look for its message
    if ! "$PY" -m neural_lam.train_model "$@" > "$LOGDIR/$N.log" 2>&1 \
        || grep -q "An error has been caught" "$LOGDIR/$N.log"; then
        grep -v "Warning" "$LOGDIR/$N.log" | tail -40
        echo "FAILED, full log: $LOGDIR/$N.log"; exit 1
    fi
    grep -E "peak GPU memory|crps_finetune\]|GraphCRPS:" "$LOGDIR/$N.log" \
        | sort -u | tail -12 || true
}
ONE_GPU="--devices 0"

# 1) Tiny graph_det, 1 epoch (the deterministic model to start from)
run $SMALL $ONE_GPU --model graph_det --epochs 1
DET=$(latest graph_det)/last.ckpt

# 2) Distributed CRPS (2 GPUs, one member each), all three sources:
#    1 epoch ar=1 only, then resume and extend to 2 epochs ar=1 + 1 epoch ar=2
run $SMALL $CRPS --stochasticity fgn mc sdl --train_distr_crps \
    --crps_epochs 1 --crps_ar2_epochs 0 --load "$DET"
P=$(latest graph_crps)/last.ckpt
run $SMALL $CRPS --stochasticity fgn mc sdl --train_distr_crps \
    --crps_epochs 2 --crps_ar2_epochs 1 --load "$P"
DISTR=$(latest graph_crps)

# 3) Both members on one GPU, with and without gradient checkpointing
run $SMALL $ONE_GPU $CRPS --stochasticity sdl --grad_checkpointing \
    --crps_epochs 1 --crps_ar2_epochs 1 --load "$DET"
run $SMALL $ONE_GPU $CRPS --stochasticity mc \
    --crps_epochs 1 --crps_ar2_epochs 0 --load "$DET"

# 4) Frozen backbone first (only the SDL parameters), fgn + sdl
run $SMALL $ONE_GPU $CRPS --stochasticity fgn sdl --freeze_backbone_epochs 1 \
    --crps_epochs 1 --crps_ar2_epochs 0 --load "$DET"

# Epoch offsets of the checkpoints
"$PY" - "$DET" "$DISTR/last.ckpt" "$DISTR/min_val_crps.ckpt" <<'PYEOF'
import sys, torch
for path in sys.argv[1:]:
    c = torch.load(path, map_location="cpu", weights_only=False)
    a = c["hyper_parameters"]["args"]
    print(path, "epoch", c["epoch"], "crps start",
          c.get("crps_stage_start_epoch"), "scheduler", a.scheduler,
          "stochasticity", getattr(a, "stochasticity", None))
PYEOF
echo "Smoke test passed"
