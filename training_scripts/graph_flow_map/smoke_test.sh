#!/bin/bash
# Smoke test of FM + the three FlowMap parts with a tiny model, on one GPU
# (fits a 40 GB devel A100). Run inside an interactive allocation:
#   interactive --gpus=1 --reservation=devel -t 02:00:00
#   bash training_scripts/graph_flow_map/smoke_test.sh
# Checkpoints go to saved_models/, wandb runs are offline.
set -euo pipefail
cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting
PY=${PY:-/proj/berzelius-2022-164/users/x_erila/.conda/envs/bsf/bin/python}
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SMALL="--hidden_dim 64 --hidden_dim_grid 32 --hidden_dim_mesh_nodes 32
    --hidden_dim_edge 16 --processor_layers 2 --input_steps 2 --batch_size 1
    --num_workers 2 --noise_embedding positional --pred_residual
    --limit_train_batches 20 --limit_val_batches 2 --val_interval 1
    --ar_steps_eval 2 --val_steps_to_log 1 2 --num_sanity_val_steps 0
    --ensemble_size 5 --n_example_pred 1 --grad_clip 1.0 --precision 32"
DEG1="--config_path data/global_ocean_1_density.yaml
    --graph global_cluster_1_deg_20_refinement_4_levels"
DEG4="--config_path data/global_ocean_1_4_density.yaml
    --graph global_cluster_1_4_deg_20_refinement_4_levels"

latest() { ls -td saved_models/train-"$1"-* | head -1; }
LOGDIR=${LOGDIR:-saved_models/smoke_logs}; mkdir -p "$LOGDIR"; N=0
run() {
    N=$((N + 1)); echo "=== [$N] $*"
    # train_model's main is wrapped in loguru's logger.catch, which logs the
    # exception and exits 0, so also look for its message
    if ! "$PY" -m neural_lam.train_model "$@" > "$LOGDIR/$N.log" 2>&1 \
        || grep -q "An error has been caught" "$LOGDIR/$N.log"; then
        grep -v "Warning" "$LOGDIR/$N.log" | tail -40; echo "FAILED, full log: $LOGDIR/$N.log"; exit 1
    fi
    grep -E "peak GPU memory|epochs [0-9]+ ->|resuming|starting a new part|teacher|initialised" "$LOGDIR/$N.log" | tail -6 || true
}

# 1) FM teacher, 1 epoch
run $DEG1 $SMALL --model FM --epochs 1 --sampler heun --sampler_steps 4
FM_CKPT=$(latest FM)/last.ckpt

# 2) Part 1 (1 deg), teacher init, psd + split pairs (batch 1 -> both)
run $DEG1 $SMALL --model FlowMap --scheduler flow_map_pretrain --epochs 1 \
    --fmap_ema --fmap_val_sample_steps_extra 2 --fmap_loss psd \
    --fmap_teacher_ckpt "$FM_CKPT" --fmap_init_from_teacher \
    --lr 5e-4 --min_lr 1e-5
# Part 1 again with lsd (the default), from scratch
run $DEG1 $SMALL --model FlowMap --scheduler flow_map_pretrain --epochs 1 \
    --fmap_ema --fmap_val_sample_steps_extra 2 --lr 5e-4 --min_lr 1e-5
P1=$(latest FlowMap)/last.ckpt

# 3) Part 2 (0.25 deg) from part 1, then extend it by one epoch
run $DEG4 $SMALL --model FlowMap --scheduler flow_map_finetune --epochs 1 \
    --fmap_ema --load "$P1" --lr 1e-4 --min_lr 1e-5
P2=$(latest FlowMap)/last.ckpt
run $DEG4 $SMALL --model FlowMap --scheduler flow_map_finetune --epochs 2 \
    --fmap_ema --load "$P2" --lr 1e-4 --min_lr 1e-5
P2=$(latest FlowMap)/last.ckpt

# 4) Part 3 (0.25 deg, ar=2, map+crps) from part 2
run $DEG4 $SMALL --model FlowMap --scheduler flow_map_crps --epochs 1 \
    --ar_steps_train 2 --fmap_objective map+crps --fmap_ema \
    --load "$P2" --lr 3e-5 --min_lr 1e-6
P3=$(latest FlowMap)/last.ckpt

# Epoch offsets and EMA in the checkpoints
"$PY" - "$P1" "$P2" "$P3" <<'PYEOF'
import sys, torch
for path in sys.argv[1:]:
    c = torch.load(path, map_location="cpu", weights_only=False)
    print(path, "epoch", c["epoch"], "stage start", c.get("fmap_stage_start_epoch"),
          "scheduler", c["hyper_parameters"]["args"].scheduler,
          "ema", "fmap_ema" in c, "ema_n", c.get("fmap_ema_n"))
PYEOF
