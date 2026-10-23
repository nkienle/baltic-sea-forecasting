# Probabilistic (CRPS) fine-tuning, both ensemble members on every GPU without
# gradient checkpointing: the fastest per step, but about twice the memory of
# the deterministic model. Only for small models -- it OOMs at hidden_dim 1024.
# Override the model and checkpoint, e.g.
#   ... seacast_crps_finetune_local.sh --hidden_dim 256 \
#       --load saved_models/<small graph_det run>/last.ckpt
# Run instructions: training_scripts/graph_crps/README.md
#   sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_finetune_local.sh
# Any flags passed are appended last and override the defaults below, e.g. the
# source(s) of stochasticity: --stochasticity mc | sdl | fgn | fgn mc sdl
SEACAST_1_4deg_DET="/proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting/saved_models/train-graph_det-3x1024-10_07_08-7783/last.ckpt"

srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_4_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model graph_crps \
    --graph global_cluster_1_4_deg_15_refinement_4_levels \
    --hidden_dim 1024 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 1 \
    --optimizer "muon_flat" \
    --scheduler crps_finetune \
    --stochasticity mc \
    --stoch_warmup_epochs 2 \
    --crps_epochs 25 \
    --crps_ar2_epochs 5 \
    --crps_lr 1e-4 \
    --crps_alpha 0.95 \
    --ensemble_size 4 \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --load "$SEACAST_1_4deg_DET" \
    "$@"

# Resume a run that hit the time limit (same flags, its own last.ckpt):
#   sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_finetune_local.sh \
#       --load saved_models/<run name>/last.ckpt
