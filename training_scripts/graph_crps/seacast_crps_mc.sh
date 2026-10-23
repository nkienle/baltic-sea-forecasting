# Probabilistic (CRPS) fine-tuning of the 1/4 deg SeaCast graph_det model with
# MC dropout (U-Cast, arXiv:2604.09041): dropout on the hidden features inside
# the GNN MLPs, kept on at inference, new masks for every member and step.
# Distributed CRPS: one ensemble member per GPU, GPUs paired (0,1), (2,3), ...
# -> needs an even number of GPUs. ~65 h on 8 GPUs (25 + 5 epochs).
# Run instructions: training_scripts/graph_crps/README.md
#   sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_mc.sh
# Any flags passed are appended last and override the values below.
SEACAST_1_4deg_DET="/proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting/saved_models/train-graph_det-3x1024-10_07_08-7783/last.ckpt"
# Resume the unfinished run (job 155476, time limit).
# To start a new fine-tuning from graph_det instead, use
# --load "$SEACAST_1_4deg_DET" and drop --restore_opt.
RESUME_CKPT="/proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting/saved_models/train-graph_crps-3x1024-10_15_11-1611/last.ckpt"

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
    --seed 42 \
    `# --- Source of stochasticity -------------------------------------` \
    --stochasticity mc \
    `# Dropout rate (U-Cast: 0.1; 0.05 was best calibrated)` \
    --mc_dropout_p 0.1 \
    `# Where: processor (down/up sweep GNNs) | mesh (+ mesh init/read-out)` \
    `# | all (whole backbone)` \
    --mc_dropout_scope processor \
    `# Without fgn, the noise vector of the conditional layer norms is the` \
    `# constant deterministic_noise_value, as in graph_det. The embedding` \
    `# must match the checkpoint.` \
    --noise_embedding linear \
    --noise_dim 32 \
    --channel_mult_emb 2 \
    --deterministic_noise_value 0.0 \
    `# --- Transition from the deterministic model ----------------------` \
    `# Noise ramp r: 0 (= graph_det) -> 1 over the first epochs` \
    --stoch_warmup_epochs 2 \
    `# LR: crps_lr/10 -> crps_lr over warmup_epochs, cosine back to crps_lr/10` \
    --crps_lr 1e-4 \
    --warmup_epochs 2 \
    `# --- Schedule (--scheduler crps_finetune) -------------------------` \
    --scheduler crps_finetune \
    --crps_epochs 25 \
    --crps_ar2_epochs 5 \
    `# --- CRPS loss ----------------------------------------------------` \
    `# almost fair CRPS, 1 = fair CRPS` \
    --crps_alpha 0.95 \
    `# one member per GPU, GPUs paired (2 members per sample)` \
    --train_distr_crps \
    `# --- Validation ---------------------------------------------------` \
    `# members for val_mean_ens_crps / spread-skill (min_val_crps.ckpt)` \
    --ensemble_size 4 \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --n_example_pred 1 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --load "$RESUME_CKPT" \
    --restore_opt \
    "$@"

# Resume a run that hit the time limit (same flags, its own last.ckpt):
#   sbatch training_scripts/train.sh training_scripts/graph_crps/seacast_crps_mc.sh \
#       --load saved_models/<run name>/last.ckpt
# Combine sources by overriding, e.g. --stochasticity fgn mc sdl (then also see
# the fgn / sdl options in seacast_crps_fgn.sh / seacast_crps_sdl.sh).
