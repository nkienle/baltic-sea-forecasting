#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out
#SBATCH -J Eval_CRPS_MC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=1
# --dependency=afterany:15131
# NOTE: Makes this job run after the job with ID JOB_ID completes, regardless of its exit status.
# 1/4 deg graph_crps (MC dropout), 20 ensemble members, test set.
# The checkpoint can be given as the first argument (e.g. the last.ckpt of the
# resumed run, which is saved in a new saved_models/ directory):
#   sbatch eval_scripts/seacast_crps_mc.sh saved_models/<run name>/last.ckpt
CKPT="${1:-/proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting/saved_models/train-graph_crps-3x1024-10_19_07-1235/last.ckpt}"

source ~/.bashrc
module load Miniforge3/25.3.1-0
mamba activate bsf

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting
wandb online

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
    --stochasticity mc \
    --mc_dropout_p 0.1 \
    --mc_dropout_scope processor \
    --noise_embedding linear \
    --noise_dim 32 \
    --channel_mult_emb 2 \
    --deterministic_noise_value 0.0 \
    --ensemble_size 20 \
    --val_interval 5 \
    --ar_steps_eval 15 \
    --val_steps_to_log 1 4 15 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --optimizer "muon_flat" \
    --eval test \
    --n_example_pred 1 \
    --limit_test_batches 1 \
    --load "$CKPT"
