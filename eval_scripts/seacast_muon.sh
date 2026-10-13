#!/bin/sh
#SBATCH -t 72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=erila85@liu.se
#SBATCH --output ./slurm_logs/%A_%x.out
#SBATCH -J Eval_SeaCast
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=4
# --dependency=afterany:15131
# NOTE: Makes this job run after the job with ID JOB_ID completes, regardless of its exit status.

source ~/.bashrc
module load Miniforge3/25.3.1-0
mamba activate bsf

cd /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting
wandb online

srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_4_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model graph_det \
    --graph global_cluster_1_4_deg_15_refinement_4_levels \
    --hidden_dim 1024 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 1 \
    --scheduler finetune_deterministic \
    --val_interval 5 \
    --ar_steps_eval 15 \
    --val_steps_to_log 1 4 15 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --optimizer "muon_flat" \
    --eval test \
    --n_example_pred 1 \
    --load /proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting/saved_models/train-graph_det-3x1024-10_07_08-7783/last.ckpt