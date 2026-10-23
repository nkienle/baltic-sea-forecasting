# FM (positional time embedding) fine-tuning at 1/4 deg, continuing the 1 deg
# fm_positional.sh run. Same LR schedule as seacast_finetune_muon.sh
# (finetune_deterministic: 5 ep warmup 1e-5 -> 1e-4, cosine to 1e-5, 60 epochs
# after the 175 pretraining epochs), but --scheduler finetune_ar1 trains all 60
# epochs at ar=1 (no ar=2 phase, too expensive for FM).
#   sbatch training_scripts/train.sh training_scripts/graph_flow/fm_positional_finetune.sh
# The --load checkpoint is the finished 1 deg run (job 159495, 175/175 epochs).
# To resume this fine-tuning after a time limit, keep the optimizer state and
# LR schedule (without --restore_opt they are reset):
#   ... fm_positional_finetune.sh --load saved_models/<this run>/last.ckpt --restore_opt
FM_1deg_Pretrained="/proj/berzelius-2022-164/users/x_erila/baltic-sea-forecasting/saved_models/train-FM-3x1024-10_19_13-2281/last.ckpt"

srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_4_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model FM \
    --scheduler finetune_ar1 \
    --graph global_cluster_1_4_deg_15_refinement_4_levels \
    --hidden_dim 1024 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 1 \
    --noise_embedding positional \
    --pred_residual \
    --sampler heun \
    --sampler_steps 20 \
    --grad_clip 1.0 \
    --ensemble_size 5 \
    --n_example_pred 1 \
    --val_interval 5 \
    --ar_steps_eval 1 \
    --val_steps_to_log 1 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --optimizer "muon_flat" \
    --load "$FM_1deg_Pretrained" \
    "$@"
