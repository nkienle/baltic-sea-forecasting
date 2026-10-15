# Flow matching (FM) model with a positional time embedding, 1 deg. Optional:
# a teacher for the FlowMap (--fmap_teacher_ckpt, --fmap_init_from_teacher,
# --fmap_offdiag_target teacher). Same backbone settings as the FlowMap scripts.
#   sbatch training_scripts/train.sh training_scripts/graph_flow/fm_positional.sh
srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model FM \
    --epochs 175 \
    --graph global_cluster_1_deg_15_refinement_4_levels \
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
    --lr 0.0005 \
    --min_lr 0.00001 \
    --grad_clip 1.0 \
    --ensemble_size 5 \
    --n_example_pred 1 \
    --val_interval 5 \
    --ar_steps_eval 1 \
    --val_steps_to_log 1 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --optimizer "muon_flat" \
    "$@"