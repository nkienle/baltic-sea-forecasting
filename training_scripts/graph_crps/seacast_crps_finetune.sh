srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_4_density.yaml \
    --num_workers 2 \
    --precision bf16-mixed \
    --model graph_crps \
    --graph global_cluster_1_4_deg_20_refinement_4_levels \
    --hidden_dim 256 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 3 \
    --scheduler finetune_deterministic \
    --val_interval 5 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    --optimizer "muon_flat" \
    --load CHECKPOINT

# Different graphs:
# global_cluster_1_4_deg_20_refinement_4_levels
# global_cluster_1_4_deg_15_refinement_4_levels
# global_cluster_1_4_deg_10_refinement_4_levels

# Graph in the paper: global_hierarchical_1_deg_6_splits_3_levels