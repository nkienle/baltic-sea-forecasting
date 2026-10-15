# FlowMap part 3/3: 0.25 deg, autoregressive training (ar=2) with the almost
# fair CRPS of 2 ensemble members, each step one flow-map jump X_{0,1}(z),
# plus the flow-map loss on the first step (weight --fmap_crps_map_weight;
# --fmap_objective crps drops it). Starts from the last checkpoint of part 2:
#   sbatch training_scripts/train.sh training_scripts/graph_flow_map/flow_map_crps.sh \
#       --load saved_models/<part-2 run>/last.ckpt
srun python -m neural_lam.train_model \
    --config_path data/global_ocean_1_4_density.yaml \
    --num_workers 2 \
    --precision 32 \
    --model FlowMap \
    --scheduler flow_map_crps \
    --epochs 20 \
    --ar_steps_train 2 \
    --fmap_objective map+crps \
    --fmap_crps_map_weight 0.5 \
    --n_ens_train 2 \
    --crps_alpha 0.95 \
    --graph global_cluster_1_4_deg_20_refinement_4_levels \
    --hidden_dim 256 \
    --hidden_dim_grid 128 \
    --hidden_dim_mesh_nodes 128 \
    --hidden_dim_edge 32 \
    --processor_layers 3 \
    --input_steps 2 \
    --batch_size 3 \
    --noise_embedding positional \
    --pred_residual \
    --lr 0.00003 \
    --min_lr 0.000001 \
    --grad_clip 1.0 \
    --fmap_loss lsd \
    --fmap_ema \
    --fmap_ema_decay 0.999 \
    --fmap_val_sample_steps_extra 4 \
    --ensemble_size 5 \
    --n_example_pred 1 \
    --val_interval 2 \
    --ar_steps_eval 4 \
    --val_steps_to_log 1 4 \
    --num_sanity_val_steps 0 \
    --num_nodes 1 \
    "$@"
