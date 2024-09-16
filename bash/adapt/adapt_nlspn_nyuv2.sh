#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=56423

for L in 4e-3
do
    for S in 3.0
    do
        for C in 0.01 0.05 0.1 0.2 0.3 0.5 1.0
        do
            for Z in 1.0
            do
                for I in 1
                do
                    python src/tta.py \
                    --train_image_path figure_dir/image/nyu_training_scene_home_office_0005.txt \
                    --train_sparse_depth_path figure_dir/sparse_depth/nyu_training_scene_home_office_0005.txt \
                    --train_intrinsics_path figure_dir/intrinsics/nyu_training_scene_home_office_0005.txt \
                    --train_ground_truth_path figure_dir/ground_truth/nyu_training_scene_home_office_0005.txt \
                    --val_image_path figure_dir/image/nyu_training_scene_home_office_0005.txt \
                    --val_sparse_depth_path figure_dir/sparse_depth/nyu_training_scene_home_office_0005.txt \
                    --val_intrinsics_path figure_dir/intrinsics/nyu_training_scene_home_office_0005.txt \
                    --val_ground_truth_path figure_dir/ground_truth/nyu_training_scene_home_office_0005.txt \
                    --n_batch 16 \
                    --n_height 224 \
                    --n_width 320 \
                    --normalized_image_range 0.485 0.456 0.406 0.229 0.224 0.225 \
                    --loss_type 'adapt_meta_selfsup_seq_ema_reverse' \
                    --prepare_mode 'meta_selfsup_seq_1layer_ema' \
                    --adapt_mode 'meta_bn' \
                    --model_name 'nlspn' \
                    --learning_rates $L \
                    --learning_schedule 10000 \
                    --inner_iter 1 \
                    --augmentation_probabilities 1.00 \
                    --augmentation_schedule -1 \
                    --augmentation_random_brightness -1 -1 \
                    --augmentation_random_contrast -1 -1 \
                    --augmentation_random_saturation -1 -1 \
                    --augmentation_random_crop_type horizontal \
                    --augmentation_random_crop_to_shape -1 -1 \
                    --augmentation_random_flip_type horizontal \
                    --augmentation_random_rotate_max -1 \
                    --augmentation_random_resize_and_crop -1 -1 \
                    --max_input_depth 8.0 \
                    --min_predict_depth 0.1 \
                    --max_predict_depth 8.0 \
                    --min_evaluate_depth 0.2 \
                    --max_evaluate_depth 5.0 \
                    --w_loss_cos $C \
                    --w_loss_sparse_depth $Z \
                    --w_loss_smoothness $S \
                    --w_loss_robust 0.0 \
                    --device gpu \
                    --n_thread 8 \
                    --checkpoint_path model_adapt_meta/nyu_v2/nlspn-selfsup-1layers-reverse-adapt-lr-$L-s-$S-c-$C-z-$Z-iter-$I/ \
                    --restore_path_model checkpoints/nlspn_prepared_indoor.pth \
                    --validation_start_step 3000 \
                    --n_step_per_checkpoint 3000 \
                    --n_step_per_summary 10
                done
            done
        done
    done
done


