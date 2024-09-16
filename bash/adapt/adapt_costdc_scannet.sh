#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=56423

for L in 3e-3
do
    for I in 1
    do
        for S in 2.0
        do
            for C in 0.05 0.1 0.2 0.3 0.4 0.5
            do
                for R in 0
                do            
                    for T in 1 2 3 4 5
                    do
                        python src/tta.py \
                        --train_image_path testing/scannet/scannet_test_image_corner.txt \
                        --train_sparse_depth_path testing/scannet/scannet_test_sparse_depth_corner.txt \
                        --train_ground_truth_path testing/scannet/scannet_test_ground_truth_corner.txt \
                        --val_image_path testing/scannet/scannet_test_image_corner.txt \
                        --val_sparse_depth_path testing/scannet/scannet_test_sparse_depth_corner.txt \
                        --val_ground_truth_path testing/scannet/scannet_test_ground_truth_corner.txt \
                        --n_batch 36 \
                        --n_height 320 \
                        --n_width 400 \
                        --normalized_image_range 0.485 0.456 0.406 0.229 0.224 0.225 \
                        --loss_type 'adapt_meta_selfsup_seq_ema_reverse' \
                        --prepare_mode 'meta_selfsup_seq_1layer_ema' \
                        --adapt_mode 'meta_bn' \
                        --model_name 'costdcnet' \
                        --learning_rates $L \
                        --learning_schedule 10000 \
                        --inner_iter $I \
                        --augmentation_probabilities 1.00 \
                        --augmentation_schedule -1 \
                        --augmentation_probabilities 1.00 \
                        --augmentation_schedule -1 \
                        --augmentation_random_brightness -1 -1 \
                        --augmentation_random_contrast -1 -1 \
                        --augmentation_random_saturation -1 -1 \
                        --augmentation_random_crop_type horizontal vertical \
                        --augmentation_random_crop_to_shape -1 -1 \
                        --augmentation_random_flip_type horizontal \
                        --augmentation_random_rotate_max 5 \
                        --augmentation_random_resize_and_crop -1 -1 \
                        --min_predict_depth 0.1 \
                        --max_predict_depth 8.0 \
                        --min_evaluate_depth 0.2 \
                        --max_evaluate_depth 5.0 \
                        --w_loss_cos $C \
                        --w_loss_sparse_depth 1.0 \
                        --w_loss_smoothness $S \
                        --device gpu \
                        --n_thread 8 \
                        --checkpoint_path model_adapt_meta/scannet/costdcnet-selfsup-1layer-reverse-adapt-lr-$L-s-$S-c-$C-r-$R-iter-$I-trial-$T/ \
                        --restore_path_model checkpoints/costdcnet_prepared_indoor.pth \
                        --validation_start_step 3000 \
                        --n_step_per_checkpoint 3000 \
                        --n_step_per_summary 10
                    done
                done
            done
        done
    done
done
