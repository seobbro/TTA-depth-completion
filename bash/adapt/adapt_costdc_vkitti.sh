#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=56423
for L in 5e-3
do
    for S in 3.0
    do
        for C in  0.04
        do
            for R in 0
            do
                for I in 1
                do
                    for T in 3 4
                    do
                    python src/tta.py \
                        --train_image_path testing/virtual_kitti/vkitti_test_image-fog.txt \
                        --train_sparse_depth_path testing/virtual_kitti/vkitti_test_sparse_depth-fog.txt \
                        --train_ground_truth_path testing/virtual_kitti/vkitti_test_ground_truth-fog.txt \
                        --val_image_path testing/virtual_kitti/vkitti_test_image-fog.txt \
                        --val_sparse_depth_path testing/virtual_kitti/vkitti_test_sparse_depth-fog.txt \
                        --val_ground_truth_path testing/virtual_kitti/vkitti_test_ground_truth-fog.txt \
                        --n_batch 12 \
                        --n_height 240 \
                        --n_width 1216 \
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
                        --augmentation_random_crop_type horizontal bottom \
                        --augmentation_random_crop_to_shape -1 -1 \
                        --augmentation_random_flip_type horizontal \
                        --augmentation_random_rotate_max -1 \
                        --augmentation_random_resize_and_crop -1 -1 \
                        --max_input_depth 80.0 \
                        --min_predict_depth 0.0 \
                        --max_predict_depth 90.0 \
                        --min_evaluate_depth 0.0 \
                        --max_evaluate_depth 80.0 \
                        --w_loss_cos $C \
                        --w_loss_sparse_depth 1.0 \
                        --w_loss_smoothness $S \
                        --device gpu \
                        --n_thread 4 \
                        --checkpoint_path model_adapt_meta/costdcnet_vkitti-selfsup-1layer-reverse-adapt-lr-$L-s-$S-c-$C-r-$R-iter-$I-trial-$T/ \
                        --restore_path_model checkpoints/costdcnet_prepared_outdoor.pth \
                        --validation_start_step 3000 \
                        --n_step_per_checkpoint 3000 \
                        --n_step_per_summary 10
                    done
                done
            done
        done
    done
done