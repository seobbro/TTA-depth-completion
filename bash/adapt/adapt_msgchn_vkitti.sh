#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=52123

for L in 1e-4 2e-4 5e-4 1e-3
do
    for S in 1.0 2.0 3.0 4.0 5.0
    do
        for C in 1e-3 5e-3 0.01 0.05 0.1 0.2 0.3 0.4 0.5 1.0
        do
            for R in 0.0
            do
                python src/tta.py \
                --train_image_path testing/virtual_kitti/vkitti_test_image.txt \
                --train_sparse_depth_path testing/virtual_kitti/vkitti_test_sparse_depth.txt \
                --train_ground_truth_path testing/virtual_kitti/vkitti_test_ground_truth.txt \
                --val_image_path testing/virtual_kitti/vkitti_test_image.txt \
                --val_sparse_depth_path testing/virtual_kitti/vkitti_test_sparse_depth.txt \
                --val_ground_truth_path testing/virtual_kitti/vkitti_test_ground_truth.txt \
                --n_batch 16 \
                --n_height 240 \
                --n_width 1216 \
                --normalized_image_range 0 1 \
                --loss_type 'adapt_meta_selfsup_seq_ema_reverse' \
                --prepare_mode 'meta_selfsup_seq_2layers_ema' \
                --adapt_mode 'meta' \
                --model_name 'msg_chn' \
                --learning_rates $L \
                --learning_schedule 1000 \
                --inner_iter 1 \
                --augmentation_probabilities 1.00 \
                --augmentation_schedule -1 \
                --augmentation_random_brightness 0.6 1.4 \
                --augmentation_random_contrast 0.6 1.4 \
                --augmentation_random_saturation 0.6 1.4 \
                --augmentation_random_crop_type horizontal bottom \
                --augmentation_random_crop_to_shape -1 -1 \
                --augmentation_random_flip_type horizontal \
                --augmentation_random_rotate_max 5 \
                --augmentation_random_resize_and_crop 1.0 1.5 \
                --max_input_depth 80.0 \
                --min_predict_depth 0.0 \
                --max_predict_depth 80.0 \
                --min_evaluate_depth 0.0 \
                --max_evaluate_depth 80.0 \
                --w_loss_cos $C \
                --w_loss_sparse_depth 1.0 \
                --w_loss_smoothness $S \
                --w_loss_robust $R \
                --device gpu \
                --n_thread 8 \
                --checkpoint_path model_adapt_meta/msgchn-selfsup-2layers-reverse-lr-$L-s-$S-c-$C-r-$R/ \
                --restore_path_model checkpoints/msgchn_prepared_outdoor.pth \
                --validation_start_step 3000 \
                --n_step_per_checkpoint 3000 \
                --n_step_per_summary 10
            done
        done
    done
done