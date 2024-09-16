#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=56423

for L in 3e-3
do
    for S in 8.5 9.0 9.5 10.0
    do
        for C in 0.2
        do
            for Z in 1
            do
                for I in 1 2
                do
                    python src/tta.py \
                   --train_image_path validation/scenenet/scenenet_val_image_corner-subset.txt \
                    --train_sparse_depth_path validation/scenenet/scenenet_val_sparse_depth_corner-subset.txt \
                    --train_ground_truth_path validation/scenenet/scenenet_val_ground_truth_corner-subset.txt \
                    --val_image_path validation/scenenet/scenenet_val_image_corner-subset.txt \
                    --val_sparse_depth_path validation/scenenet/scenenet_val_sparse_depth_corner-subset.txt \
                    --val_ground_truth_path validation/scenenet/scenenet_val_ground_truth_corner-subset.txt \
                    --n_batch 16 \
                    --n_height 228 \
                    --n_width 304 \
                    --normalized_image_range 0 1 \
                    --loss_type 'adapt_meta_selfsup_seq_ema_reverse' \
                    --prepare_mode 'meta_selfsup_seq_1layer_ema' \
                    --adapt_mode 'meta' \
                    --model_name 'msg_chn' \
                    --learning_rates $L \
                    --learning_schedule 10000 \
                    --inner_iter 3 \
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
                    --max_input_depth 8.0 \
                    --min_predict_depth 0.1 \
                    --max_predict_depth 8.0 \
                    --min_evaluate_depth 0.2 \
                    --max_evaluate_depth 5.0 \
                    --w_loss_cos $C \
                    --w_loss_sparse_depth $Z \
                    --w_loss_smoothness $S \
                    --device gpu \
                    --n_thread 8 \
                    --checkpoint_path model_adapt_meta/scenenet/msg_chn-selfsup-1layer-reverse-adapt-lr-$L-s-$S-c-$C-z-$Z-iter-$I/ \
                    --restore_path_model model_save/indoor/msg_chn-selfsup-1layer-reverse-1e-3_head/checkpoints/model-00012264.pth \
                    --validation_start_step 3000 \
                    --n_step_per_checkpoint 3000 \
                    --n_step_per_summary 10
                done
            done
        done
    done
done