'''
Authors:
Hyoungseob Park <hyoungseob.park@yale.edu>
Anjali Gupta <anjali.gupta@yale.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
H. Park, A. Gupta, and A. Wong. Test-Time Adaptation for Depth Completion.
https://arxiv.org/abs/2402.03312

@inproceedings{park2024test,
  title={Test-Time Adaptation for Depth Completion},
  author={Park, Hyoungseob and Gupta, Anjali and Wong, Alex},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20519--20529},
  year={2024}
}
'''
import os, time
import numpy as np
import torch
import data_utils, datasets, eval_utils
from log_utils import log
from external_model_adapt import ExternalModel_Adapt
from transforms import Transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
import torch.cuda.amp as amp

def prepare_ddp(rank,
          ngpus_per_node,
          train_image_path,
          train_sparse_depth_path,
          train_intrinsics_path,
          train_ground_truth_path,
          val_image_path,
          val_sparse_depth_path,
          val_ground_truth_path,
          val_intrinsics_path,
          # Batch settings
          n_batch,
          n_height,
          n_width,
          normalized_image_range,
          # Network settings
          model_name,
          prepare_mode,
          max_input_depth,
          min_predict_depth,
          max_predict_depth,
          # Training settings
          learning_rates,
          learning_schedule,
          optimizer_betas,
          optimizer_epsilon,
          do_loss_inpainting,
          do_loss_perceptual,
          do_loss_style,
          warm_up,
          from_scratch,
          box_occlusion_size,
          # Augmentation settings
          augmentation_probabilities,
          augmentation_schedule,
          # Photometric data augmentations
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_gamma,
          augmentation_random_hue,
          augmentation_random_saturation,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          # Geometric data augmentations
          augmentation_random_crop_type,
          augmentation_random_crop_to_shape,
          augmentation_random_flip_type,
          augmentation_random_rotate_max,
          augmentation_random_crop_and_pad,
          augmentation_random_resize_and_pad,
          augmentation_random_resize_and_crop,
            # Occlusion data augmentations
          augmentation_random_remove_patch_percent_range_image,
          augmentation_random_remove_patch_size_image,
          augmentation_random_remove_patch_percent_range_depth,
          augmentation_random_remove_patch_size_depth,
          # Loss function settings
          w_weight_decay,
          loss_type,
          # Evaluation settings
          min_evaluate_depth,
          max_evaluate_depth,
          # Checkpoint settings
          checkpoint_path,
          n_step_per_checkpoint,
          n_step_per_summary,
          n_image_per_summary,
          validation_start_step,
          restore_path_model,
          # Hardware settings
          device,
          n_thread):
    if '01' in loss_type:
        init_method = 'tcp://localhost:56424'
    else:
        init_method = 'tcp://localhost:56423'
    dist.init_process_group(
        backend='nccl',
        init_method=init_method,
        world_size=torch.cuda.device_count(),
        rank=rank)

    torch.distributed.barrier(device_ids=[rank])
    torch.cuda.set_device(rank)  # set the cuda device

    '''
    Setup checkpoint path
    '''
    model_checkpoint_path = os.path.join(checkpoint_path, 'checkpoints', 'model-{:08d}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')
    if rank == 0:
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if not os.path.exists(os.path.join(checkpoint_path, "checkpoints")):
            os.mkdir(os.path.join(checkpoint_path, "checkpoints"))

    '''
    Set up training dataloader
    '''
    train_image_paths = data_utils.read_paths(train_image_path)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)

    n_sample = len(train_image_paths)

    input_paths = [
        train_image_paths,
        train_sparse_depth_paths,
        train_intrinsics_paths,
        train_ground_truth_paths
    ]

    for paths in input_paths:
        assert n_sample == len(paths)

    n_train_sample = len(train_image_paths)

    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)
    train_dataset = datasets.DepthCompletionSupervisedTrainingDataset(
        image_paths=train_image_paths,
        sparse_depth_paths=train_sparse_depth_paths,
        ground_truth_paths=train_ground_truth_paths,
        intrinsics_paths=train_intrinsics_paths,
        random_crop_shape=(n_height, n_width),
        random_crop_type=augmentation_random_crop_type,
        load_image_triplets=False)

    from torch.utils.data import DataLoader, DistributedSampler

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=ngpus_per_node, rank=rank)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=n_batch // ngpus_per_node,
        shuffle=False,
        sampler=train_sampler,
        num_workers=n_thread)

    log('Training input paths:', log_path)
    train_input_paths = [
        train_image_path,
        train_sparse_depth_path,
        train_intrinsics_path,
        train_ground_truth_path
    ]
    for path in train_input_paths:
        if path is not None and rank == 0:
            log(path, log_path)

    '''
    Set up validation dataloader
    '''
    is_available_validation = \
        val_image_path is not None and \
        val_sparse_depth_path is not None and \
        val_ground_truth_path is not None

    if is_available_validation:
        val_image_paths = data_utils.read_paths(val_image_path)
        val_sparse_depth_paths = data_utils.read_paths(val_sparse_depth_path)
        val_ground_truth_paths = data_utils.read_paths(val_ground_truth_path)

        n_val_sample = len(val_image_paths)

        if val_intrinsics_path is not None:
            val_intrinsics_paths = data_utils.read_paths(val_intrinsics_path)
        else:
            val_intrinsics_paths = [None] * n_val_sample

        for paths in [val_sparse_depth_paths, val_intrinsics_paths, val_ground_truth_paths]:
            assert len(paths) == n_val_sample

        val_dataloader = torch.utils.data.DataLoader(
            datasets.DepthCompletionInferenceDataset(
                image_paths=val_image_paths,
                sparse_depth_paths=val_sparse_depth_paths,
                intrinsics_paths=val_intrinsics_paths,
                ground_truth_paths=val_ground_truth_paths,
                load_image_triplets=False),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        best_results = {
            'step': -1,
            'mae': np.infty,
            'rmse': np.infty,
            'imae': np.infty,
            'irmse': np.infty
        }

        log('Validation input paths:', log_path)
        val_input_paths = [
            val_image_path,
            val_sparse_depth_path,
            val_intrinsics_path,
            val_ground_truth_path
        ]
        for path in val_input_paths:
            if path is not None:
                log(path, log_path)
        log('', log_path)

    '''
    Set up base external model
    '''
    # legacy applies offset to Deformable Convolution (only for NLSPN original pre-trained model)
    legacy = \
        (os.path.basename(restore_path_model) == 'NLSPN_KITTI_DC.pt' or \
        os.path.basename(restore_path_model) == 'NLSPN_NYUV2.pt')\
        if restore_path_model is not None else False

    model = ExternalModel_Adapt(
        model_name=model_name[0],
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=torch.device('cuda'),
        from_scratch=from_scratch,
        max_input_depth=max_evaluate_depth,
        offset=legacy)

    # Restore pretrained model parameters, preparation mode should have input restore_path_model.
    model._prepare_head(prepare_mode)

    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    model.restore_model(restore_path_model)
    head_params = model.prepare_parameters('head_selfsup_ema')

    head_optimizer = torch.optim.Adam(
        head_params,
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_epsilon,
        weight_decay=w_weight_decay)

    # Set up learning and augmentation schedule and optimizer
    model.convert_syncbn()
    model.distributed_data_parallel(rank=rank)
    learning_schedule_pos = 0
    parameters_model = model.parameters()

    # Setup data parallel if multiple GPUs are available within session
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # model.distributed_data_parallel(rank=rank)
    '''
    Log training settings
    '''
    if rank == 0:
        log_input_settings(
            log_path,
            # Batch settings
            n_batch=n_batch,
            n_height=n_height,
            n_width=n_width)

        log_network_settings(
            log_path,
            # Depth network settings
            model_name_depth=model_name,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            # Weight settings
            parameters_depth_model=parameters_model)

        log_training_settings(
            log_path,
            # Training settings
            n_batch=n_batch,
            n_train_sample=n_train_sample,
            n_train_step=n_train_step,
            learning_rates=learning_rates,
            learning_schedule=learning_schedule,
            # Augmentation settings
            augmentation_probabilities=augmentation_probabilities,
            augmentation_schedule=augmentation_schedule,
            # Photometric data augmentations
            augmentation_random_brightness=augmentation_random_brightness,
            augmentation_random_contrast=augmentation_random_contrast,
            augmentation_random_gamma=augmentation_random_gamma,
            augmentation_random_hue=augmentation_random_hue,
            augmentation_random_saturation=augmentation_random_saturation,
            augmentation_random_noise_type=augmentation_random_noise_type,
            augmentation_random_noise_spread=augmentation_random_noise_spread,
            # Geometric data augmentations
            augmentation_random_crop_type=augmentation_random_crop_type,
            augmentation_random_crop_to_shape=augmentation_random_crop_to_shape,
            augmentation_random_flip_type=augmentation_random_flip_type,
            augmentation_random_rotate_max=augmentation_random_rotate_max,
            augmentation_random_crop_and_pad=augmentation_random_crop_and_pad,
            # Occlusion data augmentations
            augmentation_random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_image,
            augmentation_random_remove_patch_size=augmentation_random_remove_patch_size_image,
            loss_inpainting=do_loss_inpainting,
            warm_up=warm_up,
            from_scratch=from_scratch,
            loss_perceptual=do_loss_perceptual,
            loss_style=do_loss_style)

        log_loss_func_settings(
            log_path,
            # Loss function settings
            w_weight_decay_depth=w_weight_decay)

        log_evaluation_settings(
            log_path,
            min_evaluate_depth=min_evaluate_depth,
            max_evaluate_depth=max_evaluate_depth)

        log_system_settings(
            log_path,
            # Checkpoint settings
            checkpoint_path=checkpoint_path,
            n_step_per_checkpoint=n_step_per_checkpoint,
            summary_event_path=event_path,
            n_step_per_summary=n_step_per_summary,
            n_image_per_summary=n_image_per_summary,
            validation_start_step=validation_start_step,
            restore_path_depth_model=restore_path_model,
            # Hardware settings
            n_thread=n_thread)

    # Set up Tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-eval')

    '''
    Set up data augmentations and transformations
    '''
    # Set up data augmentations
    train_transforms_geometric = Transforms(
        random_crop_to_shape=augmentation_random_crop_to_shape,
        random_flip_type=augmentation_random_flip_type,
        random_rotate_max=augmentation_random_rotate_max,
        random_crop_and_pad=augmentation_random_crop_and_pad,
        random_resize_and_pad=augmentation_random_resize_and_pad,
        random_resize_and_crop=augmentation_random_resize_and_crop)

    train_transforms_photometric_0 = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_gamma=augmentation_random_gamma,
        random_hue=augmentation_random_hue,
        random_saturation=augmentation_random_saturation,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread)

    # Map interpolation mode names to enums
    interpolation_modes = \
        train_transforms_geometric.map_interpolation_mode_names_to_enums(
            ['bilinear', 'nearest', 'nearest', 'nearest'])

    # Mask configuration for inpainting

    log('Begin training...', log_path)

    # Set model into training mode

    train_step = 0
    time_start = time.time()

    n_epoch = learning_schedule[-1]
    start_epoch = 1

    model.eval()
    model.train(prepare=True)

    iter_per_epoch = len(train_dataloader)
    if rank == 0:
        pbar = tqdm(total=iter_per_epoch * (n_epoch))

    dist.barrier(device_ids=[rank])

    for epoch in range(start_epoch, n_epoch + 1):
        train_sampler.set_epoch(epoch)
        model.train(prepare=True)
        # Learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in head_optimizer.param_groups:
                g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        for inputs in train_dataloader:
            train_step = train_step + 1

            # Move inputs to device
            inputs = [
                in_.to(device) for in_ in inputs
            ]

            # Unpack inputs
            image, sparse_depth, ground_truth, intrinsics = inputs
            validity_map = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                ground_truth)

            # Perform geometric augmentation i.e. crop, flip, etc.
            [image, sparse_depth, validity_map, ground_truth], [intrinsics] = train_transforms_geometric.transform(
                images_arr=[image, sparse_depth, validity_map, ground_truth],
                intrinsics_arr=[intrinsics],
                interpolation_modes=interpolation_modes,
                random_transform_probability=augmentation_probability)

            if model_name == 'rgb_guidance_uncertainty':
                sparse_depth = sparse_depth / max_predict_depth

            # Perform photometric augmentation i.e. masking, brightness, contrast, etc.
            [image_0] = train_transforms_photometric_0.transform(
                images_arr=[image],
                random_transform_probability=augmentation_probability)

            # Perform occlusion augmentation i.e. point removal

            output_depth, embedding, reference = model.forward(image=image_0,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics,
                loss_type=loss_type)
            loss, loss_info = model.compute_loss(
                input_rgb=image,
                output_depth=output_depth,
                validity_map=validity_map,
                ground_truth=ground_truth,
                embedding=embedding,
                reference=reference,
                loss_type='prepare')

            # optimization
            head_optimizer.zero_grad()
            loss.backward()
            head_optimizer.step()
            tag = 'ema'
            
            if (train_step % n_step_per_summary) == 0:
                model.log_summary(
                    train_summary_writer,
                    tag='contrast_{}'.format(tag),
                    step=train_step,
                    image=image,
                    sparse_depth=sparse_depth,
                    output_depth=output_depth,
                    validity_map=validity_map,
                    ground_truth=ground_truth,
                    output_image=None,
                    normalized_image_range=normalized_image_range,
                    scalars=loss_info,
                    n_image_per_summary=4)

            if rank == 0:
                desc = ''
                for k in loss_info.keys():
                    desc += '{}= {:.5f}, '.format(k, loss_info[k])
                pbar.set_description('meta_contrast - {}'.format(desc))
                pbar.update(1)
            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step
                if rank == 0:
                    log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                        train_step, n_train_step, loss.item(), time_elapse, time_remain),
                        log_path)
                    # Save checkpoint
                    model.save_model(
                        checkpoint_path=model_checkpoint_path.format(train_step),
                        step=train_step,
                        optimizer=head_optimizer)
                    with torch.no_grad():
                        model.eval()
                        validate(
                            model=model,
                            dataloader=val_dataloader,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            loss_type=loss_type,
                            device=device,
                            summary_writer=val_summary_writer,
                            normalized_image_range=normalized_image_range,
                            n_image_per_summary=n_image_per_summary,
                            log_path=log_path)
                    model.train(prepare=True)

    # Run validation on last step of training
    model.eval()
    # Save checkpoint on last step
    if rank == 0:
        model.save_model(
            checkpoint_path=model_checkpoint_path.format(train_step),
            step=train_step,
            optimizer=head_optimizer)
        with torch.no_grad():
            validate(
                model=model,
                dataloader=val_dataloader,
                step=train_step,
                best_results=best_results,
                min_evaluate_depth=min_evaluate_depth,
                max_evaluate_depth=max_evaluate_depth,
                loss_type=loss_type,
                device=device,
                summary_writer=val_summary_writer,
                normalized_image_range=normalized_image_range,
                n_image_per_summary=n_image_per_summary,
                log_path=log_path)


def validate(model,
             dataloader,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             loss_type,
             device,
             summary_writer,
             normalized_image_range,
             n_image_per_summary=4,
             n_interval_summary=10,
             log_path=None):
    '''
    Run model on validation dataset and return best results
    '''
    n_sample = len(dataloader)

    eval_transforms_photometric = Transforms(
        normalized_image_range=normalized_image_range)

    n_sample = len(dataloader)
    import tqdm
    loss_list = 0.0
    for idx, inputs in tqdm.tqdm(enumerate(dataloader), desc='Evaluating', total=n_sample):
        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics, ground_truth = inputs

        validity_map = torch.where(
            sparse_depth > 0,
            1.0,
            0.0)
        [image] = eval_transforms_photometric.transform(images_arr=[image])

        # Forward through network
        with torch.no_grad():
            output_depth, embedding, reference = model.forward(
                image=image,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics,
                loss_type=loss_type)
            loss, loss_info = model.compute_loss(
                input_rgb=image,
                output_depth=output_depth,
                validity_map=validity_map,
                ground_truth=ground_truth,
                embedding=embedding,
                reference=reference,
                loss_type='prepare')
            loss_list += loss.item() / n_sample

    log('avg. cosine-similarity-loss on validation : {}'.format(loss_list), log_path)
    # return best_results


'''
Helper functions for logging
'''


def log_input_settings(log_path,
                       input_channels_image=None,
                       input_channels_depth=None,
                       normalized_image_range=None,
                       outlier_removal_kernel_size=None,
                       outlier_removal_threshold=None,
                       n_batch=None,
                       n_height=None,
                       n_width=None):
    batch_settings_text = ''
    batch_settings_vars = []

    if n_batch is not None:
        batch_settings_text = batch_settings_text + 'n_batch={}'
        batch_settings_vars.append(n_batch)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_height is not None:
        batch_settings_text = batch_settings_text + 'n_height={}'
        batch_settings_vars.append(n_height)

    batch_settings_text = \
        batch_settings_text + '  ' if len(batch_settings_text) > 0 else batch_settings_text

    if n_width is not None:
        batch_settings_text = batch_settings_text + 'n_width={}'
        batch_settings_vars.append(n_width)

    log('Input settings:', log_path)

    if len(batch_settings_vars) > 0:
        log(batch_settings_text.format(*batch_settings_vars),
            log_path)

    if input_channels_image is not None or input_channels_depth is not None:
        log('input_channels_image={}  input_channels_depth={}'.format(
            input_channels_image, input_channels_depth),
            log_path)

    if normalized_image_range is not None:
        log('normalized_image_range={}'.format(normalized_image_range),
            log_path)

    if outlier_removal_kernel_size is not None and outlier_removal_threshold is not None:
        log('outlier_removal_kernel_size={}  outlier_removal_threshold={:.2f}'.format(
            outlier_removal_kernel_size, outlier_removal_threshold),
            log_path)
    log('', log_path)


def log_network_settings(log_path,
                         # Depth network settings
                         model_name_depth,
                         min_predict_depth,
                         max_predict_depth,
                         # Pose network settings
                         encoder_type_pose=None,
                         rotation_parameterization_pose=None,
                         # Weight settings
                         parameters_depth_model=[],
                         parameters_pose_model=[]):
    # Computer number of parameters
    n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    n_parameter_pose = sum(p.numel() for p in parameters_pose_model)

    n_parameter = n_parameter_depth + n_parameter_pose

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0:
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    if n_parameter_pose > 0:
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    log('Depth network settings:', log_path)
    log('model_name={}'.format(model_name_depth),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    if encoder_type_pose is not None:
        log('Pose network settings:', log_path)
        log('encoder_type_pose={}'.format(encoder_type_pose),
            log_path)
        log('rotation_parameterization_pose={}'.format(
            rotation_parameterization_pose),
            log_path)
        log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('', log_path)


def log_training_settings(log_path,
                          # Training settings
                          n_batch,
                          n_train_sample,
                          n_train_step,
                          learning_rates,
                          learning_schedule,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          # Photometric data augmentations
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_gamma,
                          augmentation_random_hue,
                          augmentation_random_saturation,
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
                          # Geometric data augmentations
                          augmentation_random_crop_type,
                          augmentation_random_crop_to_shape,
                          augmentation_random_flip_type,
                          augmentation_random_rotate_max,
                          augmentation_random_crop_and_pad,
                          # Occlusion data augmentations
                          augmentation_random_remove_patch_percent_range,
                          augmentation_random_remove_patch_size,
                          loss_inpainting,
                          warm_up,
                          from_scratch,
                          loss_perceptual,
                          loss_style):

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // n_batch), le * (n_train_sample // n_batch), v)
            for ls, le, v in
            zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)

    log('Photometric data augmentations:', log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_gamma={}'.format(augmentation_random_gamma),
        log_path)
    log('augmentation_random_hue={}'.format(augmentation_random_hue),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)

    log('Geometric data augmentations:', log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)
    log('augmentation_random_rotate_max={}'.format(augmentation_random_rotate_max),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_remove_patch_percent_range={}  augmentation_random_remove_patch_size={}'.format(
        augmentation_random_remove_patch_percent_range, augmentation_random_remove_patch_size),
        log_path)

    log('loss_inpainting = {}'.format(loss_inpainting), log_path)
    log('lr_warmup = {}'.format(warm_up), log_path)
    log('training_from_scratch = {}'.format(from_scratch), log_path)
    log('loss_perceptual = {}'.format(loss_perceptual), log_path)
    log('loss_style = {}'.format(loss_style), log_path)
    log('', log_path)


def log_loss_func_settings(log_path,
                           # Loss function settings
                           w_losses={},
                           w_weight_decay_depth=None,
                           w_weight_decay_pose=None):
    w_losses_text = ''
    for idx, (key, value) in enumerate(w_losses.items()):

        if idx > 0 and idx % 3 == 0:
            w_losses_text = w_losses_text + '\n'

        w_losses_text = w_losses_text + '{}={:.1e}'.format(key, value) + '  '

    log('Loss function settings:', log_path)
    if len(w_losses_text) > 0:
        log(w_losses_text, log_path)

    if w_weight_decay_depth is not None:
        log('w_weight_decay_depth={:.1e}'.format(
            w_weight_decay_depth),
            log_path)

    if w_weight_decay_pose is not None:
        log('w_weight_decay_pose={:.1e}'.format(
            w_weight_decay_pose),
            log_path)
    log('', log_path)


def log_evaluation_settings(log_path,
                            min_evaluate_depth,
                            max_evaluate_depth):
    log('Evaluation settings:', log_path)
    log('min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        min_evaluate_depth, max_evaluate_depth),
        log_path)
    log('', log_path)


def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_image_per_summary=None,
                        validation_start_step=None,
                        restore_path_depth_model=None,
                        restore_path_pose_model=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):
    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('checkpoint_save_frequency={}'.format(n_step_per_checkpoint), log_path)

        if validation_start_step is not None:
            log('validation_start_step={}'.format(validation_start_step), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'log_summary_frequency={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_image_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_image_per_summary={}'
        summary_settings_vars.append(n_image_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_path_depth_model is not None and restore_path_depth_model != '':
        log('restore_path_depth_model={}'.format(restore_path_depth_model),
            log_path)

    if restore_path_pose_model is not None and restore_path_pose_model != '':
        log('restore_path_pose_model={}'.format(restore_path_pose_model),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
