import os
import numpy as np
import torch
import data_utils, datasets, eval_utils
from log_utils import log
from external_model_adapt import ExternalModel_Adapt
from transforms import Transforms
from net_utils import OutlierRemoval
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import time
import gc
from pynvml import *
from datetime import datetime
def get_sampler(dataset, ngpus_per_node, rank, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = DistributedSampler(dataset, num_replicas=ngpus_per_node, rank=rank, seed=seed)
    return sampler

def adapt_ddp(rank,
          ngpus_per_node,
          train_image_path,
          train_sparse_depth_path,
          train_ground_truth_path,
          train_intrinsics_path,
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
          adapt_mode,
          inner_iter,
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
          w_loss_sparse_depth,
          w_loss_smoothness,
          w_loss_cos,
          w_loss_robust,
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
    if '23' in loss_type:
        init_method = 'tcp://localhost:56431'
    else:
        init_method = 'tcp://localhost:56434'
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

    # datetime object containing current date and time
    now = datetime.now()

    # print("now =", now)

    # # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
    # print("date and time =", dt_string)
    if rank == 0 :
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
            os.mkdir(os.path.join(checkpoint_path, 'checkpoints'))
    model_checkpoint_path = os.path.join(checkpoint_path, 'checkpoints', 'model-{:08d}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_base_path = checkpoint_path.replace('model_adapt_meta', 'tensorboard')
    event_path = os.path.join(event_base_path, 'events')
    if rank == 0 :
        if not os.path.exists(checkpoint_path):
            os.mkdir(event_base_path)
            os.mkdir(os.path.join(event_base_path, 'events-train'))
            os.mkdir(os.path.join(event_base_path, 'events-test'))

    '''
    Set up training dataloader
    '''
    if 'nuscenes' in train_image_path:
        dataset_name = 'nuscenes'
    elif 'waymo' in train_image_path:
        dataset_name = 'waymo'
    elif 'vkitti' in train_image_path:
        dataset_name = 'vkitti'
    elif 'synthia' in train_image_path:
        dataset_name = 'synthia'
    else:
        dataset_name = ''

    train_image_paths = data_utils.read_paths(train_image_path)
    train_sparse_depth_paths = data_utils.read_paths(train_sparse_depth_path)
    train_ground_truth_paths = data_utils.read_paths(train_ground_truth_path)
    train_intrinsics_paths = data_utils.read_paths(train_intrinsics_path) if train_intrinsics_path is not None else None

    n_sample = len(train_image_paths)
    if "concat" in prepare_mode:
        flag_concat = True
    else:
        flag_concat = False

    input_paths = [
        train_image_paths,
        train_sparse_depth_paths,
        train_intrinsics_paths,
        train_ground_truth_paths
    ]

    for paths in input_paths:
        if paths is not None:
            assert n_sample == len(paths)

    n_train_sample = len(train_image_paths)

    n_train_step = \
        learning_schedule[-1] * np.ceil(n_train_sample / n_batch).astype(np.int32)
    if flag_concat:
        train_dataset = datasets.DepthCompletionSupervisedTrainingDataset_ConCat(
            image_paths=train_image_paths,
            sparse_depth_paths=train_sparse_depth_paths,
            ground_truth_paths=train_ground_truth_paths,
            intrinsics_paths=train_intrinsics_paths,
            inner_iter=inner_iter,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type,
            load_image_triplets=False)

        import random
        rand_seed = random.randint(0, 1000000)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ngpus_per_node, rank=rank)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=n_batch // ngpus_per_node // inner_iter,
            shuffle=False,
            sampler=get_sampler(train_dataset, ngpus_per_node=ngpus_per_node, rank=rank, seed=rand_seed),
            num_workers=n_thread,
            pin_memory=False)

        if rank == 0:
            log('Training input paths:', log_path)

    else:
        train_dataset = datasets.DepthCompletionSupervisedTrainingDataset(
            image_paths=train_image_paths,
            sparse_depth_paths=train_sparse_depth_paths,
            ground_truth_paths=train_ground_truth_paths,
            intrinsics_paths=train_intrinsics_paths,
            random_crop_shape=(n_height, n_width),
            random_crop_type=augmentation_random_crop_type,
            load_image_triplets=False)
        import random
        rand_seed = random.randint(0, 1000000)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ngpus_per_node, rank=rank)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=n_batch // ngpus_per_node,
            shuffle=False,
            sampler=get_sampler(train_dataset, ngpus_per_node=ngpus_per_node, rank=rank, seed=rand_seed),
            num_workers=n_thread,
            pin_memory=False)
        if rank == 0:
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
    if rank == 0:
        log('', log_path)

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
        val_dataset = datasets.DepthCompletionInferenceDataset(
            image_paths=val_image_paths,
            sparse_depth_paths=val_sparse_depth_paths,
            intrinsics_paths=val_intrinsics_paths,
            ground_truth_paths=val_ground_truth_paths,
            load_image_triplets=False)
        if flag_concat:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=(n_batch // ngpus_per_node // inner_iter),
                shuffle=False,
                sampler=get_sampler(val_dataset, ngpus_per_node=ngpus_per_node, rank=rank, seed=rand_seed),
                num_workers=n_thread,
                pin_memory=False,
                drop_last=False)
        else:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=(n_batch // ngpus_per_node),
                shuffle=False,
                sampler=get_sampler(val_dataset, ngpus_per_node=ngpus_per_node, rank=rank, seed=rand_seed),
                num_workers=n_thread,
                pin_memory=False,
                drop_last=False)

        best_results = {
            'step': -1,
            'mae': np.infty,
            'rmse': np.infty,
            'imae': np.infty,
            'irmse': np.infty
        }

    '''
    Set up base external model
    '''
    # legacy applies offset to Deformable Convolution (only for NLSPN original pre-trained model)
    # legacy = \
    #     (os.path.basename(restore_path_model) == 'NLSPN_KITTI_DC.pt' or \
    #     os.path.basename(restore_path_model) == 'NLSPN_NYUV2.pt')\
    #     if restore_path_model is not None else False

    model = ExternalModel_Adapt(
        model_name=model_name[0],
        max_input_depth=max_input_depth,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=torch.device('cuda'),
        from_scratch=from_scratch,
        dataset_name=dataset_name,
        offset=True)

    # Setup adapt
    # For pre-trained parameters + MLP from preparation stage

    model._prepare_head(prepare_mode)
    model.restore_model(restore_path_model)

    # For adaptation parameters
    model.convert_syncbn()

    # Setup adapt
    # For pre-trained parameters + MLP from preparation stage

    # Get model parameters to be optimized
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]
    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    # Setup optimizer
    # model.init_bn_stats()
    params = model.adapt_parameters(mode=adapt_mode)

    optimizer = torch.optim.Adam(
        params=params,
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_epsilon,
        weight_decay=w_weight_decay)

    learning_schedule_pos = 0

    parameters_model = model.parameters()

    # Setup data parallel if multiple GPUs are available within session

    model.distributed_data_parallel(rank=rank)
    '''
    Log training settings
    '''
    if rank == 0 :
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
    if rank == 0:
        train_summary_writer = SummaryWriter(event_path + '-train')
        val_summary_writer = SummaryWriter(event_path + '-eval')
    dist.barrier(device_ids=[rank])
    '''
    Set up learning and augmentation schedule and optimizer
    '''

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

    train_transforms_photometric = Transforms(
        normalized_image_range=normalized_image_range,
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_gamma=augmentation_random_gamma,
        random_hue=augmentation_random_hue,
        random_saturation=augmentation_random_saturation,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread)

    eval_transforms_photometric = Transforms(
        normalized_image_range=normalized_image_range)

    train_transforms_occlusion_image = Transforms(
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_depth,
        random_remove_patch_size=augmentation_random_remove_patch_size_depth)

    # Map interpolation mode names to enums
    interpolation_modes = \
        train_transforms_geometric.map_interpolation_mode_names_to_enums(
            ['bilinear', 'nearest', 'nearest', 'nearest'])

    # Mask configuration for inpainting

    log('Begin training...', log_path)

    # Set model into training mode
    model.train()

    train_step = 0
    if dataset_name != 'nuscenes':
        outlier_removal = OutlierRemoval(7, 1.5)

    dist.barrier(device_ids=[rank])
    mae = 0.0
    rmse = 0.0
    imae = 0.0
    irmse = 0.0
    update_time = 0.0
    if rank == 0:
        pbar = tqdm(total=len(train_dataloader), bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}')
    train_sampler.set_epoch(1)

    best_results = {
        'step': -1,
        'mae': np.infty,
        'rmse': np.infty,
        'imae': np.infty,
        'irmse': np.infty
    }

    for inputs, val_inputs in zip(train_dataloader, val_dataloader):
        model.train()
        train_step = train_step + 1
        if train_step > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]
            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        # Learning rate schedule

        # Set augmentation schedule
        if -1 not in augmentation_schedule and train_step > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        # Move inputs to device
        inputs = [
            in_.to(device, non_blocking=False) for in_ in inputs
        ]
        if flag_concat:
            image, sparse_depth, ground_truth, intrinsics = inputs
            image = image.view(-1, 3, n_height, n_width)
            sparse_depth = sparse_depth.view(-1, 1, n_height, n_width)
            ground_truth = ground_truth.view(-1, 1, n_height, n_width)
            intrinsics = intrinsics.view(-1, 3, 3)

            validity_map_depth = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)
            if dataset_name != 'nuscenes':
                filtered_sparse_depth, filtered_validity_map_depth = outlier_removal.remove_outliers(sparse_depth=sparse_depth, validity_map=validity_map_depth)
            else:
                filtered_sparse_depth, filtered_validity_map_depth = sparse_depth, validity_map_depth

                # Perform geometric augmentation i.e. crop, flip, etc.
            [image, filtered_sparse_depth, filtered_validity_map_depth, ground_truth], [intrinsics] = train_transforms_geometric.transform(
                images_arr=[image, filtered_sparse_depth, filtered_validity_map_depth, ground_truth],
                intrinsics_arr=[intrinsics],
                interpolation_modes=interpolation_modes,
                random_transform_probability=augmentation_probability)

            # Perform photometric augmentation i.e. masking, brightness, contrast, etc.
            [image1] = train_transforms_photometric.transform(
                images_arr=[image],
                random_transform_probability=augmentation_probability)

            # [filtered_sparse_depth] = train_transforms_occlusion_image.transform(
            #     images_arr=[filtered_sparse_depth],
            #     random_transform_probability=augmentation_probability)

            output_depth, emb, ref = model.forward(image=image1,
                sparse_depth=filtered_sparse_depth,
                intrinsics=intrinsics,
                crop_mask=None,
                loss_type=loss_type)

            loss, loss_info = model.compute_loss(
                input_rgb=image.detach(),
                output_depth=output_depth,
                sparse_depth=filtered_sparse_depth.detach(),
                validity_map=filtered_validity_map_depth.detach(),
                embedding=emb,
                reference=ref,
                w_loss_sparse_depth=w_loss_sparse_depth,
                w_loss_smoothness=w_loss_smoothness,
                w_loss_cos=w_loss_cos,
                loss_type=loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            for _ in range(inner_iter):
                # Unpack inputs
                image, sparse_depth, ground_truth, intrinsics = inputs

                validity_map_depth = torch.where(
                    sparse_depth > 0,
                    torch.ones_like(sparse_depth),
                    sparse_depth)

                # Remove outlier points and update sparse depth and validity map
                if dataset_name not in ['nuscenes', 'waymo'] :
                    filtered_sparse_depth, filtered_validity_map_depth = outlier_removal.remove_outliers(sparse_depth=sparse_depth, validity_map=validity_map_depth)
                else:
                    filtered_sparse_depth, filtered_validity_map_depth = sparse_depth, validity_map_depth

                # Perform geometric augmentation i.e. crop, flip, etc.
                [image, filtered_sparse_depth, filtered_validity_map_depth, ground_truth], [intrinsics] = train_transforms_geometric.transform(
                    images_arr=[image, filtered_sparse_depth, filtered_validity_map_depth, ground_truth],
                    intrinsics_arr=[intrinsics],
                    interpolation_modes=interpolation_modes,
                    random_transform_probability=augmentation_probability)

                # Perform photometric augmentation i.e. masking, brightness, contrast, etc.
                [image1] = train_transforms_photometric.transform(
                    images_arr=[image],
                    random_transform_probability=augmentation_probability)

                # [filtered_sparse_depth] = train_transforms_occlusion_image.transform(
                #     images_arr=[filtered_sparse_depth],
                #     random_transform_probability=augmentation_probability)

                output_depth, emb, ref = model.forward(image=image1,
                    sparse_depth=filtered_sparse_depth,
                    intrinsics=intrinsics,
                    crop_mask=None,
                    loss_type=loss_type)

                if 'time' in loss_type:
                    before_backward = time.time()

                loss, loss_info = model.compute_loss(
                    input_rgb=image.detach(),
                    output_depth=output_depth,
                    sparse_depth=filtered_sparse_depth.detach(),
                    validity_map=filtered_validity_map_depth.detach(),
                    embedding=emb,
                    reference=ref,
                    w_loss_sparse_depth=w_loss_sparse_depth,
                    w_loss_smoothness=w_loss_smoothness,
                    w_loss_cos=w_loss_cos,
                    loss_type='adapt')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if 'time' in loss_type:
                    update_time += time.time() - before_backward
        image1 = image1.to(torch.device('cpu'))
        filtered_sparse_depth = filtered_sparse_depth.to(torch.device('cpu'))
        filtered_validity_map_depth = filtered_validity_map_depth.to(torch.device('cpu'))
        ground_truth = ground_truth.to(torch.device('cpu'))
        if (train_step % n_step_per_summary) == 0 and rank == 0:
            model.log_summary(
                train_summary_writer,
                tag='adapt',
                step=train_step,
                image=[image1],
                sparse_depth=filtered_sparse_depth,
                output_depth=output_depth.clone().detach().to(torch.device('cpu')),
                validity_map=filtered_validity_map_depth,
                ground_truth=ground_truth,
                output_image=None,
                normalized_image_range=normalized_image_range,
                scalars=loss_info,
                n_image_per_summary=1)
        del image, sparse_depth, ground_truth, intrinsics, image1, validity_map_depth, filtered_validity_map_depth, filtered_sparse_depth, loss_info
        # disc = ''
        # for k in loss_info.keys():
        #     disc += '{}: {:.5f} ,'.format(k, loss_info[k])
        # disc += "|| {}: {:.5f}".format('loss_kldiv', loss_align_depth.item())

        # Eval

        n_sample = len(val_dataloader.dataset)

        val_inputs = [
            in_.to(rank, non_blocking=False) for in_ in val_inputs
        ]
        image, sparse_depth, intrinsics, ground_truth = val_inputs
        # pbar = tqdm(total=iter_per_epoch * dataloader.batch_size * torch.cuda.device_count())
        if dataset_name == 'vkitti':
            # Crop output_depth and ground_truth
            crop_height = 240
            crop_width = 1216
            crop_mask = [240, 1216]
        elif dataset_name == 'nuscenes':
            # Crop output_depth and ground_truth
            crop_height = 544
            crop_width = 1600
            crop_mask = [544, 1600]
        elif dataset_name == 'waymo':
            # Crop output_depth and ground_truth
            crop_height = 640
            crop_width = 1920
            crop_mask = [640, 1920]
        elif dataset_name == 'synthia':
            crop_height = 320
            crop_width = 640
            crop_mask = [320, 640]
        else:
            crop_mask = None

        if dataset_name in ['nuscenes', 'waymo']:
            filtered_sparse_depth = sparse_depth
            validity_map_depth = torch.where(sparse_depth > 0,
                torch.ones_like(sparse_depth),
                torch.zeros_like(sparse_depth))
            filtered_validity_map_depth = validity_map_depth
        else:
            validity_map_depth = torch.where(sparse_depth > 0,
                torch.ones_like(sparse_depth),
                torch.zeros_like(sparse_depth))
            # if evaluate_dataset_name != 'nuscenes':
            filtered_sparse_depth, filtered_validity_map_depth = outlier_removal.remove_outliers(sparse_depth=sparse_depth, validity_map=validity_map_depth)
        # else:
        #   filtered_sparse_depth, filtered_validity_map_depth = sparse_depth, validity_map_depth

        if crop_mask is not None:
            H, W = ground_truth.size()[-2], ground_truth.size()[-1]
            center = W // 2

            start_x = center - crop_width // 2
            end_x = center + crop_width // 2

            # bottom crop
            end_y = H
            start_y = end_y - crop_height

            image = image[..., start_y:end_y, start_x:end_x]
            sparse_depth = sparse_depth[..., start_y:end_y, start_x:end_x]
            validity_map_depth = validity_map_depth[..., start_y:end_y, start_x:end_x]
            filtered_sparse_depth = filtered_sparse_depth[..., start_y:end_y, start_x:end_x]
            filtered_validity_map_depth = filtered_validity_map_depth[..., start_y:end_y, start_x:end_x]
            ground_truth = ground_truth[..., start_y:end_y, start_x:end_x]

        [image] = eval_transforms_photometric.transform(images_arr=[image])
        # Validity map is where sparse depth is available
        batch_size = image.size(0)

        model.eval()
        with torch.no_grad():
            output_depth = model.forward(
                image=image,
                sparse_depth=filtered_sparse_depth,
                intrinsics=intrinsics,
                crop_mask=None,
                loss_type=loss_type)

            if rank == 0 and (train_step % n_step_per_summary) == 0 and val_summary_writer is not None:
                output_depth = output_depth.cuda()
                model.log_summary(
                    summary_writer=val_summary_writer,
                    tag='eval-{}'.format(dt_string),
                    step=train_step,
                    image=image,
                    output_depth=output_depth.clone().detach(),
                    sparse_depth=sparse_depth,
                    validity_map=validity_map_depth,
                    ground_truth=ground_truth,
                    normalized_image_range=normalized_image_range,
                    scalars={'mae' : mae*1000/(batch_size * train_step), 'rmse' : rmse*1000/(batch_size * train_step), 'imae' : imae*1000/(batch_size * train_step), 'irmse': irmse*1000/(batch_size * train_step)},
                    n_image_per_summary=n_image_per_summary)
        # Convert to numpy to validate
        # Only for Vkitti

        if crop_mask is not None:
            H, W = ground_truth.size()[-2], ground_truth.size()[-1]
            center = W // 2

            start_x = center - crop_width // 2
            end_x = center + crop_width // 2

            # bottom crop
            end_y = H
            start_y = end_y - crop_height

            output_depth = output_depth[..., start_y:end_y, start_x:end_x]
            ground_truth = ground_truth[..., start_y:end_y, start_x:end_x]

        output_depth = torch.squeeze(output_depth.clone().detach()).to(torch.device('cpu'))
        ground_truth = torch.squeeze(ground_truth).to(torch.device('cpu'))

        # Separate validity map from ground truth

        # ground_truth = ground_truth[0, :, :] # Will be back to this code after

        validity_mask = torch.where(ground_truth > 0, torch.ones_like(ground_truth), torch.zeros_like(ground_truth)).to(torch.device('cpu'))

        # Select valid regions based on validity map and min/max values

        validity_mask[ground_truth < min_evaluate_depth] = 0.0
        validity_mask[ground_truth > max_evaluate_depth] = 0.0
        # min_max_mask = torch.logical_and(
        #     ground_truth > min_evaluate_depth,
        #     ground_truth < max_evaluate_depth)
        # mask = torch.logical_and(
        #     ground_truth > min_evaluate_depth,
        #     ground_truth < max_evaluate_depth)
        # print(validity_mask.size())
        # print(output_depth.size())
        output_depth = output_depth[validity_mask.nonzero(as_tuple=True)]
        ground_truth = ground_truth[validity_mask.nonzero(as_tuple=True)]
        output_depth = output_depth.clone().detach()
        ground_truth = ground_truth.clone().detach()
        # Compute validation metrics
        mae += eval_utils.torch_mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth).to(torch.device('cpu')) * batch_size / 1000
        rmse += eval_utils.torch_root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth).to(torch.device('cpu')) * batch_size / 1000
        imae += eval_utils.torch_inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth).to(torch.device('cpu')) * batch_size / 1000
        irmse += eval_utils.torch_inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth).to(torch.device('cpu')) * batch_size / 1000
        del output_depth, ground_truth, image, filtered_sparse_depth, sparse_depth, filtered_validity_map_depth, validity_map_depth, validity_mask
        if rank == 0:
            pbar.set_description('tb path: {} / {}'.format(event_base_path, loss.detach()))
            pbar.update(1)
        gc.collect()
        dist.barrier()

    mae   = (mae) / n_sample * 1000
    rmse  = (rmse) / n_sample * 1000
    imae  = (imae) / n_sample * 1000
    irmse = (irmse) / n_sample * 1000

    total_time, train_time, eval_time = model.forward(
        image=None,
        sparse_depth=None,
        intrinsics=None,
        crop_mask=None,
        loss_type='get_time')

    if 'time' in loss_type and rank == 0:
        log("{}: {}, {}: {}, {}: {}, train_backward: {}, train: {}".format('total', total_time/len(train_dataset), 'train_forward', train_time/len(train_dataset), 'eval', eval_time/len(train_dataset), update_time/len(train_dataset), (train_time + update_time)/len(train_dataset)))
        log("{}: {}, {}: {}, {}: {}, train_backward_fps: {}, train_fps: {}".format('total_fps', 1/(total_time/len(train_dataset)), 'train_forward_fps', 1/(train_time/len(train_dataset)), 'eval_fps', 1/(eval_time/len(train_dataset)), 1/(update_time/len(train_dataset)), 1/((train_time + update_time)/len(train_dataset))))

    # for dist eval
    metric_values = torch.tensor([mae, rmse, imae, irmse]).cuda(rank)
    dist.all_reduce(metric_values)
    mae, rmse, imae, irmse = metric_values.cpu()

    #  Print validation results to console
    if rank == 0:
        log('Validation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            train_step, mae, rmse, imae, irmse),
            log_path)

        n_improve = 0
        if np.round(mae, 2) <= np.round(best_results['mae'], 2):
            n_improve = n_improve + 1
        if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
            n_improve = n_improve + 1
        if np.round(imae, 2) <= np.round(best_results['imae'], 2):
            n_improve = n_improve + 1
        if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
            n_improve = n_improve + 1

        if n_improve > 2:
            best_results['step'] = train_step
            best_results['mae'] = mae
            best_results['rmse'] = rmse
            best_results['imae'] = imae
            best_results['irmse'] = irmse

        log('Best results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            best_results['step'],
            best_results['mae'],
            best_results['rmse'],
            best_results['imae'],
            best_results['irmse']), log_path)
# Run validation on last step of training
    # with torch.no_grad():
    #     model.eval()
    #     validate(model=model,
    #             dataloader=val_dataloader,
    #             step=train_step,
    #             best_results=best_results,
    #             min_evaluate_depth=min_evaluate_depth,
    #             max_evaluate_depth=max_evaluate_depth,
    #             evaluate_dataset_name=dataset_name,
    #             device=device,
    #             summary_writer=None,
    #             normalized_image_range=normalized_image_range,
    #             loss_type=loss_type,
    #             n_image_per_summary=4,
    #             n_interval_summary=250,
    #             log_path=None)
    #     model.train()
    if rank == 0:
        model.save_model(
            checkpoint_path=model_checkpoint_path.format(train_step),
            step=train_step,
            optimizer=optimizer)


def validate(model,
             dataloader,
             step,
             best_results,
             min_evaluate_depth,
             max_evaluate_depth,
             evaluate_dataset_name,
             device,
             summary_writer,
             normalized_image_range,
             loss_type,
             n_image_per_summary=4,
             n_interval_summary=250,
             log_path=None,
             describe=''):
    '''
    Run model on validation dataset and return best results
    '''
    n_sample = len(dataloader.dataset)
    print(n_sample)
    n_batch = len(dataloader)
    print(n_batch)
    print(dataloader.batch_size)
    mae = np.zeros(n_batch)
    rmse = np.zeros(n_batch)
    imae = np.zeros(n_batch)
    irmse = np.zeros(n_batch)

    image_summary = []
    output_depth_summary = []
    sparse_depth_summary = []
    validity_map_summary = []
    ground_truth_summary = []

    eval_transforms_photometric = Transforms(
        normalized_image_range=normalized_image_range)

    # pbar = tqdm(total=iter_per_epoch * dataloader.batch_size * torch.cuda.device_count())
    outlier_removal = OutlierRemoval(7, 1.5)

    from tqdm import tqdm
    pbar = tqdm(len(dataloader))
    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        image, sparse_depth, intrinsics, ground_truth = inputs

        batch_size = image.size(0)

        validity_map_depth = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        if evaluate_dataset_name != 'nuscenes':
            filtered_sparse_depth, filtered_validity_map_depth = outlier_removal.remove_outliers(sparse_depth=sparse_depth, validity_map=validity_map_depth)
        else:
            filtered_sparse_depth, filtered_validity_map_depth = sparse_depth, validity_map_depth

        [image] = eval_transforms_photometric.transform(images_arr=[image])
        # Validity map is where sparse depth is available

        output_depth = model.forward(
            image=image,
            sparse_depth=filtered_sparse_depth,
            intrinsics=intrinsics,
            crop_mask=None,
            loss_type=loss_type)

        # ground_truth = torch.clamp(ground_truth, 0, max_input_depth)
        # pbar.update(dataloader.batch_size * torch.cuda.device_count())
        # Store for Tensorboard logging
        if (idx % n_interval_summary) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth)
            sparse_depth_summary.append(filtered_sparse_depth)
            validity_map_summary.append(filtered_validity_map_depth)
            ground_truth_summary.append(ground_truth)

        # Convert to numpy to validate
        # Only for Vkitti

        if evaluate_dataset_name == 'vkitti':
            # Crop output_depth and ground_truth
            crop_height = 240
            crop_width = 1216
            crop_mask = [240, 1216]
        elif evaluate_dataset_name == 'nuscenes':
            # Crop output_depth and ground_truth
            crop_height = 544
            crop_width = 1600
            crop_mask = [544, 1600]
        else:
            crop_mask = None

        if crop_mask is not None:
            H, W = ground_truth.size()[-2], ground_truth.size()[-1]
            center = W // 2
            start_x = center - crop_width // 2
            end_x = center + crop_width // 2

            # bottom crop
            end_y = H
            start_y = end_y - crop_height

            output_depth = output_depth[..., start_y:end_y, start_x:end_x]
            ground_truth = ground_truth[..., start_y:end_y, start_x:end_x]

        output_depth = np.squeeze(output_depth.cpu().numpy())
        ground_truth = np.squeeze(ground_truth.cpu().numpy())

        # Separate validity map from ground truth

        # ground_truth = ground_truth[0, :, :] # Will be back to this code after

        validity_mask = np.where(ground_truth > 0, 1, 0)

        # Select valid regions based on validity map and min/max values
        min_max_mask = np.logical_and(
            ground_truth > min_evaluate_depth,
            ground_truth < max_evaluate_depth)

        mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

        output_depth = output_depth[mask]
        ground_truth = ground_truth[mask]

        # Compute validation metrics
        mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth) * batch_size
        rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth) * batch_size
        imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth) * batch_size
        irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth) * batch_size
        pbar.update(1)
    # Compute mean metrics
    mae   = np.sum(mae) / n_sample
    rmse  = np.sum(rmse) / n_sample
    imae  = np.sum(imae) / n_sample
    irmse = np.sum(irmse) / n_sample

    # # Log to tensorboard
    if summary_writer is not None:
        model.log_summary(
            summary_writer=summary_writer,
            tag='eval-{}'.format(describe),
            step=step,
            image=torch.cat(image_summary, dim=0),
            output_depth=torch.cat(output_depth_summary, dim=0),
            sparse_depth=torch.cat(sparse_depth_summary, dim=0),
            validity_map=torch.cat(validity_map_summary, dim=0),
            ground_truth=torch.cat(ground_truth_summary, dim=0),
            normalized_image_range=normalized_image_range,
            scalars={'mae' : mae, 'rmse' : rmse, 'imae' : imae, 'irmse': irmse},
            n_image_per_summary=n_image_per_summary)

    # Print validation results to console
    log('Validation results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step, mae, rmse, imae, irmse),
        log_path)

    n_improve = 0
    if np.round(mae, 2) <= np.round(best_results['mae'], 2):
        n_improve = n_improve + 1
    if np.round(rmse, 2) <= np.round(best_results['rmse'], 2):
        n_improve = n_improve + 1
    if np.round(imae, 2) <= np.round(best_results['imae'], 2):
        n_improve = n_improve + 1
    if np.round(irmse, 2) <= np.round(best_results['irmse'], 2):
        n_improve = n_improve + 1

    if n_improve > 2:
        best_results['step'] = step
        best_results['mae'] = mae
        best_results['rmse'] = rmse
        best_results['imae'] = imae
        best_results['irmse'] = irmse

    log('Best results:', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        best_results['step'],
        best_results['mae'],
        best_results['rmse'],
        best_results['imae'],
        best_results['irmse']), log_path)

    return best_results


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
