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
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.cuda.amp as amp
from tqdm import tqdm

from log_utils import log
from external_model_adapt import ExternalModel_Adapt
from transforms import Transforms
import data_utils, datasets, eval_utils

def init_ddp(rank,
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

    if 'kitti' in train_image_path:
        dataset_name = 'kitti'
    elif 'void' in train_image_path:
        dataset_name = 'void'
    elif 'waymo' in train_image_path:
        dataset_name = 'waymo'

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
        if path is not None:
            log(path, log_path)
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

        val_sampler = DistributedSampler(
            val_dataset, num_replicas=ngpus_per_node, shuffle=False, rank=rank)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=n_batch // ngpus_per_node,
            shuffle=False,
            num_workers=1,
            sampler=val_sampler,
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

    pretrained_list = ['external_models/nlspn/kitti/checkpoints/model-00175000.pth',
                       'external_models/nlspn/kitti/model-00175000.pth',
                       'external_models/nlspn/void/nlspn-void1500_.pth',
                       'external_models/enet/void/enet-void1500.pth',
                       'external_models/enet/kitti/e.pth.tar',
                       'external_models/rgb_guidance_uncertainty/kitti/model_best_epoch.pth.tar',
                       'external_models/completionformer/kitti/KITTIDC_L1L2.pt',
                       'external_models/completionformer/nyu/NYUv2.pt',
                       'external_models/costdcnet/kitti/costdcnet_kitti.pth',
                       'external_models/costdcnet/void/costdcnet_void.pth',
                       'external_models/waymo/msgchn_waymo.pth'
                       ]
    # Restore pretrained model parameters, preparation mode should have input restore_path_model.
    if restore_path_model in pretrained_list:
        model.restore_model(restore_path_model)
        metaparams = model.prepare_parameters(prepare_mode)
        learning_schedule_pos = 0
        learning_rate = learning_rates[0]

        augmentation_schedule_pos = 0
        augmentation_probability = augmentation_probabilities[0]

        # Get model parameters to be optimized

        meta_optimizer = torch.optim.Adam(
            metaparams,
            lr=learning_rate,
            betas=optimizer_betas,
            eps=optimizer_epsilon,
            weight_decay=w_weight_decay)
    # Continue Meta train
    else:
        metaparams = model.prepare_parameters(prepare_mode)
        learning_schedule_pos = 0
        learning_rate = learning_rates[0]

        augmentation_schedule_pos = 0
        augmentation_probability = augmentation_probabilities[0]
        meta_optimizer = torch.optim.Adam(
            metaparams,
            lr=learning_rate,
            betas=optimizer_betas,
            eps=optimizer_epsilon,
            weight_decay=w_weight_decay)

        meta_optimizer = model.restore_model(restore_path_model, meta_optimizer)
        learning_schedule_pos = 0
        learning_rate = learning_rates[learning_schedule_pos]

        # Update optimizer learning rates
        for g in meta_optimizer.param_groups:
            g['lr'] = learning_rate

    # Get model parameters to be optimized
    
    model.convert_syncbn()
    parameters_model = model.parameters()
    # Set up learning and augmentation schedule and optimizer
    learning_schedule_pos = 0
    
    # Setup data parallel if multiple GPUs are available within session
    model.distributed_data_parallel(rank=rank)
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
    model.train()

    train_step = 0
    time_start = time.time()
    n_epoch = learning_schedule[-1]
    start_epoch = 1

    model.train(meta=True)
    iter_per_epoch = len(train_dataloader)
    if rank == 0:
        pbar = tqdm(total=iter_per_epoch * (n_epoch))

    dist.barrier(device_ids=[rank])

    for epoch in range(start_epoch, n_epoch + 1):
        train_sampler.set_epoch(epoch)

        # Learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in meta_optimizer.param_groups:
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

            # Perform photometric augmentation i.e. masking, brightness, contrast, etc.
            [image_0] = train_transforms_photometric_0.transform(
                images_arr=[image],
                random_transform_probability=augmentation_probability)

            output_depth = model.forward(image=image_0,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics,
                loss_type=loss_type)
            loss, loss_info = model.compute_loss(
                input_rgb=image,
                output_depth=output_depth,
                validity_map=validity_map,
                ground_truth=ground_truth,
                embedding=None,
                reference=None,
                dataset_name=dataset_name,
                loss_type='pretrain')

            meta_optimizer.zero_grad()
            loss.backward()
            meta_optimizer.step()

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

                # Run validation
                if train_step >= validation_start_step and is_available_validation:
                    # Switch to validation mode
                    model.eval()
                    with torch.no_grad():
                        best_results = validate(
                            model=model,
                            dataloader=val_dataloader,
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depth=min_evaluate_depth,
                            max_evaluate_depth=max_evaluate_depth,
                            loss_type=loss_type,
                            evaluate_dataset_name='kitti',
                            device=device,
                            rank=rank,
                            summary_writer=val_summary_writer,
                            normalized_image_range=normalized_image_range,
                            n_image_per_summary=n_image_per_summary,
                            log_path=log_path)
                    # Switch back to training
                    model.train(meta=True)

                # Save checkpoint
                model.save_model(
                    checkpoint_path=model_checkpoint_path.format(train_step),
                    step=train_step,
                    optimizer=meta_optimizer)

    # Run validation on last step of training
    model.eval()

    # Save checkpoint on last step
    if rank == 0:
        model.save_model(
            checkpoint_path=model_checkpoint_path.format(train_step),
            step=train_step,
            optimizer=meta_optimizer)
        with torch.no_grad():
            best_results = validate(
                model=model,
                dataloader=val_dataloader,
                step=train_step,
                best_results=best_results,
                evaluate_dataset_name='kitti',
                min_evaluate_depth=min_evaluate_depth,
                max_evaluate_depth=max_evaluate_depth,
                loss_type=loss_type,
                device=device,
                rank=rank,
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
             evaluate_dataset_name,
             device,
             summary_writer,
             normalized_image_range,
             loss_type,
             rank,
             n_image_per_summary=4,
             n_interval_summary=250,
             log_path=None,
             describe=''):
    '''
    Run model on validation dataset and return best results
    '''
    n_sample = len(dataloader.dataset)
    n_batch = len(dataloader)
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

    from tqdm import tqdm
    pbar = tqdm(len(dataloader),  bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}')
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

        [image] = eval_transforms_photometric.transform(images_arr=[image])
        # Validity map is where sparse depth is available
        with torch.no_grad():
            output_depth = model.forward(
                image=image,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics,
                crop_mask=None,
                loss_type=loss_type)
            if isinstance(output_depth, list):
                output_depth = output_depth[0]
        
        # Store for Tensorboard logging
        if (idx % n_interval_summary) == 0 and summary_writer is not None:
            image_summary.append(image)
            output_depth_summary.append(output_depth.cpu())
            sparse_depth_summary.append(sparse_depth.cpu())
            validity_map_summary.append(validity_map_depth.cpu())
            ground_truth_summary.append(ground_truth.cpu())

        # Convert to numpy to validate
        # Only for Vkitti
        if evaluate_dataset_name == 'vkitti':
            # Crop output_depth and ground_truth
            crop_height = 240
            crop_width = 1216
            crop_mask = [240, 1216]
        elif evaluate_dataset_name == 'nuscenes':
            # Crop output_depth and ground_truth
            crop_height = 540
            crop_width = 1600
            crop_mask = [540, 1600]
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

        output_depth = np.squeeze(output_depth.to(torch.device('cpu')).numpy())
        ground_truth = np.squeeze(ground_truth.to(torch.device('cpu')).numpy())

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
    results = torch.tensor([mae, rmse, imae, irmse]).cuda()
    dist.all_reduce(results)
    mae, rmse, imae, irmse = results.cpu()

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
    if rank == 0:
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

def calibrate(rank,
    model,
    train_dataloader,
    train_transforms_geometric,
    train_transforms_photometric,
    interpolation_modes,
    augmentation_probability,
    checkpoint_path,
    device):

    if rank == 0 :
        pbar = tqdm(total=len(train_dataloader))
    for inputs in train_dataloader:
        inputs = [
            in_.to(device) for in_ in inputs
        ]
        # Unpack inputs
        image, sparse_depth, ground_truth, intrinsics = inputs.copy()
        validity_map_depth = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        # Remove outlier points and update sparse depth and validity map
        # Perform geometric augmentation i.e. crop, flip, etc.
        [image, sparse_depth, validity_map_depth, ground_truth], [intrinsics] = train_transforms_geometric.transform(
            images_arr=[image, sparse_depth, validity_map_depth, ground_truth],
            intrinsics_arr=[intrinsics],
            interpolation_modes=interpolation_modes,
            random_transform_probability=augmentation_probability)

        # Perform photometric augmentation i.e. masking, brightness, contrast, etc.

        [image1] = train_transforms_photometric.transform(
            images_arr=[image],
            random_transform_probability=augmentation_probability)

        # Apply masking to images for inpainting auxiliary task
        with torch.no_grad():

            model.forward(image=image1,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics,
                crop_mask=None,
                loss_type="get_meanvar")

        if rank == 0:
            pbar.set_description('get mean')
            pbar.update(1)
            #     loss, loss_info = model.compute_loss(
            #         input_rgb=image,
            #         output_depth=output_depth,
            #         sparse_depth=filtered_sparse_depth,
            #         validity_map=filtered_validity_map_depth,
            #         embedding=emb,
            #         reference=ref,
            #         w_loss_sparse_depth=w_loss_sparse_depth,
            #         w_loss_smoothness=w_loss_smoothness,
            #         w_loss_cos=w_loss_cos,
            #         loss_type=loss_type)

            # loss_total += loss_info['loss_cos'] / float(len(train_dataloader))

        # if (train_step % n_step_per_summary) == 0:
        #     model.log_summary(
        #         train_summary_writer,
        #         tag='train_lr={}'.format(learning_rate),
        #         step=train_step,
        #         image=[image1],
        #         sparse_depth=filtered_sparse_depth,
        #         output_depth=output_depth,
        #         validity_map=filtered_validity_map_depth,
        #         ground_truth=ground_truth,
        #         output_image=None,
        #         normalized_image_range=normalized_image_range,
        #         scalars=loss_info,
        #         n_image_per_summary=4)
        # disc = ""
        # for k in loss_info.keys():
        #     disc += "{}: {:.5f} ,".format(k, loss_info[k])
# Run validation on last step of training
    dict_meanval = model.get_mean_val(mean_only=True)
    torch.save(dict_meanval, os.path.join(checkpoint_path, 'checkpoints', 'mean_var_dict'))
    model.model.model.module.glob_mean = dict_meanval
    model.model.glob_mean = True

    if rank == 0 :
        pbar = tqdm(total=len(train_dataloader))
    for inputs in train_dataloader:
        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]
        # Unpack inputs
        image, sparse_depth, ground_truth, intrinsics = inputs.copy()
        validity_map_depth = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        # Remove outlier points and update sparse depth and validity map
        # Perform geometric augmentation i.e. crop, flip, etc.
        [image, sparse_depth, ground_truth], [intrinsics] = train_transforms_geometric.transform(
            images_arr=[image, sparse_depth, ground_truth],
            intrinsics_arr=[intrinsics],
            interpolation_modes=interpolation_modes,
            random_transform_probability=augmentation_probability)

        # Perform photometric augmentation i.e. masking, brightness, contrast, etc.

        [image1] = train_transforms_photometric.transform(
            images_arr=[image],
            random_transform_probability=augmentation_probability)

        # Apply masking to images for inpainting auxiliary task
        with torch.no_grad():
            model.forward(image=image1,
                sparse_depth=sparse_depth,
                intrinsics=intrinsics,
                crop_mask=None,
                loss_type='get_meanvar')

        if rank == 0:
            pbar.set_description('get cov')
            pbar.update(1)

# Run validation on last step of training
    dict_meancov = model.get_mean_val()
    torch.save(dict_meancov, os.path.join(checkpoint_path, 'checkpoints', 'mean_var_dict'))
