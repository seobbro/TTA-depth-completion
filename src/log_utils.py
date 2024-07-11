'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import os
import torch
import numpy as np
from matplotlib import pyplot as plt


def log(s, filepath=None, to_console=True):
    '''
    Logs a string to either file or console

    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''

    if to_console:
        print(s)

    if filepath is not None:
        if not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')


def colorize(T, colormap='magma'):
    '''
    Colorizes a 1-channel tensor with matplotlib colormaps

    Arg(s):
        T : torch.Tensor[float32]
            1-channel tensor
        colormap : str
            matplotlib colormap
    '''

    cm = plt.cm.get_cmap(colormap)
    shape = T.shape

    # Convert to numpy array and transpose
    if shape[0] > 1:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)))
    else:
        T = np.squeeze(np.transpose(T.cpu().numpy(), (0, 2, 3, 1)), axis=-1)

    # Colorize using colormap and transpose back
    color = np.concatenate([
        np.expand_dims(cm(T[n, ...])[..., 0:3], 0) for n in range(T.shape[0])],
        axis=0)
    color = np.transpose(color, (0, 3, 1, 2))

    # Convert back to tensor
    return torch.from_numpy(color.astype(np.float32))


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
