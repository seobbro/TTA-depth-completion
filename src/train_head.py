import argparse, torch
from head_main import prepare_ddp
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image_path',
    type=str, required=True, help='Path to list of training image paths')
parser.add_argument('--train_sparse_depth_path',
    type=str, required=True, help='Path to list of training sparse depth paths')
parser.add_argument('--train_ground_truth_path',
    type=str, required=True, help='Path to list of training ground truth paths')
parser.add_argument('--train_intrinsics_path',
    type=str, default=None, help='Path to list of training intrinsics paths')

parser.add_argument('--val_image_path',
    type=str, default=None, help='Path to list of validation image paths')
parser.add_argument('--val_sparse_depth_path',
    type=str, default=None, help='Path to list of validation sparse depth paths')
parser.add_argument('--val_ground_truth_path',
    type=str, default=None, help='Path to list of validation ground truth paths')
parser.add_argument('--val_intrinsics_path',
    type=str, default=None, help='Path to list of validation intrinsics paths')

# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=2, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=256, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=256, help='Width of each sample')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 255], help='Range of image intensities after normalization, if given more than 2 elements then standard normalization')
parser.add_argument('--box_occlusion_size',
    nargs='+', type=float, default=[46, 152], help='The range of height and width of occlusion box')

# Network settings
parser.add_argument('--model_name',
    nargs='+', type=str, help='Depth completion model to instantiate')
parser.add_argument('--prepare_mode',
    type=str, default=None, help='Depth completion model to instantiate')
parser.add_argument('--max_input_depth',
    type=float, default=None, help='Maximum value of depth to evaluate')
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum depth prediction value')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum depth prediction value')

# Training settings
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[2e-4, 1e-4, 5e-5, 1e-5], help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[10, 20, 30, 35], help='Space delimited list to change learning rate')
parser.add_argument('--optimizer_betas',
    nargs='+', default=[0.9, 0.999], help='Beta for Adam optimizer')
parser.add_argument('--optimizer_epsilon',
    type=float, default=1e-8, help='Epsilon for Adam optimizer')
parser.add_argument('--do_loss_inpainting',
   action='store_true', help='If set, then perform inpainting')
parser.add_argument('--do_loss_perceptual',
   action='store_true', help='If set, then perform perceptual loss')
parser.add_argument('--do_loss_style',
   action='store_true', help='If set, then perform style (gram matrix) loss')
parser.add_argument('--warm_up',
   action='store_true', help='If set, do warm up learning schedule for 1st epoch')
parser.add_argument('--from_scratch',
   action='store_true', help='If set, then train from scratch')

# Augmentation settings
parser.add_argument('--augmentation_probabilities',
    nargs='+', type=float, default=[1.00], help='Probabilities to use data augmentation. Note: there is small chance that no augmentation take place even when we enter augmentation pipeline.')
parser.add_argument('--augmentation_schedule',
    nargs='+', type=int, default=[-1], help='If not -1, then space delimited list to change augmentation probability')

# Photometric data augmentations
parser.add_argument('--augmentation_random_brightness',
    nargs='+', type=float, default=[-1, -1], help='Range of brightness adjustments for augmentation, if does not contain -1, apply random brightness')
parser.add_argument('--augmentation_random_contrast',
    nargs='+', type=float, default=[-1, -1], help='Range of contrast adjustments for augmentation, if does not contain -1, apply random contrast')
parser.add_argument('--augmentation_random_gamma',
    nargs='+', type=float, default=[-1, 1], help='Range of gamma adjustments for augmentation, if does not contain -1, apply random gamma')
parser.add_argument('--augmentation_random_hue',
    nargs='+', type=float, default=[-1, -1], help='Range of hue adjustments for augmentation, if does not contain -1, apply random hue')
parser.add_argument('--augmentation_random_saturation',
    nargs='+', type=float, default=[-1, -1], help='Range of saturation adjustments for augmentation, if does not contain -1, apply random saturation')
parser.add_argument('--augmentation_random_noise_type',
    type=str, default='none', help='Random noise to add: gaussian, uniform')
parser.add_argument('--augmentation_random_noise_spread',
    type=float, default=-1, help='If gaussian noise, then standard deviation; if uniform, then min-max range')

# Geometric data augmentations
parser.add_argument('--augmentation_random_crop_type',
    nargs='+', type=str, default=['none'], help='Random crop type for data augmentation: horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_crop_to_shape',
    nargs='+', type=int, default=[-1, -1], help='Random crop to : horizontal, vertical, anchored, bottom')
parser.add_argument('--augmentation_random_flip_type',
    nargs='+', type=str, default=['none'], help='Random flip type for data augmentation: horizontal, vertical')
parser.add_argument('--augmentation_random_rotate_max',
    type=float, default=-1, help='Max angle for random rotation, disabled if -1')
parser.add_argument('--augmentation_random_crop_and_pad',
    nargs='+', type=float, default=[-1, -1], help='If set to positive numbers, then treat as min and max percentage to crop and pad')
parser.add_argument('--augmentation_random_resize_and_pad',
    nargs='+', type=float, default=[-1, -1], help='Min/Max resize ratio')
parser.add_argument('--augmentation_random_resize_and_crop',
    nargs='+', type=float, default=[-1, -1], help='Min/Max resize ratio')

# Occlusion data augmentations
parser.add_argument('--augmentation_random_remove_patch_percent_range_image',
    nargs='+', type=float, default=[-1, -1], help='If not -1, randomly remove patches covering percentage of image as augmentation')
parser.add_argument('--augmentation_random_remove_patch_size_image',
    nargs='+', type=int, default=[-1, -1], help='If not -1, patch size for random remove patch augmentation for image')
parser.add_argument('--augmentation_random_remove_patch_percent_range_depth',
    nargs='+', type=float, default=[-1, -1], help='If not -1, randomly remove patches covering percentage of depth map as augmentation')
parser.add_argument('--augmentation_random_remove_patch_size_depth',
    nargs='+', type=int, default=[-1, -1], help='If not -1, patch size for random remove patch augmentation for depth map')

# Loss settings
parser.add_argument('--w_weight_decay',
    type=float, default=0.0, help='Weight of weight decay loss')
parser.add_argument('--loss_type',
    type=str, default='prepare', help='Defines forward and loss function type.')
# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.1, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')

# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to save checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=5000, help='Number of iterations for each checkpoint')
parser.add_argument('--n_step_per_summary',
    type=int, default=5000, help='Number of iterations before logging summary')
parser.add_argument('--n_image_per_summary',
    type=int, default=4, help='Number of samples to include in visual display summary')
parser.add_argument('--validation_start_step',
    type=int, default=80000, help='Number of steps before starting validation')
parser.add_argument('--restore_path_model',
    type=str, default=None, help='Path to restore depth model from checkpoint')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of threads for fetching')


args = parser.parse_args()


if __name__ == '__main__':

    # Network settings
    args.model_name = [
        name.lower() for name in args.model_name
    ]

    # Training settings
    assert len(args.learning_rates) == len(args.learning_schedule)

    args.augmentation_random_crop_type = [
        crop_type.lower() for crop_type in args.augmentation_random_crop_type
    ]

    args.augmentation_random_flip_type = [
        flip_type.lower() for flip_type in args.augmentation_random_flip_type
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    mp.spawn(prepare_ddp, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(),
          args.train_image_path,
          args.train_sparse_depth_path,
          args.train_intrinsics_path,
          args.train_ground_truth_path,
          args.val_image_path,
          args.val_sparse_depth_path,
          args.val_ground_truth_path,
          args.val_intrinsics_path,
          # Batch settings
          args.n_batch,
          args.n_height,
          args.n_width,
          args.normalized_image_range,
          # Network settings
          args.model_name,
          args.prepare_mode,
          args.max_input_depth,
          args.min_predict_depth,
          args.max_predict_depth,
          # Training settings
          args.learning_rates,
          args.learning_schedule,
          args.optimizer_betas,
          args.optimizer_epsilon,
          args.do_loss_inpainting,
          args.do_loss_perceptual,
          args.do_loss_style,
          args.warm_up,
          args.from_scratch,
          args.box_occlusion_size,
          # Augmentation settings
          args.augmentation_probabilities,
          args.augmentation_schedule,
          # Photometric data augmentations
          args.augmentation_random_brightness,
          args.augmentation_random_contrast,
          args.augmentation_random_gamma,
          args.augmentation_random_hue,
          args.augmentation_random_saturation,
          args.augmentation_random_noise_type,
          args.augmentation_random_noise_spread,
          # Geometric data augmentations
          args.augmentation_random_crop_type,
          args.augmentation_random_crop_to_shape,
          args.augmentation_random_flip_type,
          args.augmentation_random_rotate_max,
          args.augmentation_random_crop_and_pad,
          args.augmentation_random_resize_and_pad,
          args.augmentation_random_resize_and_crop,
          # Occlusion data augmentations
          args.augmentation_random_remove_patch_percent_range_image,
          args.augmentation_random_remove_patch_size_image,
          args.augmentation_random_remove_patch_percent_range_depth,
          args.augmentation_random_remove_patch_size_depth,
          # Loss function settings
          args.w_weight_decay,
          args.loss_type,
          # Evaluation settings
          args.min_evaluate_depth,
          args.max_evaluate_depth,
          # Checkpoint settings
          args.checkpoint_path,
          args.n_step_per_checkpoint,
          args.n_step_per_summary,
          args.n_image_per_summary,
          args.validation_start_step,
          args.restore_path_model,
          # Hardware settings
          args.device,
          args.n_thread))
