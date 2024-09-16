import os, sys, argparse, shutil
import tensorflow.compat.v1 as tf
import numpy as np
tf.enable_eager_execution()
from PIL import Image
import glob
from natsort import natsorted
import time
import shutil
import concurrent.futures
from waymo_open_dataset import dataset_pb2 as open_dataset
import sys
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import torch
sys.path.append('/home/vaezhov/workspace/depth-estimation-in-dark/')  # for datautils
from src import data_utils

# Note: need to copy and paste and run this file in /home/vaezhov/workspace/waymo-open-dataset/src/? or is the path enough.
sys.path.append('/home/vaezhov/workspace/waymo-open-dataset/src/bazel-bin')
sys.path.insert(0, 'src')
sys.path.append(os.getcwd())


'''
Set up input arguments
'''
parser = argparse.ArgumentParser()

parser.add_argument('--waymo_data_root_dirpath',
    type=str, required=True, help='Path to waymo dataset')
parser.add_argument('--waymo_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--n_forward_frames_to_reproject',
    type=int, default=7, help='Number of forward frames to project onto a target frame')
parser.add_argument('--n_backward_frames_to_reproject',
    type=int, default=7, help='Number of backward frames to project onto a target frame')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=4, help='Number of threads to use in parallel pool')
# NOTE: arguments below are in nuscenes but not in Waymo yet can be applied optionally
parser.add_argument('--debug',
    action='store_true', help='If set, then enter debug mode')
parser.add_argument('--filter_threshold_photometric_reconstruction',
    type=float, default=-1, help='If set to greater than 0 then perform photometric reconstruction filtering (unvalidated)')
# If --enable_outlier_removal is specified, args.enable_outlier_removal will be True
parser.add_argument('--enable_outlier_removal',
                    action='store_true',
                    default=False,
                    help='Enable outlier removal post-processing on ground truth')
parser.add_argument('--outlier_removal_kernel_size',
    type=int, default=7, help='Kernel size for removing outliers from camera lidar position offest')
parser.add_argument('--outlier_removal_threshold',
    type=float, default=1.5, help='Distance threshold for consider a point outlier')
parser.add_argument('--concatenation_block_size',
    type=int, default=3, help='k, defines [i, i+k] sequential concatenation of frames')


args = parser.parse_args()


'''
Setup output filepaths
'''
DATASET_NAME = "waymo"

# Set up directory paths
TRAIN_REF_DIRPATH = os.path.join(args.waymo_data_derived_dirpath,'training', DATASET_NAME)
VAL_REF_DIRPATH = os.path.join(args.waymo_data_derived_dirpath,'validation', DATASET_NAME)
TEST_REF_DIRPATH = os.path.join(args.waymo_data_derived_dirpath,'testing', DATASET_NAME)
TESTVAL_REF_DIRPATH = os.path.join(args.waymo_data_derived_dirpath,'testval', DATASET_NAME)

# Training set for all, day, and night splits
TRAIN_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_image.txt')
TRAIN_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_lidar.txt')
TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_ground_truth.txt')
TRAIN_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_intrinsics.txt')
TRAIN_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_absolute_pose.txt')
TRAIN_IMAGE_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_image_klet.txt')
TRAIN_LIDAR_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_lidar_klet.txt')
TRAIN_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_ground_truth_klet.txt')

TRAIN_DAY_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_image.txt')
TRAIN_DAY_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_lidar.txt')
TRAIN_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_ground_truth.txt')
TRAIN_DAY_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_intrinsics.txt')
TRAIN_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_absolute_pose.txt')
TRAIN_DAY_IMAGE_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_image_klet.txt')
TRAIN_DAY_LIDAR_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_lidar_klet.txt')
TRAIN_DAY_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_day_ground_truth_klet.txt')

TRAIN_NIGHT_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_image.txt')
TRAIN_NIGHT_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_lidar.txt')
TRAIN_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_ground_truth.txt')
TRAIN_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_intrinsics.txt')
TRAIN_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_absolute_pose.txt')
TRAIN_NIGHT_IMAGE_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_image_klet.txt')
TRAIN_NIGHT_LIDAR_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_lidar_klet.txt')
TRAIN_NIGHT_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, DATASET_NAME + '_train_night_ground_truth_klet.txt')

# Validation set for all, day, and night splits
VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_image.txt')
VAL_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_lidar.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_ground_truth.txt')
VAL_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_intrinsics.txt')
VAL_ABSOLUTE_POSE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_absolute_pose.txt')
VAL_IMAGE_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_image_klet.txt')
VAL_LIDAR_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_lidar_klet.txt')
VAL_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_ground_truth_klet.txt')

VAL_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_image-subset.txt')
VAL_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_lidar-subset.txt')
VAL_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_ground_truth-subset.txt')
VAL_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_intrinsics-subset.txt')
VAL_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_absolute_pose-subset.txt')
VAL_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_image_klet-subset.txt')
VAL_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_lidar_klet-subset.txt')
VAL_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_ground_truth_klet-subset.txt')

VAL_DAY_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_image.txt')
VAL_DAY_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_lidar.txt')
VAL_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_ground_truth.txt')
VAL_DAY_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_intrinsics.txt')
VAL_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_absolute_pose.txt')
VAL_DAY_IMAGE_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_image_klet.txt')
VAL_DAY_LIDAR_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_lidar_klet.txt')
VAL_DAY_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_ground_truth_klet.txt')

VAL_DAY_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_image-subset.txt')
VAL_DAY_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_lidar-subset.txt')
VAL_DAY_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_ground_truth-subset.txt')
VAL_DAY_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_intrinsics-subset.txt')
VAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_absolute_pose-subset.txt')
VAL_DAY_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_image_klet-subset.txt')
VAL_DAY_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_lidar_klet-subset.txt')
VAL_DAY_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_day_ground_truth_klet-subset.txt')

VAL_NIGHT_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_image.txt')
VAL_NIGHT_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_lidar.txt')
VAL_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_ground_truth.txt')
VAL_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_intrinsics.txt')
VAL_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_absolute_pose.txt')
VAL_NIGHT_IMAGE_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_image_klet.txt')
VAL_NIGHT_LIDAR_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_lidar_klet.txt')
VAL_NIGHT_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_ground_truth_klet.txt')

VAL_NIGHT_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_image-subset.txt')
VAL_NIGHT_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_lidar-subset.txt')
VAL_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_ground_truth-subset.txt')
VAL_NIGHT_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_intrinsics-subset.txt')
VAL_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_absolute_pose-subset.txt')
VAL_NIGHT_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_image_klet-subset.txt')
VAL_NIGHT_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_lidar_klet-subset.txt')
VAL_NIGHT_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, DATASET_NAME + '_val_night_ground_truth_klet-subset.txt')

# Testing set for all, day, and night splits
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_image.txt')
TEST_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_lidar.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_ground_truth.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_intrinsics.txt')
TEST_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_absolute_pose.txt')
TEST_IMAGE_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_image_klet.txt')
TEST_LIDAR_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_lidar_klet.txt')
TEST_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_ground_truth_klet.txt')

TEST_IMAGE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_image-subset.txt')
TEST_LIDAR_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_lidar-subset.txt')
TEST_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_ground_truth-subset.txt')
TEST_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_intrinsics-subset.txt')
TEST_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_absolute_pose-subset.txt')
TEST_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_image_klet-subset.txt')
TEST_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_lidar_klet-subset.txt')
TEST_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_ground_truth_klet-subset.txt')

TEST_DAY_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_image.txt')
TEST_DAY_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_lidar.txt')
TEST_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_ground_truth.txt')
TEST_DAY_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_intrinsics.txt')
TEST_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_absolute_pose.txt')
TEST_DAY_IMAGE_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_image_klet.txt')
TEST_DAY_LIDAR_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_lidar_klet.txt')
TEST_DAY_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_ground_truth_klet.txt')

TEST_DAY_IMAGE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_image-subset.txt')
TEST_DAY_LIDAR_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_lidar-subset.txt')
TEST_DAY_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_ground_truth-subset.txt')
TEST_DAY_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_intrinsics-subset.txt')
TEST_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_absolute_pose-subset.txt')
TEST_DAY_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_image_klet-subset.txt')
TEST_DAY_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_lidar_klet-subset.txt')
TEST_DAY_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_day_ground_truth_klet-subset.txt')

TEST_NIGHT_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_image.txt')
TEST_NIGHT_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_lidar.txt')
TEST_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_ground_truth.txt')
TEST_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_intrinsics.txt')
TEST_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_absolute_pose.txt')
TEST_NIGHT_IMAGE_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_image_klet.txt')
TEST_NIGHT_LIDAR_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_lidar_klet.txt')
TEST_NIGHT_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_ground_truth_klet.txt')

TEST_NIGHT_IMAGE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_image-subset.txt')
TEST_NIGHT_LIDAR_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_lidar-subset.txt')
TEST_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_ground_truth-subset.txt')
TEST_NIGHT_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_intrinsics-subset.txt')
TEST_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_absolute_pose-subset.txt')
TEST_NIGHT_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_image_klet-subset.txt')
TEST_NIGHT_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_lidar_klet-subset.txt')
TEST_NIGHT_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, DATASET_NAME + '_test_night_ground_truth_klet-subset.txt')

# Testing + validation set to include larger amounts of night-time samples
TESTVAL_IMAGE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_image.txt')
TESTVAL_LIDAR_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_lidar.txt')
TESTVAL_GROUND_TRUTH_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_ground_truth.txt')
TESTVAL_INTRINSICS_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_intrinsics.txt')
TESTVAL_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_absolute_pose.txt')
TESTVAL_IMAGE_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_image_klet.txt')
TESTVAL_LIDAR_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_lidar_klet.txt')
TESTVAL_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_ground_truth_klet.txt')

TESTVAL_IMAGE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_image-subset.txt')
TESTVAL_LIDAR_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_lidar-subset.txt')
TESTVAL_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_ground_truth-subset.txt')
TESTVAL_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_intrinsics-subset.txt')
TESTVAL_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_absolute_pose-subset.txt')
TESTVAL_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_image_klet-subset.txt')
TESTVAL_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_lidar_klet-subset.txt')
TESTVAL_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_ground_truth_klet-subset.txt')

TESTVAL_DAY_IMAGE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_image.txt')
TESTVAL_DAY_LIDAR_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_lidar.txt')
TESTVAL_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_ground_truth.txt')
TESTVAL_DAY_INTRINSICS_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_intrinsics.txt')
TESTVAL_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_absolute_pose.txt')
TESTVAL_DAY_IMAGE_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_image_klet.txt')
TESTVAL_DAY_LIDAR_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_lidar_klet.txt')
TESTVAL_DAY_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_ground_truth_klet.txt')

TESTVAL_DAY_IMAGE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_image-subset.txt')
TESTVAL_DAY_LIDAR_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_lidar-subset.txt')
TESTVAL_DAY_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_ground_truth-subset.txt')
TESTVAL_DAY_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_intrinsics-subset.txt')
TESTVAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_absolute_pose-subset.txt')
TESTVAL_DAY_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_image_klet-subset.txt')
TESTVAL_DAY_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_lidar_klet-subset.txt')
TESTVAL_DAY_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_day_ground_truth_klet-subset.txt')

TESTVAL_NIGHT_IMAGE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_image.txt')
TESTVAL_NIGHT_LIDAR_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_lidar.txt')
TESTVAL_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_ground_truth.txt')
TESTVAL_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_intrinsics.txt')
TESTVAL_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_absolute_pose.txt')
TESTVAL_NIGHT_IMAGE_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_image_klet.txt')
TESTVAL_NIGHT_LIDAR_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_lidar_klet.txt')
TESTVAL_NIGHT_GROUND_TRUTH_KLET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_ground_truth_klet.txt')

TESTVAL_NIGHT_IMAGE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_image-subset.txt')
TESTVAL_NIGHT_LIDAR_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_lidar-subset.txt')
TESTVAL_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_ground_truth-subset.txt')
TESTVAL_NIGHT_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_intrinsics-subset.txt')
TESTVAL_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_absolute_pose-subset.txt')
TESTVAL_NIGHT_IMAGE_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_image_klet-subset.txt')
TESTVAL_NIGHT_LIDAR_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_lidar_klet-subset.txt')
TESTVAL_NIGHT_GROUND_TRUTH_KLET_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, DATASET_NAME + '_testval_night_ground_truth_klet-subset.txt')


def get_time_of_day(scene_label_input_filepath):
    '''
    Checks what time of day the frame belongs to.

    Arg(s):
        scene_label_input_filepath : str
            path to the frame's scene npy file. Refer to extract_waymo_cleaned for formatting.
    Returns:
        string : either 'day' or 'night'
    '''
    frame_scene_label = np.load(scene_label_input_filepath)
    frame_time_of_day = frame_scene_label[0]
    if 'Day' in frame_time_of_day or 'Dawn' in frame_time_of_day or 'Dusk' in frame_time_of_day:
        tag_daynight = 'day'
    else:
        tag_daynight = 'night'
    return tag_daynight

# Helper function from the old code.
def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file
    NOTE: similar to data_utils.load_image(...), but different since loading in lidar png.

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def read_frame_data(frame):
    '''
    Given a path to a Waymo frame directory, extracts all the data files from 
    the directory.
    Arg(s):
        frame : str
            path to the current frame
    Returns:
        A bunch of file paths, each containing info denoted by the variable's name.
    '''
    lidar_input_filepath = os.path.join(frame,'sparse_depth.png')  # depth
    image_input_filepath = os.path.join(frame,'front_camera.jpeg')
    veh_pose_input_filepath = os.path.join(frame,'vehicle_to_global_pose.npy')
    cam_extrinsic_input_filepath = os.path.join(frame,'camera_to_vehicle.npy')
    cam_intrinsics_input_filepath = os.path.join(frame,'front_cam_intrinsics.npy')
    bboxes_input_filepath = os.path.join(frame,'bboxes.npy')
    metadata_input_filepath = os.path.join(frame,'camera_metadata.npy')
    scene_label_input_filepath = os.path.join(frame,'scene_label.npy')
    return lidar_input_filepath, image_input_filepath, veh_pose_input_filepath, \
        cam_extrinsic_input_filepath, cam_intrinsics_input_filepath, bboxes_input_filepath, \
        metadata_input_filepath, scene_label_input_filepath

def merge_pointcloud_unidirectional(nonkf_frames, 
                                    main_lidar_image,
                                    main_validity_map,
                                    pose_vehicle_to_global_kf,
                                    extrinsics_camera_to_vehicle_kf,
                                    camera_intrinsics_kf, 
                                    bboxes_kf,
                                    frame_metadata_kf):
    '''
    Given a set of non-kf frames and info of the kf frame, project the depth estimations of the
    non-kf frames onto the kf frame, updating in place.
    Arg(s):
        nonkf_frames : List[str]
            list of paths to all nonkf frames (keyframe = current/main frame; non keyframe = prev/backward or next/forward frame)
        main_lidar_image : numpy.arary
            3D array of (y, x, depth); depth map of the main/kf/current frame of interest
        main_validity_map : numpy.array
            3D array of (y, x, int{0,1}); marks whether a valid depth exists at a pixel location (1), else (0).
        pose_vehicle_to_global_kf : numpy.array
            vehicle to global pose tensor of keyframe
        extrinsics_camera_to_vehicle_kf : numpy.array
            camera to vehicle extrinsic tensor of keyframe
        camera_intrinsics_kf : numpy.array
            camera intrinsic tensor of keyframe
        bboxes_kf : numpy.array
            bounding boxes tensor of dynamic objects in keyframe
        frame_metadata_kf : numpy.array
            frame metadata tensor of keyframe
    Returns:
        nothing; updates main_lidar_image and main_validity_map in place.
    '''

    # NOTE: nonkf (non-keyframe frame) can either be forward/next, or backward/prev
    n_nonkf = len(nonkf_frames)
    n_nonkf_processed = 0

    # Process each nonkf frame
    while n_nonkf_processed < n_nonkf:
        '''
        Process non-keyframe to obtain its projected depth estimation.
        ''' 
        nonkf_frame = nonkf_frames[n_nonkf_processed]

        # Get all input file paths of nonkf frame
        nonkf_frame_lidar_input_filepath, \
        __, \
        nonkf_frame_veh_pose_input_filepath, \
        nonkf_frame_cam_extrinsic_input_filepath, \
        nonkf_frame_cam_intrinsics_input_filepath, \
        nonkf_frame_bboxes_input_filepath, \
        nonkf_frame_metadata_input_filepath, \
        __ = read_frame_data(nonkf_frame)

        # Load lidar depth tensor from path
        nonkf_frame_lidar_image = load_depth(nonkf_frame_lidar_input_filepath)
        nonkf_frame_height, nonkf_frame_width = nonkf_frame_lidar_image.shape[0:2]
        assert (nonkf_frame_height==1280), f"{nonkf_frame} has incorrect height!"
        assert (nonkf_frame_width==1920), f"{nonkf_frame} has incorrect width!"

        # load bboxes and remove moving objects denoted by bboxes
        nonkf_frame_bboxes = np.load(nonkf_frame_bboxes_input_filepath)

        for box in nonkf_frame_bboxes:
            (center_x, center_y, width, length) = box
            min_x = int(max(center_x - length//2,0))
            min_y = int(max(center_y - width//2,0))
            max_x = int(min(center_x + length //2,nonkf_frame_width))
            max_y = int(min(center_y + width//2,nonkf_frame_height))

            # filter out points inside the bounding boxes, since dynamic objects vary too much
            nonkf_frame_lidar_image[min_y:max_y, min_x:max_x] = 0  # discrete region since int flooring
            
        # Now we need to convert image format to point cloud array (pcl) format (y,x,z)
        lidar_points_nonkf_y, lidar_points_nonkf_x = np.nonzero(nonkf_frame_lidar_image)
        lidar_points_nonkf_z = nonkf_frame_lidar_image[lidar_points_nonkf_y, lidar_points_nonkf_x]

        # NOTE: No need to do the manual mathematical point cloud conversions like in NuScene,
        # Can helper functions later to achieve so, so all we need is just loading stuff in atm.
        # No need to convert to LidarPointCloud object, since not NuScene.
        # Stack image points Nx3 and convert to tensor
        image_points_nonkf = np.column_stack((lidar_points_nonkf_x, lidar_points_nonkf_y, lidar_points_nonkf_z))
        image_points_nonkf = tf.constant(image_points_nonkf, dtype = tf.float32) # type must match extrinsics!

        # Load trinsics-matrices and convert to tensors
        camera_intrinsics_nonkf = np.load(nonkf_frame_cam_intrinsics_input_filepath)
        camera_intrinsics_nonkf = tf.constant(tf.reshape(tf.cast(camera_intrinsics_nonkf, tf.float32)
                                                    , (9, ))  # NOTE: doing (-1, ) instead of -1 in TF leads to weird bug as it thinks the reshape is (3, 3) sometimes,
                                                , dtype = tf.float32)   # flattens to (9, )
        # print("NonKf Shape is now: " + str(camera_intrinsics_nonkf.shape))
        camera_to_vehicle_extrinsics_nonkf = np.load(nonkf_frame_cam_extrinsic_input_filepath)
        camera_to_vehicle_extrinsics_nonkf = tf.constant(camera_to_vehicle_extrinsics_nonkf, dtype = tf.float32)
        
        # Construct nonkf image metadata via pose + metadata.
        frame_metadata_nonkf = np.load(nonkf_frame_metadata_input_filepath)  # [0.0] * 10
        pose_vehicle_to_global_nonkf = np.load(nonkf_frame_veh_pose_input_filepath)
        camera_image_metadata_nonkf = list(tf.reshape(pose_vehicle_to_global_nonkf, -1)) + list(frame_metadata_nonkf)  # velocity and latency filled with zero

        # Source code convention
        metadata = tf.constant([
            nonkf_frame_width,
            nonkf_frame_height,
            open_dataset.CameraCalibration.LEFT_TO_RIGHT,
        ], dtype=tf.int32)

        # NOTE: this avoids the manual mathematical procedures with NuScene object, unlike NuScene
        # NOTE: takes the job of "camera_to_lidar_frame"
        # Call the API function to backproject relative lidar points to 3D pointcloud space in world coordinate.
        # Goes from nonkf frame 2D lidar to nonkf frame 3D world points
        # NOTE: sometimes, with too high of multi-threading load, tf.reshape can fail. If ops call fails,
        # uncomment the two debugging codes below to see. 
        # print("Camera intrinsic shape of input to image_to_world (nonkf): " +   // for debugging
        #     str(camera_intrinsics_nonkf.get_shape()))
        world_points_nonkf = py_camera_model_ops.image_to_world(camera_to_vehicle_extrinsics_nonkf,
                                                        camera_intrinsics_nonkf,
                                                        metadata,
                                                        camera_image_metadata_nonkf,
                                                        image_points_nonkf)
        # reshape to Nx3
        world_points_nonkf = tf.reshape(world_points_nonkf, (-1, 3))

        # Convert the current frame (keyframe)'s trinsics-matrices to tensor
        camera_to_vehicle_extrinsics_kf = tf.constant(extrinsics_camera_to_vehicle_kf, dtype = tf.float32)
        camera_intrinsics_kf = tf.constant(tf.reshape(tf.cast(camera_intrinsics_kf, tf.float32)
                                                    , (9, ))
                                                , dtype = tf.float32)  # flattens to (9, )
        # print("Kf Shape is now: " + str(camera_intrinsics_kf.shape))

        # Loads in 'metadata' again. Guess it's potentially altered by prev helper, so that's why reload...?
        metadata = tf.constant([
            nonkf_frame_width,
            nonkf_frame_height,
            open_dataset.CameraCalibration.LEFT_TO_RIGHT,
        ], dtype=tf.int32)

        # Construct kf image metadata via pose + metadata.
        camera_image_metadata_kf = list(tf.reshape(pose_vehicle_to_global_kf, -1)) + list(frame_metadata_kf)  #[0.0] * 10; velocity and latency filled with zero
    
        # NOTE: takes the job of "point_cloud_to_image"
        # Call the API function to project the world points back/forward to the image frame of reference
        # NOTE: Goes from nonkf (forward) frame 3D world points to nonkf frame 2D lidar points PROJECTED through 
        # kf (current) frame's matrices to reconstruct "where these nonkf lidar points would've been if they linearly
        # go backward/forward in time up (assume static objects) till kf frame's time", resulting in another possible sparse lidar depth map for kf.
        # print("Camera intrinsic shape of input to world_to_image (kf): " +   // for debugging
        #     str(camera_intrinsics_kf.get_shape()))
        image_frame_points_nonkf = py_camera_model_ops.world_to_image(camera_to_vehicle_extrinsics_kf,
                                                                camera_intrinsics_kf,
                                                                metadata,
                                                                camera_image_metadata_kf,
                                                                world_points_nonkf,
                                                                return_depth=True)  # Nx4 [x,y,depth,valid]
    
        # Clean out projected points that are "out of sight" of current kf frame.
        # Extract valid point map: 1 - valid, 0 - invalid
        image_frame_points_nonkf = tf.transpose(image_frame_points_nonkf)
        x_im, y_im, z_im, validity = image_frame_points_nonkf
        valid_mask = tf.cast(validity, dtype=tf.bool)
        if not tf.reduce_any(valid_mask):  # no valid points!
            print(f"Validity Map contais only zeroes for {nonkf_frame}")
            n_nonkf_processed += 1
            continue
        valid_mask_np = valid_mask.numpy()
        x_im_np = x_im.numpy()[valid_mask_np]
        y_im_np = y_im.numpy()[valid_mask_np]
        z_im_np = z_im.numpy()[valid_mask_np]

        '''
        Process keyframe via the filtering process.
        ''' 
        # Create a nonkf lidar image by plotting valid depth values from projection onto the image
        lidar_image_nonkf = np.zeros_like(main_lidar_image)
        for x, y, depth in zip(x_im_np, y_im_np, z_im_np):
            # NOTE: Going to change them here to include 0 index.
            if y >= 0 and y < nonkf_frame_height and x >= 0 and x < nonkf_frame_width:
                lidar_image_nonkf[int(y), int(x)] = depth  # only takes valid depth not out of sight

        # Remove nonkf points projected onto moving object in main kf frame
        for box in bboxes_kf:
            (center_x, center_y, width, length) = box
            min_x = int(max(center_x - length//2, 0))
            min_y = int(max(center_y - width//2, 0))
            max_x = int(min(center_x + length//2, nonkf_frame_width))
            max_y = int(min(center_y + width//2, nonkf_frame_height))
            
            # Filter out points inside the bounding boxes
            lidar_image_nonkf[min_y:max_y, min_x:max_x] = 0
    
        # Convert image format to point cloud format
        lidar_points_nonkf_y, lidar_points_nonkf_x = np.nonzero(lidar_image_nonkf)
        lidar_points_nonkf_z = lidar_image_nonkf[lidar_points_nonkf_y, lidar_points_nonkf_x]
    
        # Stack y and x to 2 x N (x, y)
        lidar_points_nonkf = np.stack([
            lidar_points_nonkf_x,
            lidar_points_nonkf_y],
            axis=0)
        lidar_points_nonkf_filtered_depth = lidar_points_nonkf_z  # filtered at this point.
        lidar_points_main_nonkf_filtered_quantized = np.round(lidar_points_nonkf).astype(int)  # fix to discrete location
    
        # NOTE: "main" as in "Ongoing Current (being updated)"
        # Check for occlusion and update main_lidar based on closer valid points
        for point_idx in range(0, lidar_points_main_nonkf_filtered_quantized.shape[1]):
            x = lidar_points_main_nonkf_filtered_quantized[0, point_idx]
            y = lidar_points_main_nonkf_filtered_quantized[1, point_idx]

            # Not occluded = pixel depth from original/current detection is valid and pixel depth from 
            # nonkf's projection is closer than the existing one at that location (starting off with 
            # detected ground truth and updating via projection estimation comparisons).
            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                lidar_points_nonkf_filtered_depth[point_idx] < main_lidar_image[y, x]

            # NOTE: empty ones will be filled, and closer ones will keep replacing.
            if is_not_occluded:  # Update to projected estimation
                main_lidar_image[y, x] = lidar_points_nonkf_filtered_depth[point_idx]
            elif main_validity_map[y, x] != 1:  # Fail to detect originally, but valid from projection result
                main_lidar_image[y, x] = lidar_points_nonkf_filtered_depth[point_idx]
                main_validity_map[y, x] = 1  # Update to valid.
        
        # Move onto next forward image
        n_nonkf_processed += 1

# Helper function from the Nuscene Code
def get_photometric_error_map(nonkf_frame, 
                                pose_vehicle_to_global_kf, camera_intrinsics_kf, camera_image_kf,
                                points, shape):
    '''
    Arg(s):
        nonkf_frame : str
            path to non keyframe frame.
        pose_vehicle_to_global_kf : numpy.array
            vehicle to global pose tensor of keyframe
        camera_intrinsics_kf : numpy.array
            camera intrinsic tensor of keyframe
        camera_image_kf : nummpy.array
            camera image tensor of keyframe
        points : torch.tensor
            torch tensor of the backprojected groundtruth in 3D
        shape : tuple
            shape of the camera_image_kf for reprojection
    Returns:
        error_map_nonkf, a torch tensor of mean photometric error across channels between nonkf and kf camera image.
    '''

    if nonkf_frame != '':
        camera_image_input_filepath_nonkf = os.path.join(nonkf_frame,'front_camera.jpeg')
        assert os.path.exists(camera_image_input_filepath_nonkf)

        camera_image_nonkf = \
            torch.from_numpy(data_utils.load_image(camera_image_input_filepath_nonkf, data_format='CHW')).unsqueeze(0)

        # Get relative pose from current to backward/forward
        # Get extrinsics matrix from source sensor to global frame
        extrinsics_matrix_kf = pose_vehicle_to_global_kf
        sensor_to_global_rotation_kf = extrinsics_matrix_kf[:3, :3]
        sensor_to_global_translation_kf = extrinsics_matrix_kf[:3, -1]

        # Get extrinsics matrix from target sensor to global frame
        veh_pose_input_filepath_nonkf = os.path.join(nonkf_frame, 'vehicle_to_global_pose.npy')
        extrinsics_matrix_nonkf = np.load(veh_pose_input_filepath_nonkf)
        sensor_to_global_rotation_nonkf = extrinsics_matrix_nonkf[:3, :3]
        sensor_to_global_translation_nonkf = extrinsics_matrix_nonkf[:3, -1]

        # Get relative pose between the two sensors
        relative_rotation = np.matmul(
            np.linalg.inv(sensor_to_global_rotation_nonkf),
            sensor_to_global_rotation_kf)
        relative_translation = np.matmul(
            np.linalg.inv(sensor_to_global_rotation_nonkf),
            (sensor_to_global_translation_kf - sensor_to_global_translation_nonkf))

        # Create 4 x 4 pose matrix
        relative_pose_matrix = np.zeros([4, 4])
        relative_pose_matrix[:3, :3] = relative_rotation
        relative_pose_matrix[:3, -1] = relative_translation
        relative_pose_matrix[-1, -1] = 1.0

        # Convert from numpy to torch tensor
        kf_to_nonkf_pose = relative_pose_matrix.astype(np.float32)
        kf_to_nonkf_pose = torch.from_numpy(kf_to_nonkf_pose)

        # Project points in camera coordinates to 2D pixel coordinates
        target_xy_nonkf = data_utils.project_to_pixel(
            points=points,
            pose=kf_to_nonkf_pose,
            intrinsics=camera_intrinsics_kf,
            shape=shape)

        # Sample the image at x,y locations to target x,y locations.
        reprojected_image_from_nonkf = data_utils.grid_sample(
            image=camera_image_nonkf,
            target_xy=target_xy_nonkf,
            shape=shape,
            padding_mode='zeros')

        # Mean error across channels
        error_map_nonkf = torch.mean(torch.abs(camera_image_kf - reprojected_image_from_nonkf), dim=1)
    else:
        error_map_nonkf = None
    
    return error_map_nonkf

def process_sample(tag_trainvaltest, \
                    segment, \
                    frames, \
                    n_forward_frames_to_reproject, \
                    n_backward_frames_to_reproject, \
                    output_dirpath, \
                    k,
                    paths_only):
    '''
    Process one sample

    Arg(s):
        tag_trainvaltest : str
            training, validation, testing split
        segment: str
            path to the current segment
        frames : List[str]
            list of paths to all frames
        n_forward_frames_to_reproject : int
            number of forward/next nonkf frames to project depth estimations onto the kf frame.
        n_backward_frames_to_reproject : int
            number of backward/prev nonkf frames to project depth estimations onto the kf frame.
        output_dirpath : str
            root of output directory
        k : int
            window size of the k-let concatenation, ex. triplet if k=3.
        paths_only : bool
            if set, then only produce paths
    Output:
        tag_trainvaltest : str
            training, validation, testing split
        tag_daynight : str
            day, night
        camera_image_scene_paths : List[str]
            paths to camera image
        lidar_scene_paths : List[str]
            paths to lidar depth map
        ground_truth_scene_paths : List[str]
            paths to ground truth (merged lidar) depth map
        camera_intrinsics_paths : List[str]
            paths to camera instrinsic tensors
        camera_absolute_pose_paths : List[str]
            paths to camera absolute pose tensors
        camera_image_klet_paths : List[str]
            paths to k-concatenated sequential camera image along dim 1
        lidar_klet_paths : List[str]
            paths to k-concatenated sequential lidar depth map along dim 1
        ground_truth_klet_paths : List[str]
            paths to k-concatenated sequential gt depth map along dim 1
    '''
    camera_image_paths = []
    lidar_paths = []
    ground_truth_paths = []
    camera_intrinsics_paths = []
    camera_absolute_pose_paths = []
    # NOTE: we add klet generation here because we are only parallelizing processing at segment
    # granularity, so frames will be sequentially handled anyways (already natsorted).
    camera_image_klet_paths = []  # an example of k-let is a triplet, when k=3.
    lidar_klet_paths = []
    ground_truth_klet_paths = []

    '''
    Set up output directories
    '''
    segment_name = os.path.basename(segment)
    # Create output image path
    image_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'image', segment_name)
    os.makedirs(image_output_dirpath, exist_ok=True)
    # Create output lidar path
    lidar_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'lidar', segment_name)
    os.makedirs(lidar_output_dirpath, exist_ok=True)
    # Create output ground truth path
    ground_truth_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'ground_truth', segment_name)
    os.makedirs(ground_truth_output_dirpath, exist_ok=True)
    # Create output intrinsics path
    intrinsics_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'intrinsics', segment_name)
    os.makedirs(intrinsics_output_dirpath, exist_ok=True)
    # Create output pose path
    pose_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, "pose", segment_name)
    os.makedirs(pose_output_dirpath, exist_ok=True)
    # Create output image klet path
    image_klet_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'image_klet', segment_name)
    os.makedirs(image_klet_output_dirpath, exist_ok=True)
    # Create output lidar klet path
    lidar_klet_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'lidar_klet', segment_name)
    os.makedirs(lidar_klet_output_dirpath, exist_ok=True)
    # Create output ground truth klet path
    ground_truth_klet_output_dirpath = os.path.join(output_dirpath, tag_trainvaltest, 'ground_truth_klet', segment_name)
    os.makedirs(ground_truth_klet_output_dirpath, exist_ok=True)

    # Waymo doens't have detection dirpath for a scene in json.

    print('Processing segment {}'.format(segment))  # use full dirpath for now.

    '''
    Process each frame
    '''
    # Create a moving buffer such that after processing k'th frame, we store the first frame's concatenations
    # prev_info = [{'img_klet_savepath': None, 'image': None, 
    #                 'lidar_klet_savepath': None, 'lidar': None,
    #                 'gt_klet_savepath': None, 'gt': None}] * k
    prev_info = []
    # Iterate through all frames up to the last frame 
    i = 0
    while i < len(frames):

        # Fetch a single frame
        # NOTE: each frame is a directory containing 5 files describing the info at that frame
        # won't name it "frame_directory" or "frame_folder" for the ease of understanding.
        current_frame = frames[i]
        
        '''
        Set up file paths/names of current frame
        '''
        # Create input file paths of current frame. Lidar = sparse depth
        lidar_input_filepath, \
        image_input_filepath, \
        veh_pose_input_filepath, \
        cam_extrinsic_input_filepath, \
        cam_intrinsics_input_filepath, \
        bboxes_input_filepath, \
        metadata_input_filepath, \
        scene_label_input_filepath = read_frame_data(current_frame)

        # Obtain frame's day/night tag
        tag_daynight = get_time_of_day(scene_label_input_filepath)

        # Create output file paths/names of current frame
        frame_name = os.path.basename(current_frame)  # frame_dirname
        # Image
        __, ext = os.path.splitext(image_input_filepath)
        image_output_filepath = os.path.join(image_output_dirpath,
                                     frame_name + ext)
        # Lidar
        __, ext = os.path.splitext(lidar_input_filepath)
        lidar_output_filepath = os.path.join(lidar_output_dirpath,
                                     frame_name + ext)
        # Ground Truth (not given in dataset; reconstructed via merging lidar points)
        ext = '.png'  # the usual ext for lidar images
        ground_truth_output_filepath = os.path.join(ground_truth_output_dirpath,
                                            frame_name + ext)
        # Camera Intrinsics
        __, ext = os.path.splitext(cam_intrinsics_input_filepath)
        intrinsics_output_filepath = os.path.join(intrinsics_output_dirpath,
                                        frame_name + ext)
        # Camera Absolute Pose
        __, ext = os.path.splitext(veh_pose_input_filepath)
        pose_output_filepath = os.path.join(pose_output_dirpath,
                                     frame_name + ext)
        # Image Triplet
        __, ext = os.path.splitext(image_input_filepath)
        image_klet_output_filepath = os.path.join(image_klet_output_dirpath,
                                        frame_name + ext)
        # Lidar (Sparse depth) Triplet
        __, ext = os.path.splitext(lidar_input_filepath)
        lidar_klet_output_filepath = os.path.join(lidar_klet_output_dirpath,
                                     frame_name + ext)
        # Ground truth (dense depth) Triplet
        ext = '.png'
        ground_truth_klet_output_filepath = os.path.join(ground_truth_klet_output_dirpath,
                                            frame_name + ext)
        # No need to care for race-condition of multithreadding since exist_ok=True

        '''
        Store file paths
        '''
        camera_image_paths.append(image_output_filepath)
        lidar_paths.append(lidar_output_filepath)
        ground_truth_paths.append(ground_truth_output_filepath)
        camera_intrinsics_paths.append(intrinsics_output_filepath)
        camera_absolute_pose_paths.append(pose_output_filepath)
        camera_image_klet_paths.append(image_klet_output_filepath)
        lidar_klet_paths.append(lidar_klet_output_filepath)
        ground_truth_klet_paths.append(ground_truth_klet_output_filepath)
        
        if not paths_only:
            '''
            Save camera image, sparse depth, and camera pose data directly provided by the dataset
            '''
            # Save image to new directory; no need to calculate, directly provided
            shutil.copy(image_input_filepath, image_output_filepath)  # jpg

            # Save sparse depth (lidar) to new directory; no need to calculate, directly provided
            shutil.copy(lidar_input_filepath, lidar_output_filepath)  # png

            # Save absolute pose to new directory; no need to calculate, directly provided
            # NOTE: nuscene's abs pose has extrinsic matrix's involvement, why not here?
            shutil.copy(veh_pose_input_filepath, pose_output_filepath)

            '''
            Loads in required data of the current/kf frame
            '''
            # Loads current sparse depth image
            main_lidar_image = load_depth(lidar_input_filepath)
            main_validity_map = np.where(main_lidar_image > 0, 1, 0)  # masks out negative depths (invalid)

            # Loads in pose transformation np matrix for vehicle -> global
            pose_vehicle_to_global_kf = np.load(veh_pose_input_filepath)

            # Loads in extrinsics transformation np matrix for camera -> vehicle
            extrinsics_camera_to_vehicle_kf = np.load(cam_extrinsic_input_filepath)

            # Loads in front camera intrinsics matrix
            camera_intrinsics_kf = np.load(cam_intrinsics_input_filepath)

            # Create the intrinsics matrix (NOTE: directly-grabbed for nuscene); Process from old code.
            # Fixes intrinsics to save for depth completion models' use
            adjusted_intrinsics_kf = np.zeros((3, 3))
            adjusted_intrinsics_kf[0,0] = camera_intrinsics_kf[0,0]
            adjusted_intrinsics_kf[1,1] = camera_intrinsics_kf[0,1]
            adjusted_intrinsics_kf[0,2] = camera_intrinsics_kf[0,2]
            adjusted_intrinsics_kf[1,2] = camera_intrinsics_kf[1,0]
            adjusted_intrinsics_kf[0,1] = camera_intrinsics_kf[2,2]  # the skew, usually 0
            adjusted_intrinsics_kf[2,2] = 1

            # Save intrinsics to new directory (no need for if not exists)
            np.save(intrinsics_output_filepath, adjusted_intrinsics_kf)

            # Loads in frame metadata
            frame_metadata_kf = np.load(metadata_input_filepath)  # [0.0] * 10  

            # Loads in current frame bounding boxes
            bboxes_kf = np.load(bboxes_input_filepath)

            '''
            Merge forward and backward point clouds for lidar to construct a dense depth
            '''
            # Merges n_forward and n_backward number of point clouds to frame at sample token
            # Grab all forward and backward frames of current frame
            # Need to account for cases where there's not enough backward frames or forward frames!
            if i < n_backward_frames_to_reproject:
                backward_frames = frames[:i]  # NOTE: excluding current_frame
            else:
                backward_frames = frames[i-n_backward_frames_to_reproject:i]
            if i + n_forward_frames_to_reproject >= len(frames):
                forward_frames = frames[i+1:]
            else:
                forward_frames = frames[i+1:i+n_forward_frames_to_reproject+1]

            # Merge forward frames onto main_lidar
            merge_pointcloud_unidirectional(forward_frames, 
                                    main_lidar_image,
                                    main_validity_map,
                                    pose_vehicle_to_global_kf,
                                    extrinsics_camera_to_vehicle_kf,
                                    camera_intrinsics_kf, 
                                    bboxes_kf,
                                    frame_metadata_kf)
            
            # Merge backward frames onto main_lidar
            merge_pointcloud_unidirectional(backward_frames, 
                                    main_lidar_image,
                                    main_validity_map,
                                    pose_vehicle_to_global_kf,
                                    extrinsics_camera_to_vehicle_kf,
                                    camera_intrinsics_kf, 
                                    bboxes_kf,
                                    frame_metadata_kf)
            
            # By this point, main_lidar_image has finished reconstructing (stacking of projections)
            # No need to carry out "points_to_depth_map" since main_lidar_image is already quantized and constructed that way.
            ground_truth_kf = main_lidar_image

            # Filter the dense lidar points using photometric error (Almost same code as in Nuscenes)
            if args.filter_threshold_photometric_reconstruction > 0:

                # Load camera image
                camera_image_kf = data_utils.load_image(image_input_filepath)  # the permute below does the job of data_format='CHW'

                # Convert to torch to utilize existing backprojection function
                ground_truth_kf = torch.from_numpy(ground_truth_kf).unsqueeze(0).float()
                camera_image_kf = torch.from_numpy(camera_image_kf).permute(2, 0, 1).unsqueeze(0).float()
                camera_intrinsics_kf = adjusted_intrinsics_kf  # Need the actual K matrix
                camera_intrinsics_kf = torch.from_numpy(camera_intrinsics_kf).unsqueeze(0).float()

                # Backproject ground truth points to 3D
                shape = camera_image_kf.shape
                points = data_utils.backproject_to_camera(ground_truth_kf, camera_intrinsics_kf, shape)

                # Get the forward/previous and backward/next nonkf frames
                backward_frame = frames[i-1] if not i <= 0 else None
                forward_frame = frames[i+1] if not i >= len(frames)-1 else None

                # Obtain the error maps, which should have same dim as ground truth
                error_map_backward = get_photometric_error_map(backward_frame, 
                                    pose_vehicle_to_global_kf, camera_intrinsics_kf, camera_image_kf,
                                    points, shape)
                error_map_forward = get_photometric_error_map(forward_frame, 
                                    pose_vehicle_to_global_kf, camera_intrinsics_kf, camera_image_kf,
                                    points, shape)
                
                # Check if we are on either ends of a sequence, if not then we choose the minimum error
                if error_map_backward is None:
                    error_map = error_map_forward
                elif error_map_forward is None:
                    error_map = error_map_backward
                elif error_map_backward is not None and error_map_forward is not None:
                    error_map, error_map_min_indices = torch.min(torch.cat([error_map_backward, error_map_forward]), dim=0, keepdim=True)
                else:
                    raise ValueError('Both forward and backward reprojected error maps are None.')

                # Convert error map to numpy format
                error_map = error_map.numpy()

                # Apply the fliter threshold to construct a validity mask
                photometric_valid_mask = np.where(
                    error_map < args.filter_threshold_photometric_reconstruction,
                    np.ones_like(error_map),
                    np.zeros_like(error_map))

                # Mask out ground truth pixels with too much photometric error
                ground_truth_kf = ground_truth_kf * photometric_valid_mask

                ground_truth_kf = np.squeeze(ground_truth_kf.numpy())

            # Filter the dense lidar points using outlier detection (Almost same code as in Nuscenes)
            if args.enable_outlier_removal:
                # Filter outliers from ground truth based on differences in neighboring points
                # Note: need 1st unsqueeze for the m=1 batchsize, and 2nd unsqueeze for z-axis scalar. Just input convention.
                ground_truth_kf = torch.from_numpy(ground_truth_kf).unsqueeze(0).unsqueeze(0).float()
                validity_map_ground_truth = torch.where(
                    ground_truth_kf > 0,
                    torch.ones_like(ground_truth_kf),
                    torch.zeros_like(ground_truth_kf))

                # Remove the outlier predictions/depths
                ground_truth_kf, _ = outlier_removal.remove_outliers(
                    sparse_depth=ground_truth_kf,
                    validity_map=validity_map_ground_truth)
                
                ground_truth_kf = np.squeeze(ground_truth_kf.numpy())

            # Save the dense depth (ground truth) of main/current/kf frame to new directory
            # Alternatively can use data_utils.save_depth from NuScene
            data_utils.save_depth(ground_truth_kf, ground_truth_output_filepath)

            if k > 0:
                # Generate and save the klets
                # 1) Generate klet and enqueue self in
                image_kf = data_utils.load_image(image_input_filepath, normalize=False)  # format is HWC by default
                lidar_kf = load_depth(lidar_input_filepath, multiplier=1)
                new_klet = {'img_klet_savepath': image_klet_output_filepath, 'image': image_kf, 
                            'lidar_klet_savepath': lidar_klet_output_filepath, 'lidar': lidar_kf,
                            'gt_klet_savepath': ground_truth_klet_output_filepath, 'gt': ground_truth_kf}
                prev_info.append(new_klet)
                # 2) If queue is at size k, then create the klets (0, ..., k-1) and save, then kick out oldest
                if len(prev_info) == k:
                    # Generate klets for idx 0 elem
                    k_images = [klet['image'] for klet in prev_info]
                    image_klet = np.concatenate(k_images, axis=1)  # H x kW x c
                    k_lidars = [klet['lidar'] for klet in prev_info]
                    lidar_klet = np.concatenate(k_lidars, axis=1)  # H x kW x c
                    k_gts = [klet['gt'] for klet in prev_info]
                    gt_klet = np.concatenate(k_gts, axis=1)  # H x kW x c
                    # Save the klets to new directories
                    oldest_klet = prev_info.pop(0)  # kick out the (oldest) one that we are processing
                    img_klet_savepath = oldest_klet['img_klet_savepath']
                    lidar_klet_savepath = oldest_klet['lidar_klet_savepath']
                    gt_klet_savepath = oldest_klet['gt_klet_savepath']
                    data_utils.save_image(image_klet, img_klet_savepath, normalized=False)  # load untouched, save untouched.
                    data_utils.save_depth(lidar_klet, lidar_klet_savepath, multiplier=1)
                    data_utils.save_depth(gt_klet, gt_klet_savepath)  # need to apply actual upscaling multiplier here
                    
        i += 1  # moving onto next frame. Important!

    print('Finished {} frames in segment {}'.format(len(frames), segment_name))
    
    return tag_trainvaltest, \
            tag_daynight, \
            camera_image_paths, \
            lidar_paths, \
            ground_truth_paths, \
            camera_intrinsics_paths, \
            camera_absolute_pose_paths, \
            camera_image_klet_paths, \
            lidar_klet_paths, \
            ground_truth_klet_paths
            
'''
Main function
'''
if __name__ == '__main__':
    # Set up variables for easier arg reference
    waymo_data_root_dirpath = args.waymo_data_root_dirpath
    waymo_data_derived_dirpath = args.waymo_data_derived_dirpath
    n_forward_frames_to_reproject = args.n_forward_frames_to_reproject
    n_backward_frames_to_reproject = args.n_backward_frames_to_reproject
    n_threads = args.n_thread
    paths_only = args.paths_only
    use_multithread = args.n_thread > 1 and not args.debug
    concatenation_block_size = args.concatenation_block_size

    # Training all, day, and night paths
    train_camera_image_paths = []
    train_lidar_paths = []
    train_ground_truth_paths = []
    train_intrinsics_paths = []
    train_absolute_pose_paths = []
    train_camera_image_klet_paths = []
    train_lidar_klet_paths = []
    train_ground_truth_klet_paths = []

    train_day_camera_image_paths = []
    train_day_lidar_paths = []
    train_day_ground_truth_paths = []
    train_day_intrinsics_paths = []
    train_day_absolute_pose_paths = []
    train_day_camera_image_klet_paths = []
    train_day_lidar_klet_paths = []
    train_day_ground_truth_klet_paths = []

    train_night_camera_image_paths = []
    train_night_lidar_paths = []
    train_night_ground_truth_paths = []
    train_night_intrinsics_paths = []
    train_night_absolute_pose_paths = []
    train_night_camera_image_klet_paths = []
    train_night_lidar_klet_paths = []
    train_night_ground_truth_klet_paths = []

    # Validation all, day, and night paths
    val_camera_image_paths = []
    val_lidar_paths = []
    val_ground_truth_paths = []
    val_intrinsics_paths = []
    val_absolute_pose_paths = []
    val_camera_image_klet_paths = []
    val_lidar_klet_paths = []
    val_ground_truth_klet_paths = []

    val_day_camera_image_paths = []
    val_day_lidar_paths = []
    val_day_ground_truth_paths = []
    val_day_intrinsics_paths = []
    val_day_absolute_pose_paths = []
    val_day_camera_image_klet_paths = []
    val_day_lidar_klet_paths = []
    val_day_ground_truth_klet_paths = []

    val_night_camera_image_paths = []
    val_night_lidar_paths = []
    val_night_ground_truth_paths = []
    val_night_intrinsics_paths = []
    val_night_absolute_pose_paths = []
    val_night_camera_image_klet_paths = []
    val_night_lidar_klet_paths = []
    val_night_ground_truth_klet_paths = []

    # Testing all, day, and night paths
    test_camera_image_paths = []
    test_lidar_paths = []
    test_ground_truth_paths = []
    test_intrinsics_paths = []
    test_absolute_pose_paths = []
    test_camera_image_klet_paths = []
    test_lidar_klet_paths = []
    test_ground_truth_klet_paths = []

    test_day_camera_image_paths = []
    test_day_lidar_paths = []
    test_day_ground_truth_paths = []
    test_day_intrinsics_paths = []
    test_day_absolute_pose_paths = []
    test_day_camera_image_klet_paths = []
    test_day_lidar_klet_paths = []
    test_day_ground_truth_klet_paths = []

    test_night_camera_image_paths = []
    test_night_lidar_paths = []
    test_night_ground_truth_paths = []
    test_night_intrinsics_paths = []
    test_night_absolute_pose_paths = []
    test_night_camera_image_klet_paths = []
    test_night_lidar_klet_paths = []
    test_night_ground_truth_klet_paths = []

    # Going to embed the n_frames counting process into the unwrapping forloop for cleaner code
    # NOTE: datasplit w/ ids thing is exclusively needed to NuScene, so not included for Waymo
    # since NuScene uses ids in json files to identify which frame belongs to which, whereas Waymo
    # comes in segements and frames w/ directory structures that we have to unwrap.
    trainvaltest_frames_counter = [0, 0, 0]
    daynight_frames_counter = [0, 0]
    
    os.makedirs(waymo_data_derived_dirpath,exist_ok=True)
    # NOTE: making the folders first then files individually during process_scene
    # is the same as making folder+file at once in nuscene's logic. But will stick to the latter.

    # No need to use scene index or check tags because unlike NuScene with is a jumbo directory with
    # txt specifying the files, Waymo already splits them into directories for you.
    # Also don't think need to make derived dirpaths here; can make them during frame processing.

    # A concept from NuScene to post-process ground truth, introduced through an optional argument
    if args.enable_outlier_removal:  # is True
        # Create outlier removal object shared by all threads
        outlier_removal = data_utils.OutlierRemoval(
            kernel_size=args.outlier_removal_kernel_size,
            threshold=args.outlier_removal_threshold)

    # Process each datasplit directory in the root directory of Waymo dataset
    # Sticking with this way of pooling, feels cleaner
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:  # prepare parallelism
        time_total_start = time.time()
        
        inputs_pool = []  # a pool of inputs for deferred execution
        pool_results = []  # NOTE: on segment-level, the output list
        data_split_dirnames = ["training", "validation", "testing"]

        # print('Setting up scene processor and reading in statistics...')

        for index, split_trainvaltest in enumerate(data_split_dirnames):
            # NOTE: the root path should contain training, validation, and testing folders, or partially.
            # Quick check to see if the datasplit directory exists, skipping over if not.
            data_split_dirpath = os.path.join(waymo_data_root_dirpath,split_trainvaltest)
            if not os.path.exists(data_split_dirpath):
                print(data_split_dirpath + " does not exist. Skipped.")
                continue

            # Grab all the video samples/segments in the data_split directories
            segments_dirpaths = os.path.join(data_split_dirpath,'segment-*')
            segments = glob.glob(segments_dirpaths)  # the list of paths matching the pattern above
            tag_trainvaltest = split_trainvaltest

            # Queue each segment to be processed
            for segment in segments:
                frames_dirpaths = glob.glob(os.path.join(segment,'frame_*'))   # sample segment's frame directories
                frames = natsorted(frames_dirpaths)  # need to reorder since out of order.
                
                # Grab count statistics
                trainvaltest_frames_counter[index] += len(frames)
                for frame in frames:
                    scene_label_input_filepath = os.path.join(frame, 'scene_label.npy')
                    tag_daynight = get_time_of_day(scene_label_input_filepath)
                    if tag_daynight == 'day':
                        daynight_frames_counter[0] += 1
                    else:
                        daynight_frames_counter[1] += 1
                
                # Append the inputs for processing this segment's frames into the pool
                inputs = (
                        tag_trainvaltest,
                        segment,
                        frames,
                        n_forward_frames_to_reproject,
                        n_backward_frames_to_reproject,
                        waymo_data_derived_dirpath,
                        concatenation_block_size,  # k
                        paths_only
                )
                # inputs_pool.append(inputs)

                # Start the processing directly, for visible feedback
                # NOTE: alternative: put it here to process as it reads to avoid stackoverflow? But python stores stuff on heap so should be fine.
                if not use_multithread:  # sequential execution
                    pool_results.append(process_sample(*inputs))
                else: # add to a pool of threads
                    process_frame_output = executor.submit(process_sample, *inputs)
                    pool_results.append(process_frame_output)  # future

        # Announce the statistics
        print('Total Scenes to process: {}'.format(sum(trainvaltest_frames_counter)))
        print('Training: {}  Validation: {}  Testing: {}'.format(trainvaltest_frames_counter[0], 
                                                                 trainvaltest_frames_counter[1], 
                                                                 trainvaltest_frames_counter[2]))
        print('Number of daytime scene ids: {}'.format(daynight_frames_counter[0]))
        print('Number of nighttime scene ids: {}'.format(daynight_frames_counter[1]))

        # Start the processing
        # for inputs in inputs_pool:
        #     if not use_multithread:  # sequential execution
        #         pool_results.append(process_sample(*inputs))
        #     else: # add to a pool of threads
        #         process_frame_output = executor.submit(process_sample, *inputs)
        #         pool_results.append(process_frame_output)  # future

        if use_multithread:  # Main thread waits for the other threads to be finished first.
            concurrent.futures.wait(pool_results)  # "join" the threads, if applicable
        
        # At this point, the results should all be ready.
        # Unpack results  (NOTE: everything below pretty much the same, with some var name corrections)
        for results in pool_results:
            if use_multithread:
                results = results.result()  # executor object syntax

            tag_trainvaltest, \
                tag_daynight, \
                camera_image_scene_paths, \
                lidar_scene_paths, \
                ground_truth_scene_paths, \
                camera_intrinsics_paths, \
                camera_absolute_pose_paths, \
                camera_image_klet_paths, \
                lidar_klet_paths, \
                ground_truth_klet_paths = results

            if tag_trainvaltest == 'training':
                train_camera_image_paths.extend(camera_image_scene_paths)
                train_lidar_paths.extend(lidar_scene_paths)
                train_ground_truth_paths.extend(ground_truth_scene_paths)
                train_intrinsics_paths.extend(camera_intrinsics_paths)
                train_absolute_pose_paths.extend(camera_absolute_pose_paths)
                train_camera_image_klet_paths.extend(camera_image_klet_paths)
                train_lidar_klet_paths.extend(lidar_klet_paths)
                train_ground_truth_klet_paths.extend(ground_truth_klet_paths)

                if tag_daynight == 'day':
                    train_day_camera_image_paths.extend(camera_image_scene_paths)
                    train_day_lidar_paths.extend(lidar_scene_paths)
                    train_day_ground_truth_paths.extend(ground_truth_scene_paths)
                    train_day_intrinsics_paths.extend(camera_intrinsics_paths)
                    train_day_absolute_pose_paths.extend(camera_absolute_pose_paths)
                    train_day_camera_image_klet_paths.extend(camera_image_klet_paths)
                    train_day_lidar_klet_paths.extend(lidar_klet_paths)
                    train_day_ground_truth_klet_paths.extend(ground_truth_klet_paths)
                elif tag_daynight == 'night':
                    train_night_camera_image_paths.extend(camera_image_scene_paths)
                    train_night_lidar_paths.extend(lidar_scene_paths)
                    train_night_ground_truth_paths.extend(ground_truth_scene_paths)
                    train_night_intrinsics_paths.extend(camera_intrinsics_paths)
                    train_night_absolute_pose_paths.extend(camera_absolute_pose_paths)
                    train_night_camera_image_klet_paths.extend(camera_image_klet_paths)
                    train_night_lidar_klet_paths.extend(lidar_klet_paths)
                    train_night_ground_truth_klet_paths.extend(ground_truth_klet_paths)
                else:
                    raise ValueError('Found invalid daynight tag: {}'.format(tag_daynight))

            elif tag_trainvaltest == 'validation':
                val_camera_image_paths.extend(camera_image_scene_paths)
                val_lidar_paths.extend(lidar_scene_paths)
                val_ground_truth_paths.extend(ground_truth_scene_paths)
                val_intrinsics_paths.extend(camera_intrinsics_paths)
                val_absolute_pose_paths.extend(camera_absolute_pose_paths)
                val_camera_image_klet_paths.extend(camera_image_klet_paths)
                val_lidar_klet_paths.extend(lidar_klet_paths)
                val_ground_truth_klet_paths.extend(ground_truth_klet_paths)

                if tag_daynight == 'day':
                    val_day_camera_image_paths.extend(camera_image_scene_paths)
                    val_day_lidar_paths.extend(lidar_scene_paths)
                    val_day_ground_truth_paths.extend(ground_truth_scene_paths)
                    val_day_intrinsics_paths.extend(camera_intrinsics_paths)
                    val_day_absolute_pose_paths.extend(camera_absolute_pose_paths)
                    val_day_camera_image_klet_paths.extend(camera_image_klet_paths)
                    val_day_lidar_klet_paths.extend(lidar_klet_paths)
                    val_day_ground_truth_klet_paths.extend(ground_truth_klet_paths)
                elif tag_daynight == 'night':
                    val_night_camera_image_paths.extend(camera_image_scene_paths)
                    val_night_lidar_paths.extend(lidar_scene_paths)
                    val_night_ground_truth_paths.extend(ground_truth_scene_paths)
                    val_night_intrinsics_paths.extend(camera_intrinsics_paths)
                    val_night_absolute_pose_paths.extend(camera_absolute_pose_paths)
                    val_night_camera_image_klet_paths.extend(camera_image_klet_paths)
                    val_night_lidar_klet_paths.extend(lidar_klet_paths)
                    val_night_ground_truth_klet_paths.extend(ground_truth_klet_paths)
                else:
                    raise ValueError('Found invalid daynight tag: {}'.format(tag_daynight))

            elif tag_trainvaltest == 'testing':
                test_camera_image_paths.extend(camera_image_scene_paths)
                test_lidar_paths.extend(lidar_scene_paths)
                test_ground_truth_paths.extend(ground_truth_scene_paths)
                test_intrinsics_paths.extend(camera_intrinsics_paths)
                test_absolute_pose_paths.extend(camera_absolute_pose_paths)
                test_camera_image_klet_paths.extend(camera_image_klet_paths)
                test_lidar_klet_paths.extend(lidar_klet_paths)
                test_ground_truth_klet_paths.extend(ground_truth_klet_paths)

                if tag_daynight == 'day':
                    test_day_camera_image_paths.extend(camera_image_scene_paths)
                    test_day_lidar_paths.extend(lidar_scene_paths)
                    test_day_ground_truth_paths.extend(ground_truth_scene_paths)
                    test_day_intrinsics_paths.extend(camera_intrinsics_paths)
                    test_day_absolute_pose_paths.extend(camera_absolute_pose_paths)
                    test_day_camera_image_klet_paths.extend(camera_image_klet_paths)
                    test_day_lidar_klet_paths.extend(lidar_klet_paths)
                    test_day_ground_truth_klet_paths.extend(ground_truth_klet_paths)
                elif tag_daynight == 'night':
                    test_night_camera_image_paths.extend(camera_image_scene_paths)
                    test_night_lidar_paths.extend(lidar_scene_paths)
                    test_night_ground_truth_paths.extend(ground_truth_scene_paths)
                    test_night_intrinsics_paths.extend(camera_intrinsics_paths)
                    test_night_absolute_pose_paths.extend(camera_absolute_pose_paths)
                    test_night_camera_image_klet_paths.extend(camera_image_klet_paths)
                    test_night_lidar_klet_paths.extend(lidar_klet_paths)
                    test_night_ground_truth_klet_paths.extend(ground_truth_klet_paths)
                else:
                    raise ValueError('Found invalid daynight tag: {}'.format(tag_daynight))

            else:
                raise ValueError('Found invalid tag: {}'.format(tag_trainvaltest))

        time_total_end = time.time()
        print(f"Sample processing completed. Total time spent: {time_total_end - time_total_start} seconds.")
        
        '''
        Subsample from validation and testing set (with interval of 2)
        '''
        # Validation set
        val_camera_image_subset_paths = val_camera_image_paths[::2]
        val_lidar_subset_paths = val_lidar_paths[::2]
        val_ground_truth_subset_paths = val_ground_truth_paths[::2]
        val_intrinsics_subset_paths = val_intrinsics_paths[::2]
        val_absolute_pose_subset_paths = val_absolute_pose_paths[::2]
        val_camera_image_klet_subset_paths = val_camera_image_klet_paths[::2]
        val_lidar_klet_subset_paths = val_lidar_klet_paths[::2]
        val_ground_truth_klet_subset_paths = val_ground_truth_klet_paths[::2]

        val_day_camera_image_subset_paths = val_day_camera_image_paths[::2]
        val_day_lidar_subset_paths = val_day_lidar_paths[::2]
        val_day_ground_truth_subset_paths = val_day_ground_truth_paths[::2]
        val_day_intrinsics_subset_paths = val_day_intrinsics_paths[::2]
        val_day_absolute_pose_subset_paths = val_day_absolute_pose_paths[::2]
        val_day_camera_image_klet_subset_paths = val_day_camera_image_klet_paths[::2]
        val_day_lidar_klet_subset_paths = val_day_lidar_klet_paths[::2]
        val_day_ground_truth_klet_subset_paths = val_day_ground_truth_klet_paths[::2]

        val_night_camera_image_subset_paths = val_night_camera_image_paths[::2]
        val_night_lidar_subset_paths = val_night_lidar_paths[::2]
        val_night_ground_truth_subset_paths = val_night_ground_truth_paths[::2]
        val_night_intrinsics_subset_paths = val_night_intrinsics_paths[::2]
        val_night_absolute_pose_subset_paths = val_night_absolute_pose_paths[::2]
        val_night_camera_image_klet_subset_paths = val_night_camera_image_klet_paths[::2]
        val_night_lidar_klet_subset_paths = val_night_lidar_klet_paths[::2]
        val_night_ground_truth_klet_subset_paths = val_night_ground_truth_klet_paths[::2]

        # Testing set
        test_camera_image_subset_paths = test_camera_image_paths[::2]
        test_lidar_subset_paths = test_lidar_paths[::2]
        test_ground_truth_subset_paths = test_ground_truth_paths[::2]
        test_intrinsics_subset_paths = test_intrinsics_paths[::2]
        test_absolute_pose_subset_paths = test_absolute_pose_paths[::2]
        test_camera_image_klet_subset_paths = test_camera_image_klet_paths[::2]
        test_lidar_klet_subset_paths = test_lidar_klet_paths[::2]
        test_ground_truth_klet_subset_paths = test_ground_truth_klet_paths[::2]

        test_day_camera_image_subset_paths = test_day_camera_image_paths[::2]
        test_day_lidar_subset_paths = test_day_lidar_paths[::2]
        test_day_ground_truth_subset_paths = test_day_ground_truth_paths[::2]
        test_day_intrinsics_subset_paths = test_day_intrinsics_paths[::2]
        test_day_absolute_pose_subset_paths = test_day_absolute_pose_paths[::2]
        test_day_camera_image_klet_subset_paths = test_day_camera_image_klet_paths[::2]
        test_day_lidar_klet_subset_paths = test_day_lidar_klet_paths[::2]
        test_day_ground_truth_klet_subset_paths = test_day_ground_truth_klet_paths[::2]

        test_night_camera_image_subset_paths = test_night_camera_image_paths[::2]
        test_night_lidar_subset_paths = test_night_lidar_paths[::2]
        test_night_ground_truth_subset_paths = test_night_ground_truth_paths[::2]
        test_night_intrinsics_subset_paths = test_night_intrinsics_paths[::2]
        test_night_absolute_pose_subset_paths = test_night_absolute_pose_paths[::2]
        test_night_camera_image_klet_subset_paths = test_night_camera_image_klet_paths[::2]
        test_night_lidar_klet_subset_paths = test_night_lidar_klet_paths[::2]
        test_night_ground_truth_klet_subset_paths = test_night_ground_truth_klet_paths[::2]

        '''
        Write paths to file
        '''
        outputs = [
            [
                'training',
                [
                    [
                        'image',
                        train_camera_image_paths,
                        TRAIN_IMAGE_FILEPATH
                    ], [
                        'lidar',
                        train_lidar_paths,
                        TRAIN_LIDAR_FILEPATH
                    ], [
                        'ground truth',
                        train_ground_truth_paths,
                        TRAIN_GROUND_TRUTH_FILEPATH
                    ], [
                        'intrinsics',
                        train_intrinsics_paths,
                        TRAIN_INTRINSICS_FILEPATH
                    ], [
                        'absolute pose',
                        train_absolute_pose_paths,
                        TRAIN_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'image klet',
                        train_camera_image_klet_paths,
                        TRAIN_IMAGE_KLET_FILEPATH
                    ], [
                        'lidar klet',
                        train_lidar_klet_paths,
                        TRAIN_LIDAR_KLET_FILEPATH
                    ], [
                        'ground truth klet',
                        train_ground_truth_klet_paths,
                        TRAIN_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'day image',
                        train_day_camera_image_paths,
                        TRAIN_DAY_IMAGE_FILEPATH
                    ], [
                        'day lidar',
                        train_day_lidar_paths,
                        TRAIN_DAY_LIDAR_FILEPATH
                    ], [
                        'day ground truth',
                        train_day_ground_truth_paths,
                        TRAIN_DAY_GROUND_TRUTH_FILEPATH
                    ], [
                        'day intrinsics',
                        train_day_intrinsics_paths,
                        TRAIN_DAY_INTRINSICS_FILEPATH
                    ], [
                        'day absolute pose',
                        train_day_absolute_pose_paths,
                        TRAIN_DAY_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'day image klet',
                        train_day_camera_image_klet_paths,
                        TRAIN_DAY_IMAGE_KLET_FILEPATH
                    ], [
                        'day lidar klet',
                        train_day_lidar_klet_paths,
                        TRAIN_DAY_LIDAR_KLET_FILEPATH
                    ], [
                        'day ground truth klet',
                        train_day_ground_truth_klet_paths,
                        TRAIN_DAY_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'night image',
                        train_night_camera_image_paths,
                        TRAIN_NIGHT_IMAGE_FILEPATH
                    ], [
                        'night lidar',
                        train_night_lidar_paths,
                        TRAIN_NIGHT_LIDAR_FILEPATH
                    ], [
                        'night ground truth',
                        train_night_ground_truth_paths,
                        TRAIN_NIGHT_GROUND_TRUTH_FILEPATH
                    ], [
                        'night intrinsics',
                        train_night_intrinsics_paths,
                        TRAIN_NIGHT_INTRINSICS_FILEPATH
                    ], [
                        'night absolute pose',
                        train_night_absolute_pose_paths,
                        TRAIN_NIGHT_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'night image klet',
                        train_night_camera_image_klet_paths,
                        TRAIN_NIGHT_IMAGE_KLET_FILEPATH
                    ], [
                        'night lidar klet',
                        train_night_lidar_klet_paths,
                        TRAIN_NIGHT_LIDAR_KLET_FILEPATH
                    ], [
                        'night ground truth klet',
                        train_night_ground_truth_klet_paths,
                        TRAIN_NIGHT_GROUND_TRUTH_KLET_FILEPATH
                    ],
                ]
            ], [
                'validation',
                [
                    [
                        'image',
                        val_camera_image_paths,
                        VAL_IMAGE_FILEPATH
                    ], [
                        'lidar',
                        val_lidar_paths,
                        VAL_LIDAR_FILEPATH
                    ], [
                        'ground truth',
                        val_ground_truth_paths,
                        VAL_GROUND_TRUTH_FILEPATH
                    ], [
                        'intrinsics',
                        val_intrinsics_paths,
                        VAL_INTRINSICS_FILEPATH
                    ], [
                        'absolute pose',
                        val_absolute_pose_paths,
                        VAL_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'image klet',
                        val_camera_image_klet_paths,
                        VAL_IMAGE_KLET_FILEPATH
                    ], [
                        'lidar klet',
                        val_lidar_klet_paths,
                        VAL_LIDAR_KLET_FILEPATH
                    ], [
                        'ground truth klet',
                        val_ground_truth_klet_paths,
                        VAL_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'image subset',
                        val_camera_image_subset_paths,
                        VAL_IMAGE_SUBSET_FILEPATH
                    ], [
                        'lidar subset',
                        val_lidar_subset_paths,
                        VAL_LIDAR_SUBSET_FILEPATH
                    ], [
                        'ground truth subset',
                        val_ground_truth_subset_paths,
                        VAL_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'intrinsics subset',
                        val_intrinsics_subset_paths,
                        VAL_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'absolute pose subset',
                        val_absolute_pose_subset_paths,
                        VAL_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'image klet subset',
                        val_camera_image_klet_subset_paths,
                        VAL_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'lidar klet subset',
                        val_lidar_klet_subset_paths,
                        VAL_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'ground truth klet',
                        val_ground_truth_klet_subset_paths,
                        VAL_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ], [
                        'day image',
                        val_day_camera_image_paths,
                        VAL_DAY_IMAGE_FILEPATH
                    ], [
                        'day lidar',
                        val_day_lidar_paths,
                        VAL_DAY_LIDAR_FILEPATH
                    ], [
                        'day ground truth',
                        val_day_ground_truth_paths,
                        VAL_DAY_GROUND_TRUTH_FILEPATH
                    ], [
                        'day intrinsics',
                        val_day_intrinsics_paths,
                        VAL_DAY_INTRINSICS_FILEPATH
                    ], [
                        'day absolute pose',
                        val_day_absolute_pose_paths,
                        VAL_DAY_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'day image klet',
                        val_day_camera_image_klet_paths,
                        VAL_DAY_IMAGE_KLET_FILEPATH
                    ], [
                        'day lidar klet',
                        val_day_lidar_klet_paths,
                        VAL_DAY_LIDAR_KLET_FILEPATH
                    ], [
                        'day ground truth klet',
                        val_day_ground_truth_klet_paths,
                        VAL_DAY_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'day image subset',
                        val_day_camera_image_subset_paths,
                        VAL_DAY_IMAGE_SUBSET_FILEPATH
                    ], [
                        'day lidar subset',
                        val_day_lidar_subset_paths,
                        VAL_DAY_LIDAR_SUBSET_FILEPATH
                    ], [
                        'day ground truth subset',
                        val_day_ground_truth_subset_paths,
                        VAL_DAY_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'day intrinsics subset',
                        val_day_intrinsics_subset_paths,
                        VAL_DAY_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'day absolute pose subset',
                        val_day_absolute_pose_subset_paths,
                        VAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'day image klet subset',
                        val_day_camera_image_klet_subset_paths,
                        VAL_DAY_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'day lidar klet subset',
                        val_day_lidar_klet_subset_paths,
                        VAL_DAY_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'day ground truth klet subset',
                        val_day_ground_truth_klet_subset_paths,
                        VAL_DAY_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ], [
                        'night image',
                        val_night_camera_image_paths,
                        VAL_NIGHT_IMAGE_FILEPATH
                    ], [
                        'night lidar',
                        val_night_lidar_paths,
                        VAL_NIGHT_LIDAR_FILEPATH
                    ], [
                        'night ground truth',
                        val_night_ground_truth_paths,
                        VAL_NIGHT_GROUND_TRUTH_FILEPATH
                    ], [
                        'night intrinsics',
                        val_night_intrinsics_paths,
                        VAL_NIGHT_INTRINSICS_FILEPATH
                    ], [
                        'night absolute pose',
                        val_night_absolute_pose_paths,
                        VAL_NIGHT_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'night image klet',
                        val_night_camera_image_klet_paths,
                        VAL_NIGHT_IMAGE_KLET_FILEPATH
                    ], [
                        'night lidar klet',
                        val_night_lidar_klet_paths,
                        VAL_NIGHT_LIDAR_KLET_FILEPATH
                    ], [
                        'night ground truth klet',
                        val_night_ground_truth_klet_paths,
                        VAL_NIGHT_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'night image subset',
                        val_night_camera_image_subset_paths,
                        VAL_NIGHT_IMAGE_SUBSET_FILEPATH
                    ], [
                        'night lidar subset',
                        val_night_lidar_subset_paths,
                        VAL_NIGHT_LIDAR_SUBSET_FILEPATH
                    ], [
                        'night ground truth subset',
                        val_night_ground_truth_subset_paths,
                        VAL_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'night intrinsics subset',
                        val_night_intrinsics_subset_paths,
                        VAL_NIGHT_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'night absolute pose subset',
                        val_night_absolute_pose_subset_paths,
                        VAL_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'night image klet subset',
                        val_night_camera_image_klet_subset_paths,
                        VAL_NIGHT_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'night lidar klet subset',
                        val_night_lidar_klet_subset_paths,
                        VAL_NIGHT_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'night ground truth klet subset',
                        val_night_ground_truth_klet_subset_paths,
                        VAL_NIGHT_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ],
                ]
            ], [
                'testing',
                [
                    [
                        'image',
                        test_camera_image_paths,
                        TEST_IMAGE_FILEPATH
                    ], [
                        'lidar',
                        test_lidar_paths,
                        TEST_LIDAR_FILEPATH
                    ], [
                        'ground truth',
                        test_ground_truth_paths,
                        TEST_GROUND_TRUTH_FILEPATH
                    ], [
                        'intrinsics',
                        test_intrinsics_paths,
                        TEST_INTRINSICS_FILEPATH
                    ], [
                        'absolute pose',
                        test_absolute_pose_paths,
                        TEST_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'image klet',
                        test_camera_image_klet_paths,
                        TEST_IMAGE_KLET_FILEPATH
                    ], [
                        'lidar klet',
                        test_lidar_klet_paths,
                        TEST_LIDAR_KLET_FILEPATH
                    ], [
                        'ground truth klet',
                        test_ground_truth_klet_paths,
                        TEST_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'image subset',
                        test_camera_image_subset_paths,
                        TEST_IMAGE_SUBSET_FILEPATH
                    ], [
                        'lidar subset',
                        test_lidar_subset_paths,
                        TEST_LIDAR_SUBSET_FILEPATH
                    ], [
                        'ground truth subset',
                        test_ground_truth_subset_paths,
                        TEST_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'intrinsics subset',
                        test_intrinsics_subset_paths,
                        TEST_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'absolute pose subset',
                        test_absolute_pose_subset_paths,
                        TEST_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'image klet subset',
                        test_camera_image_klet_subset_paths,
                        TEST_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'lidar klet subset',
                        test_lidar_klet_subset_paths,
                        TEST_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'ground truth klet subset',
                        test_ground_truth_klet_subset_paths,
                        TEST_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ], [
                        'day image',
                        test_day_camera_image_paths,
                        TEST_DAY_IMAGE_FILEPATH
                    ], [
                        'day lidar',
                        test_day_lidar_paths,
                        TEST_DAY_LIDAR_FILEPATH
                    ], [
                        'day ground truth',
                        test_day_ground_truth_paths,
                        TEST_DAY_GROUND_TRUTH_FILEPATH
                    ], [
                        'day intrinsics',
                        test_day_intrinsics_paths,
                        TEST_DAY_INTRINSICS_FILEPATH
                    ], [
                        'day absolute pose',
                        test_day_absolute_pose_paths,
                        TEST_DAY_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'day image klet',
                        test_day_camera_image_klet_paths,
                        TEST_DAY_IMAGE_KLET_FILEPATH
                    ], [
                        'day lidar klet',
                        test_day_lidar_klet_paths,
                        TEST_DAY_LIDAR_KLET_FILEPATH
                    ], [
                        'day ground truth klet',
                        test_day_ground_truth_klet_paths,
                        TEST_DAY_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'day image subset',
                        test_day_camera_image_subset_paths,
                        TEST_DAY_IMAGE_SUBSET_FILEPATH
                    ], [
                        'day lidar subset',
                        test_day_lidar_subset_paths,
                        TEST_DAY_LIDAR_SUBSET_FILEPATH
                    ], [
                        'day ground truth subset',
                        test_day_ground_truth_subset_paths,
                        TEST_DAY_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'day intrinsics subset',
                        test_day_intrinsics_subset_paths,
                        TEST_DAY_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'day absolute pose subset',
                        test_day_absolute_pose_subset_paths,
                        TEST_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'day image klet subset',
                        test_day_camera_image_klet_subset_paths,
                        TEST_DAY_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'day lidar klet subset',
                        test_day_lidar_klet_subset_paths,
                        TEST_DAY_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'day ground truth klet subset',
                        test_day_ground_truth_klet_subset_paths,
                        TEST_DAY_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ], [
                        'night image',
                        test_night_camera_image_paths,
                        TEST_NIGHT_IMAGE_FILEPATH
                    ], [
                        'night lidar',
                        test_night_lidar_paths,
                        TEST_NIGHT_LIDAR_FILEPATH
                    ], [
                        'night ground truth',
                        test_night_ground_truth_paths,
                        TEST_NIGHT_GROUND_TRUTH_FILEPATH
                    ], [
                        'night intrinsics',
                        test_night_intrinsics_paths,
                        TEST_NIGHT_INTRINSICS_FILEPATH
                    ], [
                        'night absolute pose',
                        test_night_absolute_pose_paths,
                        TEST_NIGHT_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'night image klet',
                        test_night_camera_image_klet_paths,
                        TEST_NIGHT_IMAGE_KLET_FILEPATH
                    ], [
                        'night lidar klet',
                        test_night_lidar_klet_paths,
                        TEST_NIGHT_LIDAR_KLET_FILEPATH
                    ], [
                        'night ground truth klet',
                        test_night_ground_truth_klet_paths,
                        TEST_NIGHT_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'night image subset',
                        test_night_camera_image_subset_paths,
                        TEST_NIGHT_IMAGE_SUBSET_FILEPATH
                    ], [
                        'night lidar subset',
                        test_night_lidar_subset_paths,
                        TEST_NIGHT_LIDAR_SUBSET_FILEPATH
                    ], [
                        'night ground truth subset',
                        test_night_ground_truth_subset_paths,
                        TEST_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'night intrinsics subset',
                        test_night_intrinsics_subset_paths,
                        TEST_NIGHT_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'night absolute pose subset',
                        test_night_absolute_pose_subset_paths,
                        TEST_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'night image klet subset',
                        test_night_camera_image_klet_subset_paths,
                        TEST_NIGHT_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'night lidar klet subset',
                        test_night_lidar_klet_subset_paths,
                        TEST_NIGHT_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'night ground truth klet subset',
                        test_night_ground_truth_klet_subset_paths,
                        TEST_NIGHT_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ],
                ]
            ], [
                'testval',
                [
                    [
                        'image',
                        test_camera_image_paths + val_camera_image_paths,
                        TESTVAL_IMAGE_FILEPATH
                    ], [
                        'lidar',
                        test_lidar_paths + val_lidar_paths,
                        TESTVAL_LIDAR_FILEPATH
                    ], [
                        'ground truth',
                        test_ground_truth_paths + val_ground_truth_paths,
                        TESTVAL_GROUND_TRUTH_FILEPATH
                    ], [
                        'intrinsics',
                        test_intrinsics_paths + val_intrinsics_paths,
                        TESTVAL_INTRINSICS_FILEPATH
                    ], [
                        'absolute pose',
                        test_absolute_pose_paths + val_absolute_pose_paths,
                        TESTVAL_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'image klet',
                        test_camera_image_klet_paths + val_camera_image_klet_paths,
                        TESTVAL_IMAGE_KLET_FILEPATH
                    ], [
                        'lidar klet',
                        test_lidar_klet_paths + val_lidar_klet_paths,
                        TESTVAL_LIDAR_KLET_FILEPATH
                    ], [
                        'ground truth klet',
                        test_ground_truth_klet_paths + val_ground_truth_klet_paths,
                        TESTVAL_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'image subset',
                        test_camera_image_subset_paths + val_camera_image_subset_paths,
                        TESTVAL_IMAGE_SUBSET_FILEPATH
                    ], [
                        'lidar subset',
                        test_lidar_subset_paths + val_lidar_subset_paths,
                        TESTVAL_LIDAR_SUBSET_FILEPATH
                    ], [
                        'ground truth subset',
                        test_ground_truth_subset_paths + val_ground_truth_subset_paths,
                        TESTVAL_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'intrinsics subset',
                        test_intrinsics_subset_paths + val_intrinsics_subset_paths,
                        TESTVAL_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'absolute pose subset',
                        test_absolute_pose_subset_paths + val_absolute_pose_subset_paths,
                        TESTVAL_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'image klet subset',
                        test_camera_image_klet_subset_paths + val_camera_image_klet_subset_paths,
                        TESTVAL_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'lidar klet subset',
                        test_lidar_klet_subset_paths + val_lidar_klet_subset_paths,
                        TESTVAL_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'ground truth klet subset',
                        test_ground_truth_klet_subset_paths + val_ground_truth_klet_subset_paths,
                        TESTVAL_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ], [
                        'day image',
                        test_day_camera_image_paths + val_day_camera_image_paths,
                        TESTVAL_DAY_IMAGE_FILEPATH
                    ], [
                        'day lidar',
                        test_day_lidar_paths + val_day_lidar_paths,
                        TESTVAL_DAY_LIDAR_FILEPATH
                    ], [
                        'day ground truth',
                        test_day_ground_truth_paths + val_day_ground_truth_paths,
                        TESTVAL_DAY_GROUND_TRUTH_FILEPATH
                    ], [
                        'day intrinsics',
                        test_day_intrinsics_paths + val_day_intrinsics_paths,
                        TESTVAL_DAY_INTRINSICS_FILEPATH
                    ], [
                        'day absolute pose',
                        test_day_absolute_pose_paths + val_day_absolute_pose_paths,
                        TESTVAL_DAY_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'day image klet',
                        test_day_camera_image_klet_paths + val_day_camera_image_klet_paths,
                        TESTVAL_DAY_IMAGE_KLET_FILEPATH
                    ], [
                        'day lidar klet',
                        test_day_lidar_klet_paths + val_day_lidar_klet_paths,
                        TESTVAL_DAY_LIDAR_KLET_FILEPATH
                    ], [
                        'day ground truth klet',
                        test_day_ground_truth_klet_paths + val_day_ground_truth_klet_paths,
                        TESTVAL_DAY_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'day image subset',
                        test_day_camera_image_subset_paths + val_day_camera_image_subset_paths,
                        TESTVAL_DAY_IMAGE_SUBSET_FILEPATH
                    ], [
                        'day lidar subset',
                        test_day_lidar_subset_paths + val_day_lidar_subset_paths,
                        TESTVAL_DAY_LIDAR_SUBSET_FILEPATH
                    ], [
                        'day ground truth subset',
                        test_day_ground_truth_subset_paths + val_day_ground_truth_subset_paths,
                        TESTVAL_DAY_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'day intrinsics subset',
                        test_day_intrinsics_subset_paths + val_day_intrinsics_subset_paths,
                        TESTVAL_DAY_INTRINSICS_SUBSET_FILEPATH
                    ], [
                        'day absolute pose subset',
                        test_day_absolute_pose_subset_paths + val_day_absolute_pose_subset_paths,
                        TESTVAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'day image klet subset',
                        test_day_camera_image_klet_subset_paths + val_day_camera_image_klet_subset_paths,
                        TESTVAL_DAY_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'day lidar klet subset',
                        test_day_lidar_klet_subset_paths + val_day_lidar_klet_subset_paths,
                        TESTVAL_DAY_LIDAR_KLET_SUBSET_FILEPATH
                    ], [
                        'day ground truth klet subset',
                        test_day_ground_truth_klet_subset_paths + val_day_ground_truth_klet_subset_paths,
                        TESTVAL_DAY_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ], [
                        'night image',
                        test_night_camera_image_paths + val_night_camera_image_paths,
                        TESTVAL_NIGHT_IMAGE_FILEPATH
                    ], [
                        'night lidar',
                        test_night_lidar_paths + val_night_lidar_paths,
                        TESTVAL_NIGHT_LIDAR_FILEPATH
                    ], [
                        'night ground truth',
                        test_night_ground_truth_paths + val_night_ground_truth_paths,
                        TESTVAL_NIGHT_GROUND_TRUTH_FILEPATH
                    ], [
                        'night intrinsics',
                        test_night_intrinsics_paths + val_night_intrinsics_paths,
                        TESTVAL_NIGHT_INTRINSICS_FILEPATH
                    ], [
                        'night absolute pose',
                        test_night_absolute_pose_paths + val_night_absolute_pose_paths,
                        TESTVAL_NIGHT_ABSOLUTE_POSE_FILEPATH
                    ], [
                        'night image klet',
                        test_night_camera_image_klet_paths + val_night_camera_image_klet_paths,
                        TESTVAL_NIGHT_IMAGE_KLET_FILEPATH
                    ], [
                        'night lidar klet',
                        test_night_lidar_klet_paths + val_night_lidar_klet_paths,
                        TESTVAL_NIGHT_LIDAR_KLET_FILEPATH
                    ], [
                        'night ground truth klet',
                        test_night_ground_truth_klet_paths + val_night_ground_truth_klet_paths,
                        TESTVAL_NIGHT_GROUND_TRUTH_KLET_FILEPATH
                    ], [
                        'night image subset',
                        test_night_camera_image_subset_paths + val_night_camera_image_subset_paths,
                        TESTVAL_NIGHT_IMAGE_SUBSET_FILEPATH
                    ], [
                        'night lidar subset',
                        test_night_lidar_subset_paths + val_night_lidar_subset_paths,
                        TESTVAL_NIGHT_LIDAR_SUBSET_FILEPATH
                    ], [
                        'night ground truth subset',
                        test_night_ground_truth_subset_paths + val_night_ground_truth_subset_paths,
                        TESTVAL_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH
                    ], [
                        'night intrinsics subset',
                        test_night_intrinsics_subset_paths + val_night_intrinsics_subset_paths,
                        TESTVAL_NIGHT_LIDAR_SUBSET_FILEPATH
                    ], [
                        'night absolute pose subset',
                        test_night_absolute_pose_subset_paths + val_night_absolute_pose_subset_paths,
                        TESTVAL_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH
                    ], [
                        'night image klet subset',
                        test_night_camera_image_klet_subset_paths + val_night_camera_image_klet_subset_paths,
                        TESTVAL_NIGHT_IMAGE_KLET_SUBSET_FILEPATH
                    ], [
                        'night lidar klet subset',
                        test_night_lidar_klet_subset_paths + val_night_lidar_klet_subset_paths,
                        TESTVAL_NIGHT_LIDAR_KLET_FILEPATH
                    ], [
                        'night ground truth klet subset',
                        test_night_ground_truth_klet_subset_paths + val_night_ground_truth_klet_subset_paths,
                        TESTVAL_NIGHT_GROUND_TRUTH_KLET_SUBSET_FILEPATH
                    ],
                ]
            ]
        ]

        # Create output directories
        for dirpath in [TRAIN_REF_DIRPATH, VAL_REF_DIRPATH, TEST_REF_DIRPATH, TESTVAL_REF_DIRPATH]:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        for output_info in outputs:

            tag, output = output_info
            for output_type, paths, filepath in output:

                print('Storing {} {} {} file paths into: {}'.format(
                    len(paths), tag, output_type, filepath))
                data_utils.write_paths(filepath, paths)





################################################################################
# Old notes: the for loop pseudo-code
# Access each directory of images. (Already has train val test split)
# Grab each segment/sample then its frames. Sort them.
# Grab each frame
# Tag sorting
# input into process_scene
# Forward and backward frames setup same as Vadim's with some mod
# Result unpacking later, pretty much the same.
# .
# Split as in "train", "val", "test"
# Focus aligning to nuscenes