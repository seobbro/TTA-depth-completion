from nuscenes.nuscenes import NuScenes, NuScenesExplorer
import os, sys, copy, argparse, json, shutil
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import torch.multiprocessing as mp
import torch
sys.path.insert(0, 'src')
sys.path.append(os.getcwd())
import src.data_utils as data_utils
import src.net_utils as net_utils

MAX_TRAIN_SCENES = 850
MAX_TEST_SCENES = 150


"""
python3 -W ignore setup/setup_dataset_nuscenes.py --nuscenes_data_root_dirpath /media/staging/common/datasets/nuscenes --nuscenes_data_derived_dirpath /media/staging/common/datasets/nuscenes_derived --n_thread 24
python3 setup/setup_dataset_nuscenes.py --nuscenes_data_root_dirpath data/nuscenes --nuscenes_data_derived_dirpath /media/staging/cchandrappa/nuscenes_derived --n_thread 1
python3 -W ignore setup/setup_dataset_nuscenes.py --nuscenes_data_root_dirpath /media/staging/common/datasets/nuscenes --nuscenes_data_derived_dirpath /media/staging/common/datasets/nuscenes_derived_v2 --n_thread 24

"""

'''
Requires running setup/setup_dataset_nuscenes_detections.py before executing script below
'''

'''
Data split filepaths
'''
DATA_SPLIT_DIRPATH = os.path.join('setup', 'nuscenes')
TRAIN_DATA_SPLIT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, 'train_scene_ids.txt')
VAL_DATA_SPLIT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, 'val_scene_ids.txt')
TEST_DATA_SPLIT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, 'test_scene_ids.txt')


'''
Output filepaths
'''
TRAIN_REF_DIRPATH = os.path.join('training', 'nuscenes')
VAL_REF_DIRPATH = os.path.join('validation', 'nuscenes')
TEST_REF_DIRPATH = os.path.join('testing', 'nuscenes')
TESTVAL_REF_DIRPATH = os.path.join('testval', 'nuscenes')

# Training set for all, day, and night splits
TRAIN_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_image.txt')
TRAIN_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_lidar.txt')
TRAIN_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_ground_truth.txt')
TRAIN_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_intrinsics.txt')
TRAIN_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_absolute_pose.txt')

TRAIN_DAY_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_day_image.txt')
TRAIN_DAY_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_day_lidar.txt')
TRAIN_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_day_ground_truth.txt')
TRAIN_DAY_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_day_intrinsics.txt')
TRAIN_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_day_absolute_pose.txt')

TRAIN_NIGHT_IMAGE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_night_image.txt')
TRAIN_NIGHT_LIDAR_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_night_lidar.txt')
TRAIN_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_night_ground_truth.txt')
TRAIN_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_night_intrinsics.txt')
TRAIN_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TRAIN_REF_DIRPATH, 'nuscenes_train_night_absolute_pose.txt')

# Validation set for all, day, and night splits
VAL_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_image.txt')
VAL_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_lidar.txt')
VAL_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth.txt')
VAL_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_intrinsics.txt')
VAL_ABSOLUTE_POSE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_absolute_pose.txt')

VAL_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_image-subset.txt')
VAL_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_lidar-subset.txt')
VAL_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_ground_truth-subset.txt')
VAL_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_intrinsics-subset.txt')
VAL_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_absolute_pose-subset.txt')

VAL_DAY_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_image.txt')
VAL_DAY_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_lidar.txt')
VAL_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_ground_truth.txt')
VAL_DAY_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_intrinsics.txt')
VAL_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_absolute_pose.txt')

VAL_DAY_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_image-subset.txt')
VAL_DAY_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_lidar-subset.txt')
VAL_DAY_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_ground_truth-subset.txt')
VAL_DAY_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_intrinsics-subset.txt')
VAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_day_absolute_pose-subset.txt')

VAL_NIGHT_IMAGE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_image.txt')
VAL_NIGHT_LIDAR_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_lidar.txt')
VAL_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_ground_truth.txt')
VAL_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_intrinsics.txt')
VAL_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_absolute_pose.txt')

VAL_NIGHT_IMAGE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_image-subset.txt')
VAL_NIGHT_LIDAR_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_lidar-subset.txt')
VAL_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_ground_truth-subset.txt')
VAL_NIGHT_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_intrinsics-subset.txt')
VAL_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    VAL_REF_DIRPATH, 'nuscenes_val_night_absolute_pose-subset.txt')

# Testing set for all, day, and night splits
TEST_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_image.txt')
TEST_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_lidar.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_ground_truth.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_intrinsics.txt')
TEST_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_absolute_pose.txt')

TEST_IMAGE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_image-subset.txt')
TEST_LIDAR_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_lidar-subset.txt')
TEST_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_ground_truth-subset.txt')
TEST_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_intrinsics-subset.txt')
TEST_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_absolute_pose-subset.txt')

TEST_DAY_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_image.txt')
TEST_DAY_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_lidar.txt')
TEST_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_ground_truth.txt')
TEST_DAY_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_intrinsics.txt')
TEST_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_absolute_pose.txt')

TEST_DAY_IMAGE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_image-subset.txt')
TEST_DAY_LIDAR_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_lidar-subset.txt')
TEST_DAY_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_ground_truth-subset.txt')
TEST_DAY_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_intrinsics-subset.txt')
TEST_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_day_absolute_pose-subset.txt')

TEST_NIGHT_IMAGE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_image.txt')
TEST_NIGHT_LIDAR_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_lidar.txt')
TEST_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_ground_truth.txt')
TEST_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_intrinsics.txt')
TEST_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_absolute_pose.txt')

TEST_NIGHT_IMAGE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_image-subset.txt')
TEST_NIGHT_LIDAR_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_lidar-subset.txt')
TEST_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_ground_truth-subset.txt')
TEST_NIGHT_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_intrinsics-subset.txt')
TEST_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TEST_REF_DIRPATH, 'nuscenes_test_night_absolute_pose-subset.txt')

# Testing + validation set to include larger amounts of night-time samples
TESTVAL_IMAGE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_image.txt')
TESTVAL_LIDAR_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_lidar.txt')
TESTVAL_GROUND_TRUTH_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_ground_truth.txt')
TESTVAL_INTRINSICS_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_intrinsics.txt')
TESTVAL_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_absolute_pose.txt')

TESTVAL_IMAGE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_image-subset.txt')
TESTVAL_LIDAR_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_lidar-subset.txt')
TESTVAL_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_ground_truth-subset.txt')
TESTVAL_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_intrinsics-subset.txt')
TESTVAL_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_absolute_pose-subset.txt')

TESTVAL_DAY_IMAGE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_image.txt')
TESTVAL_DAY_LIDAR_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_lidar.txt')
TESTVAL_DAY_GROUND_TRUTH_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_ground_truth.txt')
TESTVAL_DAY_INTRINSICS_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_intrinsics.txt')
TESTVAL_DAY_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_absolute_pose.txt')

TESTVAL_DAY_IMAGE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_image-subset.txt')
TESTVAL_DAY_LIDAR_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_lidar-subset.txt')
TESTVAL_DAY_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_ground_truth-subset.txt')
TESTVAL_DAY_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_intrinsics-subset.txt')
TESTVAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_day_absolute_pose-subset.txt')

TESTVAL_NIGHT_IMAGE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_image.txt')
TESTVAL_NIGHT_LIDAR_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_lidar.txt')
TESTVAL_NIGHT_GROUND_TRUTH_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_ground_truth.txt')
TESTVAL_NIGHT_INTRINSICS_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_intrinsics.txt')
TESTVAL_NIGHT_ABSOLUTE_POSE_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_absolute_pose.txt')

TESTVAL_NIGHT_IMAGE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_image-subset.txt')
TESTVAL_NIGHT_LIDAR_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_lidar-subset.txt')
TESTVAL_NIGHT_GROUND_TRUTH_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_ground_truth-subset.txt')
TESTVAL_NIGHT_INTRINSICS_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_intrinsics-subset.txt')
TESTVAL_NIGHT_ABSOLUTE_POSE_SUBSET_FILEPATH = os.path.join(
    TESTVAL_REF_DIRPATH, 'nuscenes_testval_night_absolute_pose-subset.txt')


'''
Set up input arguments
'''
parser = argparse.ArgumentParser()

parser.add_argument('--nuscenes_data_root_dirpath',
    type=str, required=True, help='Path to nuscenes dataset')
parser.add_argument('--nuscenes_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--n_forward_frames_to_reproject',
    type=int, default=12, help='Number of forward frames to project onto a target frame')
parser.add_argument('--n_backward_frames_to_reproject',
    type=int, default=12, help='Number of backward frames to project onto a target frame')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=40, help='Number of threads to use in parallel pool')
parser.add_argument('--debug',
    action='store_true', help='If set, then enter debug mode')
parser.add_argument('--filter_threshold_photometric_reconstruction',
    type=float, default=-1, help='If set to greater than 0 then perform photometric reconstruction filtering')

args = parser.parse_args()


# Create global nuScene object
nusc_trainval = NuScenes(
    version='v1.0-trainval',
    dataroot=args.nuscenes_data_root_dirpath,
    verbose=True)

nusc_explorer_trainval = NuScenesExplorer(nusc_trainval)

nusc_test = NuScenes(
    version='v1.0-test',
    dataroot=args.nuscenes_data_root_dirpath,
    verbose=True)

nusc_explorer_test = NuScenesExplorer(nusc_test)

# Create outlier removal object shared by all threads
outlier_removal = net_utils.OutlierRemoval(kernel_size=7, threshold=1.5)


def get_train_val_test_split_scene_ids(train_data_split_path,
                                       val_data_split_path,
                                       test_data_split_path):
    '''
    Given the nuscenes object, find out which scene ids correspond to which set.
    The split is taken from the official nuScene split available here:
    https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py

    Arg(s):
        train_data_split_path : str
            path to file containing data split for training
        val_data_split_path : str
            path to file containing data split for validation
        test_data_split_path : str
            path to file containing data split for testing
    Returns:
        list[int] : list containing ids of the scenes that are in training split
        list[int] : list containing ids of the scenes that are in validation split
        list[int] : list containing ids of the scenes that are in testing split
    '''

    train_scene_ids = data_utils.read_paths(train_data_split_path)
    val_scene_ids = data_utils.read_paths(val_data_split_path)
    test_scene_ids = data_utils.read_paths(test_data_split_path)

    return train_scene_ids, val_scene_ids, test_scene_ids

def get_train_val_test_daynight_split_scene_ids():
    '''
    Get night time and day time splits for training, validation and testing

    Returns:
        list[str] : day time scene ids
        list[str] : night time scene ids
    '''
    day_ids = []
    night_ids = []

    for split in ['train', 'val', 'test']:
        day_scene_ids_path = os.path.join(DATA_SPLIT_DIRPATH, '{}_day_scene_ids.txt'.format(split))
        night_scene_ids_path = os.path.join(DATA_SPLIT_DIRPATH, '{}_night_scene_ids.txt'.format(split))

        if os.path.exists(day_scene_ids_path) and os.path.exists(night_scene_ids_path):
            with open(day_scene_ids_path, 'r') as f:
                for line in f.readlines():
                    day_ids.append(line.strip())

            with open(night_scene_ids_path, 'r') as f:
                for line in f.readlines():
                    night_ids.append(line.strip())

        else:
            data_split_path = os.path.join(DATA_SPLIT_DIRPATH, '{}_scene_ids.txt'.format(split))
            scene_ids = []
            with open(data_split_path, 'r') as f:
                for line in f.readlines():
                    scene_ids.append(line.strip())

            json_path = os.path.join(args.nuscenes_data_root_dirpath, f'v1.0-{"test" if split == "test" else "trainval"}/scene.json')
            with open(json_path, 'r') as f:
                data = json.load(f)

            for item in data:
                description = item['description'].lower()
                if item['name'] not in scene_ids:
                    continue
                if ('dark' in description or 'night' in description) and item['name'] in scene_ids:
                    night_ids.append(item['name'])
                else:
                    day_ids.append(item['name'])

            with open(day_scene_ids_path, 'w+') as f:
                for id in day_ids:
                    f.writelines(str(id) + '\n')

            with open(night_scene_ids_path, 'w+') as f:
                for id in night_ids:
                    f.writelines(str(id) + '\n')

        print(f'[{split}] day: {len(day_ids)}, night: {len(night_ids)}')

    return day_ids, night_ids

def point_cloud_to_image(nusc,
                         point_cloud,
                         lidar_sensor_token,
                         camera_token,
                         min_distance_from_camera=1.0):
    '''
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        nusc : Object
            nuScenes data object
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
        minimum_distance_from_camera : float32
            threshold for removing points that exceeds minimum distance from camera
    Returns:
        numpy[float32] : 3 x N array of x, y, z
        numpy[float32] : N array of z
        numpy[float32] : camera image
    '''

    # Get dictionary of containing path to image, pose, etc.
    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    image_path = os.path.join(nusc.dataroot, camera['filename'])
    image = data_utils.load_image(image_path)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_lidar_to_body['translation']))

    # Second step: transform from ego to the global frame.
    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pose_body_to_camera = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_body_to_camera['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_camera['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depth = point_cloud.points[2, :]

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Points will be 3 x N
    points = view_points(point_cloud.points[:3, :], np.array(pose_body_to_camera['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depth.shape[0], dtype=bool)
    mask = np.logical_and(mask, depth > min_distance_from_camera)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < image.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < image.shape[0] - 1)

    # Select points that are more than min distance from camera and not on edge of image
    points = points[:, mask]
    depth = depth[mask]

    return points, depth, image

def camera_to_lidar_frame(nusc,
                          point_cloud,
                          lidar_sensor_token,
                          camera_token):
    '''
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Arg(s):
        nusc : Object
            nuScenes data object
        point_cloud : PointCloud
            nuScenes point cloud object
        lidar_sensor_token : str
            token to access lidar data in nuscenes sample_data object
        camera_token : str
            token to access camera data in nuscenes sample_data object
    Returns:
        PointCloud : nuScenes point cloud object
    '''

    camera = nusc.get('sample_data', camera_token)
    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

    # First step: transform from camera into ego.
    pose_camera_to_body = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    point_cloud.rotate(Quaternion(pose_camera_to_body['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_camera_to_body['translation']))

    # Second step: transform from ego vehicle frame to global frame for the timestamp of the image.
    pose_body_to_global = nusc.get('ego_pose', camera['ego_pose_token'])
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix)
    point_cloud.translate(np.array(pose_body_to_global['translation']))

    # Third step: transform from global frame to ego frame
    pose_body_to_global = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
    point_cloud.translate(-np.array(pose_body_to_global['translation']))
    point_cloud.rotate(Quaternion(pose_body_to_global['rotation']).rotation_matrix.T)

    # Fourth step: transform point cloud from body to lidar
    pose_lidar_to_body = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
    point_cloud.translate(-np.array(pose_lidar_to_body['translation']))
    point_cloud.rotate(Quaternion(pose_lidar_to_body['rotation']).rotation_matrix.T)

    return point_cloud

def get_extrinsics_matrix(nusc, sensor_token):
    '''
    Get extrinsics (in world coordinate frame) of calibrated sensor

    Arg(s):
        nusc : Object
            nuScenes data object
        sensor_token : str
            token to access sensor data in nuscenes sample_data object
    Returns:
        torch.Tensor[float32] : 4 x 4 extrinsics matrix
    '''

    sensor = nusc.get('sample_data', sensor_token)

    # Get sensor to body transformation
    pose_sensor_to_body = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])

    pose_sensor_to_body_rotation = \
        np.array(Quaternion(pose_sensor_to_body['rotation']).rotation_matrix, dtype=np.float32)
    pose_sensor_to_body_translation = \
        np.array(pose_sensor_to_body['translation'], dtype=np.float32)

    # Get body to world transformation
    pose_body_to_global = nusc.get('ego_pose', sensor['ego_pose_token'])

    pose_body_to_global_rotation = \
        np.array(Quaternion(pose_body_to_global['rotation']).rotation_matrix, dtype=np.float32)
    pose_body_to_global_translation = \
        np.array(pose_body_to_global['translation'], dtype=np.float32)

    # Apply translation component from sensor to world
    pose_sensor_to_global_translation = \
        np.matmul(pose_body_to_global_rotation, pose_sensor_to_body_translation) + \
        pose_body_to_global_translation

    pose_sensor_to_global_translation = np.expand_dims(pose_sensor_to_global_translation, -1)

    # Apply rotation component from sensor to world and compose rotation and translation
    pose_sensor_to_global_rotation = \
        np.matmul(pose_body_to_global_rotation, pose_sensor_to_body_rotation)
    pose_sensor_to_global_matrix_3x4 = np.concatenate([
        pose_sensor_to_global_rotation,
        pose_sensor_to_global_translation],
        axis=-1)

    # Append homogeneous coordinates to make 4 x 4
    pose_sensor_to_global_matrix_4x4 = np.concatenate([
        pose_sensor_to_global_matrix_3x4,
        np.array([[0, 0, 0, 1]], dtype=np.float32)],
        axis=0)

    return pose_sensor_to_global_matrix_4x4.astype(np.float32)

def get_relative_pose(nusc, source_sensor_token, target_sensor_token):
    '''
    Get relative camera pose between source and target sensors

    Arg(s):
        nusc : Object
            nuScenes data object
        source_sensor_token : str
            token to access source sensor data in nuscenes sample_data object
        target_sensor_token : str
            token to access target sensor data in nuscenes sample_data object
    Returns:
        torch.Tensor[float32] : 4 x 4 extrinsics matrix
    '''

    # Get extrinsics matrix from source sensor to global frame
    source_extrinsics_matrix = get_extrinsics_matrix(nusc, source_sensor_token)
    source_sensor_to_global_rotation = source_extrinsics_matrix[:3, :3]
    source_sensor_to_global_translation = source_extrinsics_matrix[:3, -1]

    # Get extrinsics matrix from target sensor to global frame
    target_extrinsics_matrix = get_extrinsics_matrix(nusc, target_sensor_token)
    target_sensor_to_global_rotation = target_extrinsics_matrix[:3, :3]
    target_sensor_to_global_translation = target_extrinsics_matrix[:3, -1]

    # Get relative pose between the two sensors
    relative_rotation = np.matmul(
        np.linalg.inv(target_sensor_to_global_rotation),
        source_sensor_to_global_rotation)
    relative_translation = np.matmul(
        np.linalg.inv(target_sensor_to_global_rotation),
        (source_sensor_to_global_translation - target_sensor_to_global_translation))

    # Create 4 x 4 pose matrix
    relative_pose_matrix = np.zeros([4, 4])
    relative_pose_matrix[:3, :3] = relative_rotation
    relative_pose_matrix[:3, -1] = relative_translation
    relative_pose_matrix[-1, -1] = 1.0

    return relative_pose_matrix.astype(np.float32)

def merge_lidar_point_clouds(nusc,
                             nusc_explorer,
                             current_sample_token,
                             n_forward,
                             n_backward,
                             detections):
    '''
    Merges Lidar point from multiple samples and adds them to a single depth image
    Picks current_sample_token as reference and projects lidar points from all other frames into current_sample.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
        n_forward : int
            number of frames to merge in the forward direction.
        n_backward : int
            number of frames to merge in the backward direction
        detections : dict[str, list[int]]
            dictionary of scene tokens to list of bounding boxes in image frame
    Returns:
        numpy[float32] : 2 x N of x, y for lidar points projected into the image
        numpy[float32] : N depths of lidar points

    '''

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get lidar token in the current sample
    main_lidar_token = current_sample['data']['LIDAR_TOP']

    # Get the camera token for the current sample
    main_camera_token = current_sample['data']['CAM_FRONT']

    # Project the lidar frame into the camera frame
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=main_lidar_token,
        camera_token=main_camera_token)

    # Convert nuScenes format to numpy for image
    main_image = np.asarray(main_image)

    # Create an empty lidar image
    main_lidar_image = np.zeros((main_image.shape[0], main_image.shape[1]))

    # Get all bounding boxes for the lidar data in the current sample
    _, main_boxes, main_camera_intrinsic = nusc.get_sample_data(
        main_camera_token,
        box_vis_level=BoxVisibility.ANY,
        use_flat_vehicle_coordinates=False)

    main_points_lidar_quantized = np.round(main_points_lidar).astype(int)

    # Iterating through each lidar point and plotting them onto the lidar image
    for point_idx in range(0, main_points_lidar_quantized.shape[1]):
        # Get x and y index in image frame
        x = main_points_lidar_quantized[0, point_idx]
        y = main_points_lidar_quantized[1, point_idx]

        # Value of y, x is the depth
        main_lidar_image[y, x] = main_depth_lidar[point_idx]

    # Create a validity map to check which elements of the lidar image are valid
    main_validity_map = np.where(main_lidar_image > 0, 1, 0)

    # Count forward and backward frames
    n_forward_processed = 0
    n_backward_processed = 0

    # Initialize next sample as current sample
    next_sample = copy.deepcopy(current_sample)

    while next_sample['next'] != "" and n_forward_processed < n_forward:

        '''
        1. Load point cloud and image in `next' non-keyframe
        2. Use any trained model to get the object bbox from non-keyframe image
        3. Remove points within bbox, backproject to point cloud
        4. Project points onto image plane in current camera frame
        5. Load point cloud and image in `next' keyframe
        6. Project onto image to remove vehicle bounding boxes
        7. Backproject to point cloud and project to camera frame to yield lidar image
        '''

        # Get the token and sample data for the next non-keyframe sample and move forward until keyframe
        next_sample_nonkf = copy.deepcopy(next_sample)
        next_camera_sample_nonkf = nusc.get('sample_data', next_sample_nonkf['data']['CAM_FRONT'])
        next_lidar_sample_nonkf = nusc.get('sample_data', next_sample_nonkf['data']['LIDAR_TOP'])

        # Get the token and sample data for the next keyframe sample amd move forward
        next_sample_token = next_sample['next']
        next_sample = nusc.get('sample', next_sample_token)

        '''
        Process non-keyframe
        '''
        while next_camera_sample_nonkf['timestamp'] <= next_sample['timestamp']:

            # Get lidar and camera sensor token
            next_lidar_nonkf_token = next_lidar_sample_nonkf['token']
            next_camera_nonkf_token = next_camera_sample_nonkf['token']

            # Get intrinsics matrix
            next_camera_intrinsics_nonkf = \
                nusc.get('calibrated_sensor', next_camera_sample_nonkf['calibrated_sensor_token'])

            # Load non-keyframe image
            next_camera_sample_nonkf_path = \
                os.path.join(nusc.dataroot, next_camera_sample_nonkf['filename'])
            next_camera_image_nonkf = data_utils.load_image(next_camera_sample_nonkf_path)

            # Get height and width of non-keyframe image
            height_nonkf, width_nonkf = next_camera_image_nonkf.shape[0:2]

            next_points_lidar_nonkf, next_depth_lidar_nonkf, _ = nusc_explorer.map_pointcloud_to_image(
                pointsensor_token=next_lidar_nonkf_token,
                camera_token=next_camera_nonkf_token)

            # Create image plane for lidar points
            next_lidar_image_nonkf = np.zeros([height_nonkf, width_nonkf])
            next_points_lidar_nonkf_quantized = np.round(next_points_lidar_nonkf).astype(int)

            # Plot non-keyframe lidar points onto the image plane
            for idx in range(0, next_points_lidar_nonkf_quantized.shape[-1]):
                x, y = next_points_lidar_nonkf_quantized[0:2, idx]
                next_lidar_image_nonkf[y, x] = next_depth_lidar_nonkf[idx]

            # Remove points which belong to moving objects using semantic segmentation from detr model
            next_boxes_nonkf = detections[next_camera_nonkf_token]

            for box in next_boxes_nonkf:
                (top_x , top_y, bottom_x , bottom_y) = box
                min_x = int(max(top_x, 0))
                min_y = int(max(top_y, 0))
                max_x = int(min(bottom_x, width_nonkf))
                max_y = int(min(bottom_y, height_nonkf))

                # Filter out the points inside the bounding boxes
                next_lidar_image_nonkf[min_y:max_y, min_x:max_x] = 0

            # Now we need to convert image format to point cloud array format (y, x, z)
            next_lidar_points_nonkf_y, next_lidar_points_nonkf_x = np.nonzero(next_lidar_image_nonkf)
            next_lidar_points_nonkf_z = next_lidar_image_nonkf[next_lidar_points_nonkf_y, next_lidar_points_nonkf_x]

            # Backproject to camera frame as 3 x N
            x_y_homogeneous_nonkf  = np.stack([
                next_lidar_points_nonkf_x,
                next_lidar_points_nonkf_y,
                np.ones_like(next_lidar_points_nonkf_x)],
                axis=0)

            camera_intrinsics_nonkf = np.array(next_camera_intrinsics_nonkf['camera_intrinsic'])
            x_y_lifted_nonkf = np.matmul(np.linalg.inv(camera_intrinsics_nonkf), x_y_homogeneous_nonkf)
            x_y_z_nonkf = x_y_lifted_nonkf * np.expand_dims(next_lidar_points_nonkf_z, axis=0)

            # To convert the lidar point cloud into a LidarPointCloud object, we need 4, N shape.
            # So we add a 4th fake intensity vector
            fake_intensity_array_nonkf = np.ones(x_y_z_nonkf.shape[1])
            fake_intensity_array_nonkf = np.expand_dims(fake_intensity_array_nonkf, axis=0)
            x_y_z_nonkf = np.concatenate((x_y_z_nonkf, fake_intensity_array_nonkf), axis=0)

            # Convert lidar point cloud into a nuScene LidarPointCloud object
            next_point_cloud_nonkf = LidarPointCloud(x_y_z_nonkf)

            # Now we can transform the points back to the lidar frame of reference
            next_point_cloud_nonkf = camera_to_lidar_frame(
                nusc=nusc,
                point_cloud=next_point_cloud_nonkf,
                lidar_sensor_token=next_lidar_nonkf_token,
                camera_token=next_camera_nonkf_token)

            # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
            next_points_lidar_main_nonkf, next_depth_lidar_main_nonkf, _ = point_cloud_to_image(
                nusc=nusc,
                point_cloud=next_point_cloud_nonkf,
                lidar_sensor_token=next_lidar_nonkf_token,
                camera_token=main_camera_token,
                min_distance_from_camera=1.0)

            # We need to do another step of filtering to filter out all the points that will be projected upon moving objects in the main frame
            next_lidar_image_main_nonkf = np.zeros_like(main_lidar_image)

            # Plots depth values onto the image
            next_points_lidar_main_quantized_nonkf = np.round(next_points_lidar_main_nonkf).astype(int)

            for idx in range(0, next_points_lidar_main_quantized_nonkf.shape[-1]):
                x, y = next_points_lidar_main_quantized_nonkf[0:2, idx]
                next_lidar_image_main_nonkf[y, x] = next_depth_lidar_main_nonkf[idx]

            for box in main_boxes:
                if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                    corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                    min_x_main = int(np.min(corners.T[:, 0]))
                    min_y_main = int(np.min(corners.T[:, 1]))
                    max_x_main = int(np.max(corners.T[:, 0]))
                    max_y_main = int(np.max(corners.T[:, 1]))

                    # Filter out the points inside the bounding box
                    next_lidar_image_main_nonkf[min_y_main:max_y_main, min_x_main:max_x_main] = 0

            # Convert image format to point cloud format
            next_lidar_points_main_y_nonkf, next_lidar_points_main_x_nonkf = np.nonzero(next_lidar_image_main_nonkf)
            next_lidar_points_main_z_nonkf = \
                next_lidar_image_main_nonkf[next_lidar_points_main_y_nonkf, next_lidar_points_main_x_nonkf]

            # Stack y and x to 2 x N (x, y)
            next_points_lidar_main_nonkf = np.stack([
                next_lidar_points_main_x_nonkf,
                next_lidar_points_main_y_nonkf],
                axis=0)
            next_depth_lidar_main = next_lidar_points_main_z_nonkf

            next_points_lidar_main_quantized_nonkf = np.round(next_points_lidar_main_nonkf).astype(int)

            for point_idx in range(0, next_points_lidar_main_quantized_nonkf.shape[1]):
                x = next_points_lidar_main_quantized_nonkf[0, point_idx]
                y = next_points_lidar_main_quantized_nonkf[1, point_idx]

                is_not_occluded = \
                    main_validity_map[y, x] == 1 and \
                    next_depth_lidar_main[point_idx] < main_lidar_image[y, x]

                if is_not_occluded:
                    main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
                elif main_validity_map[y, x] != 1:
                    main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
                    main_validity_map[y, x] = 1

            # Move camera token to the next
            if next_camera_sample_nonkf['next'] != "" and next_lidar_sample_nonkf['next'] != "":
                next_camera_sample_nonkf = nusc.get('sample_data', next_camera_sample_nonkf['next'])
            else:
                break

            # Lidar has higher frequency than camera, find the closest lidar token
            while next_lidar_sample_nonkf['timestamp'] < next_camera_sample_nonkf['timestamp'] and next_lidar_sample_nonkf['next'] != "":
                next_lidar_sample_nonkf = nusc.get('sample_data', next_lidar_sample_nonkf['next'])

        '''
        Process keyframe
        '''
        # Get lidar and camera token in the current sample
        next_lidar_token = next_sample['data']['LIDAR_TOP']
        next_camera_token = next_sample['data']['CAM_FRONT']

        # Get bounding box in image frame to remove vehicles from point cloud
        _, next_boxes, next_camera_intrinsics = nusc.get_sample_data(
            next_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        # Map next frame point cloud to image so we can remove vehicle based on bounding boxes
        next_points_lidar, next_depth_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=next_lidar_token,
            camera_token=next_camera_token)

        next_lidar_image = np.zeros_like(main_lidar_image)

        next_points_lidar_quantized = np.round(next_points_lidar).astype(int)
        # Plots depth values onto the image
        for idx in range(0, next_points_lidar_quantized.shape[-1]):
            x, y = next_points_lidar_quantized[0:2, idx]
            next_lidar_image[y, x] = next_depth_lidar[idx]

        # Remove points in vehicle bounding boxes
        for box in next_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=next_camera_intrinsics, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:, 0]))
                min_y = int(np.min(corners.T[:, 1]))
                max_x = int(np.max(corners.T[:, 0]))
                max_y = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                next_lidar_image[min_y:max_y, min_x:max_x] = 0

        # Now we need to convert image format to point cloud array format (y, x, z)
        next_lidar_points_y, next_lidar_points_x  = np.nonzero(next_lidar_image)
        next_lidar_points_z = next_lidar_image[next_lidar_points_y, next_lidar_points_x]

        # Backproject to camera frame as 3 x N
        x_y_homogeneous = np.stack([
            next_lidar_points_x,
            next_lidar_points_y,
            np.ones_like(next_lidar_points_x)],
            axis=0)

        x_y_lifted = np.matmul(np.linalg.inv(next_camera_intrinsics), x_y_homogeneous)
        x_y_z = x_y_lifted * np.expand_dims(next_lidar_points_z, axis=0)

        # To convert the lidar point cloud into a LidarPointCloud object, we need 4, N shape.
        # So we add a 4th fake intensity vector
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array, axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis=0)

        # Convert lidar point cloud into a nuScene LidarPointCloud object
        next_point_cloud = LidarPointCloud(x_y_z)

        # Now we can transform the points back to the lidar frame of reference
        next_point_cloud = camera_to_lidar_frame(
            nusc=nusc,
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_lidar_token,
            camera_token=next_camera_token)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        next_points_lidar_main, next_depth_lidar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=next_point_cloud,
            lidar_sensor_token=next_lidar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        # We need to do another step of filtering to filter out all the points who will be projected upon moving objects in the main frame
        next_lidar_image_main = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for idx in range(0, next_points_lidar_main_quantized.shape[-1]):
            x, y = next_points_lidar_main_quantized[0:2, idx]
            next_lidar_image_main[y, x] = next_depth_lidar_main[idx]

        # We do not want to reproject any points onto a moving object in the main frame. So we find out the moving objects in the main frame
        for box in main_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                min_x_main = int(np.min(corners.T[:, 0]))
                min_y_main = int(np.min(corners.T[:, 1]))
                max_x_main = int(np.max(corners.T[:, 0]))
                max_y_main = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                next_lidar_image_main[min_y_main:max_y_main, min_x_main:max_x_main] = 0

        # Convert image format to point cloud format
        next_lidar_points_main_y, next_lidar_points_main_x  = np.nonzero(next_lidar_image_main)
        next_lidar_points_main_z = next_lidar_image_main[next_lidar_points_main_y, next_lidar_points_main_x]

        # Stack y and x to 2 x N (x, y)
        next_points_lidar_main = np.stack([
            next_lidar_points_main_x,
            next_lidar_points_main_y],
            axis=0)
        next_depth_lidar_main = next_lidar_points_main_z

        next_points_lidar_main_quantized = np.round(next_points_lidar_main).astype(int)

        for point_idx in range(0, next_points_lidar_main_quantized.shape[1]):
            x = next_points_lidar_main_quantized[0, point_idx]
            y = next_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                next_depth_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = next_depth_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_forward_processed = n_forward_processed + 1

    # Initialize previous sample as current sample
    prev_sample = copy.deepcopy(current_sample)

    while prev_sample['prev'] != "" and n_backward_processed < n_backward:
        '''
        1. Load point cloud and image in `prev' non-keyframe
        2. Use any trained model to get the object bbox from non-keyframe image
        3. Remove points within bbox, backproject to point cloud
        4. Project points onto image plane in current camera frame
        5. Load point cloud and image in `prev' keyframe
        6. Project onto image to remove vehicle bounding boxes
        7. Backproject to point cloud and project to camera frame to yield lidar image
        '''

        # Get the token and sample data for the previous non-keyframe sample and move backward until keyframe
        prev_sample_nonkf = copy.deepcopy(prev_sample)
        prev_camera_sample_nonkf = nusc.get('sample_data', prev_sample_nonkf['data']['CAM_FRONT'])
        prev_lidar_sample_nonkf = nusc.get('sample_data', prev_sample_nonkf['data']['LIDAR_TOP'])

        # Get the token and sample data for the previous sample and move sample backward
        prev_sample_token = prev_sample['prev']
        prev_sample = nusc.get('sample', prev_sample_token)

        '''
        Process non-keyframe
        '''
        while prev_camera_sample_nonkf['timestamp'] >= prev_sample['timestamp']:

            # Get lidar and camera sensor token
            prev_lidar_nonkf_token = prev_lidar_sample_nonkf['token']
            prev_camera_nonkf_token = prev_camera_sample_nonkf['token']

            # Get intrinsics matrix
            prev_camera_intrinsics_nonkf = \
                nusc.get('calibrated_sensor', prev_camera_sample_nonkf['calibrated_sensor_token'])

            # Load non-keyframe image
            prev_camera_sample_nonkf_path = \
                os.path.join(nusc.dataroot, prev_camera_sample_nonkf['filename'])
            prev_camera_image_nonkf = data_utils.load_image(prev_camera_sample_nonkf_path)

            # Get height and width of non-keyframe image
            height_nonkf, width_nonkf = prev_camera_image_nonkf.shape[0:2]

            prev_points_lidar_nonkf, prev_depth_lidar_nonkf, _ = nusc_explorer.map_pointcloud_to_image(
                pointsensor_token=prev_lidar_nonkf_token,
                camera_token=prev_camera_nonkf_token)

            # Create image plane for lidar points
            prev_lidar_image_nonkf = np.zeros([height_nonkf, width_nonkf])
            prev_points_lidar_nonkf_quantized = np.round(prev_points_lidar_nonkf).astype(int)

            # Plot non-keyframe lidar points onto the image plane
            for idx in range(0, prev_points_lidar_nonkf_quantized.shape[-1]):
                x, y = prev_points_lidar_nonkf_quantized[0:2, idx]
                prev_lidar_image_nonkf[y, x] = prev_depth_lidar_nonkf[idx]

            # Remove points which belong to moving objects using semantic segmentation from detr model
            prev_boxes_nonkf = detections[prev_camera_nonkf_token]

            for box in prev_boxes_nonkf:
                (top_x , top_y, bottom_x , bottom_y) = box
                min_x = int(max(top_x, 0))
                min_y = int(max(top_y, 0))
                max_x = int(min(bottom_x, width_nonkf))
                max_y = int(min(bottom_y, height_nonkf))

                # Filter out the points inside the bounding boxes
                prev_lidar_image_nonkf[min_y:max_y, min_x:max_x] = 0

            # Now we need to convert image format to point cloud array format (y, x, z)
            prev_lidar_points_nonkf_y, prev_lidar_points_nonkf_x = np.nonzero(prev_lidar_image_nonkf)
            prev_lidar_points_nonkf_z = prev_lidar_image_nonkf[prev_lidar_points_nonkf_y, prev_lidar_points_nonkf_x]

            # Backproject to camera frame as 3 x N
            x_y_homogeneous_nonkf  = np.stack([
                prev_lidar_points_nonkf_x,
                prev_lidar_points_nonkf_y,
                np.ones_like(prev_lidar_points_nonkf_x)],
                axis=0)

            camera_intrinsics_nonkf = np.array(prev_camera_intrinsics_nonkf['camera_intrinsic'])
            x_y_lifted_nonkf = np.matmul(np.linalg.inv(camera_intrinsics_nonkf), x_y_homogeneous_nonkf)
            x_y_z_nonkf = x_y_lifted_nonkf * np.expand_dims(prev_lidar_points_nonkf_z, axis=0)

            # To convert the lidar point cloud into a LidarPointCloud object, we need 4, N shape.
            # So we add a 4th fake intensity vector
            fake_intensity_array_nonkf = np.ones(x_y_z_nonkf.shape[1])
            fake_intensity_array_nonkf = np.expand_dims(fake_intensity_array_nonkf, axis=0)
            x_y_z_nonkf = np.concatenate((x_y_z_nonkf, fake_intensity_array_nonkf), axis=0)

            # Convert lidar point cloud into a nuScene LidarPointCloud object
            prev_point_cloud_nonkf = LidarPointCloud(x_y_z_nonkf)

            # Now we can transform the points back to the lidar frame of reference
            prev_point_cloud_nonkf = camera_to_lidar_frame(
                nusc=nusc,
                point_cloud=prev_point_cloud_nonkf,
                lidar_sensor_token=prev_lidar_nonkf_token,
                camera_token=prev_camera_nonkf_token)

            # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
            prev_points_lidar_main_nonkf, prev_depth_lidar_main_nonkf, _ = point_cloud_to_image(
                nusc=nusc,
                point_cloud=prev_point_cloud_nonkf,
                lidar_sensor_token=prev_lidar_nonkf_token,
                camera_token=main_camera_token,
                min_distance_from_camera=1.0)

            # We need to do another step of filtering to filter out all the points that will be projected upon moving objects in the main frame
            prev_lidar_image_main_nonkf = np.zeros_like(main_lidar_image)

            # Plots depth values onto the image
            prev_points_lidar_main_quantized_nonkf = np.round(prev_points_lidar_main_nonkf).astype(int)

            for idx in range(0, prev_points_lidar_main_quantized_nonkf.shape[-1]):
                x, y = prev_points_lidar_main_quantized_nonkf[0:2, idx]
                prev_lidar_image_main_nonkf[y, x] = prev_depth_lidar_main_nonkf[idx]

            for box in main_boxes:
                if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                    corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                    min_x_main = int(np.min(corners.T[:, 0]))
                    min_y_main = int(np.min(corners.T[:, 1]))
                    max_x_main = int(np.max(corners.T[:, 0]))
                    max_y_main = int(np.max(corners.T[:, 1]))

                    # Filter out the points inside the bounding box
                    prev_lidar_image_main_nonkf[min_y_main:max_y_main, min_x_main:max_x_main] = 0

            # Convert image format to point cloud format
            prev_lidar_points_main_y_nonkf, prev_lidar_points_main_x_nonkf = np.nonzero(prev_lidar_image_main_nonkf)
            prev_lidar_points_main_z_nonkf = \
                prev_lidar_image_main_nonkf[prev_lidar_points_main_y_nonkf, prev_lidar_points_main_x_nonkf]

            # Stack y and x to 2 x N (x, y)
            prev_points_lidar_main_nonkf = np.stack([
                prev_lidar_points_main_x_nonkf,
                prev_lidar_points_main_y_nonkf],
                axis=0)
            prev_depth_lidar_main = prev_lidar_points_main_z_nonkf

            prev_points_lidar_main_quantized_nonkf = np.round(prev_points_lidar_main_nonkf).astype(int)

            for point_idx in range(0, prev_points_lidar_main_quantized_nonkf.shape[1]):
                x = prev_points_lidar_main_quantized_nonkf[0, point_idx]
                y = prev_points_lidar_main_quantized_nonkf[1, point_idx]

                is_not_occluded = \
                    main_validity_map[y, x] == 1 and \
                    prev_depth_lidar_main[point_idx] < main_lidar_image[y, x]

                if is_not_occluded:
                    main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
                elif main_validity_map[y, x] != 1:
                    main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
                    main_validity_map[y, x] = 1

            # Move camera token to the previous
            if prev_camera_sample_nonkf['prev'] != "" and prev_lidar_sample_nonkf['prev'] != "":
                prev_camera_sample_nonkf = nusc.get('sample_data', prev_camera_sample_nonkf['prev'])
            else:
                break

            # Lidar has higher frequency than camera, find the closest lidar token
            while prev_lidar_sample_nonkf['timestamp'] > prev_camera_sample_nonkf['timestamp'] and prev_lidar_sample_nonkf['prev'] != "":
                prev_lidar_sample_nonkf = nusc.get('sample_data', prev_lidar_sample_nonkf['prev'])

        '''
        Process keyframe
        '''
        # Get lidar and camera token in the previous sample
        prev_lidar_token = prev_sample['data']['LIDAR_TOP']
        prev_camera_token = prev_sample['data']['CAM_FRONT']

        # Get bounding box in image frame to remove vehicles from point cloud
        _, prev_boxes, prev_camera_intrinsics = nusc.get_sample_data(
            prev_camera_token,
            box_vis_level=BoxVisibility.ANY,
            use_flat_vehicle_coordinates=False)

        # Map prev frame point cloud to image so we can remove vehicle based on bounding boxes
        prev_points_lidar, prev_depth_lidar, _ = nusc_explorer.map_pointcloud_to_image(
            pointsensor_token=prev_lidar_token,
            camera_token=prev_camera_token)

        prev_lidar_image = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        prev_points_lidar_quantized = np.round(prev_points_lidar).astype(int)

        for idx in range(0, prev_points_lidar_quantized.shape[-1]):
            x, y = prev_points_lidar_quantized[0:2, idx]
            prev_lidar_image[y, x] = prev_depth_lidar[idx]

        # Remove points in vehicle bounding boxes
        for box in prev_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=prev_camera_intrinsics, normalize=True)[:2, :]
                min_x = int(np.min(corners.T[:, 0]))
                min_y = int(np.min(corners.T[:, 1]))
                max_x = int(np.max(corners.T[:, 0]))
                max_y = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                prev_lidar_image[min_y:max_y, min_x:max_x] = 0

        # Now we need to convert image format to point cloud array format
        prev_lidar_points_y, prev_lidar_points_x  = np.nonzero(prev_lidar_image)
        prev_lidar_points_z = prev_lidar_image[prev_lidar_points_y, prev_lidar_points_x]

        # Backproject to camera frame as 3 x N
        x_y_homogeneous = np.stack([
            prev_lidar_points_x,
            prev_lidar_points_y,
            np.ones_like(prev_lidar_points_x)],
            axis=0)

        x_y_lifted = np.matmul(np.linalg.inv(prev_camera_intrinsics), x_y_homogeneous)
        x_y_z = x_y_lifted * np.expand_dims(prev_lidar_points_z, axis=0)

        # To convert the lidar point cloud into a LidarPointCloud object, we need 4, N shape.
        # So we add a 4th fake intensity vector
        fake_intensity_array = np.ones(x_y_z.shape[1])
        fake_intensity_array = np.expand_dims(fake_intensity_array, axis=0)
        x_y_z = np.concatenate((x_y_z, fake_intensity_array), axis=0)

        # Convert lidar point cloud into a nuScene LidarPointCloud object
        prev_point_cloud = LidarPointCloud(x_y_z)

        # Now we can transform the points back to the lidar frame of reference
        prev_point_cloud = camera_to_lidar_frame(
            nusc=nusc,
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_lidar_token,
            camera_token=prev_camera_token)

        # Project the points to the image frame of reference as 2 x N x, y and 1 x N z arrays
        prev_points_lidar_main, prev_depth_lidar_main, _ = point_cloud_to_image(
            nusc=nusc,
            point_cloud=prev_point_cloud,
            lidar_sensor_token=prev_lidar_token,
            camera_token=main_camera_token,
            min_distance_from_camera=1.0)

        # We need to do another step of filtering to filter out all the points who will be projected upon moving objects in the main frame
        prev_lidar_image_main = np.zeros_like(main_lidar_image)

        # Plots depth values onto the image
        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for idx in range(0, prev_points_lidar_main_quantized.shape[-1]):
            x, y = prev_points_lidar_main_quantized[0:2, idx]
            prev_lidar_image_main[y, x] = prev_depth_lidar_main[idx]

        # We do not want to reproject any points onto a moving object in the main frame. So we find out the moving objects in the main frame
        for box in main_boxes:
            if box.name[:7] == 'vehicle' or box.name[:5] == 'human':
                corners = view_points(box.corners(), view=main_camera_intrinsic, normalize=True)[:2, :]
                min_x_main = int(np.min(corners.T[:, 0]))
                min_y_main = int(np.min(corners.T[:, 1]))
                max_x_main = int(np.max(corners.T[:, 0]))
                max_y_main = int(np.max(corners.T[:, 1]))

                # Filter out the points inside the bounding box
                prev_lidar_image_main[min_y_main:max_y_main, min_x_main:max_x_main] = 0

        # Convert image format to point cloud format 2 x N
        prev_lidar_points_main_y, prev_lidar_points_main_x  = np.nonzero(prev_lidar_image_main)
        prev_lidar_points_main_z = prev_lidar_image_main[prev_lidar_points_main_y, prev_lidar_points_main_x]

        # Stack y and x to 2 x N (x, y)
        prev_points_lidar_main = np.stack([
            prev_lidar_points_main_x,
            prev_lidar_points_main_y],
            axis=0)
        prev_depth_lidar_main = prev_lidar_points_main_z

        prev_points_lidar_main_quantized = np.round(prev_points_lidar_main).astype(int)

        for point_idx in range(0, prev_points_lidar_main_quantized.shape[1]):
            x = prev_points_lidar_main_quantized[0, point_idx]
            y = prev_points_lidar_main_quantized[1, point_idx]

            is_not_occluded = \
                main_validity_map[y, x] == 1 and \
                prev_depth_lidar_main[point_idx] < main_lidar_image[y, x]

            if is_not_occluded:
                main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
            elif main_validity_map[y, x] != 1:
                main_lidar_image[y, x] = prev_depth_lidar_main[point_idx]
                main_validity_map[y, x] = 1

        n_backward_processed = n_backward_processed + 1

    # need to convert this to the same format used by nuScenes to return Lidar points
    # nuscenes outputs this in the form of a xy tuple and depth. We do the same here.
    # we also make x -> y and y -> x to stay consistent with nuScenes
    return_points_lidar_y, return_points_lidar_x = np.nonzero(main_lidar_image)

    # Array of 1, N depth
    return_depth_lidar = main_lidar_image[return_points_lidar_y, return_points_lidar_x]

    # Array of 2, N x, y coordinates for lidar, swap (y, x) components to (x, y)
    return_points_lidar = np.stack([
        return_points_lidar_x,
        return_points_lidar_y],
        axis=0)

    return return_points_lidar, return_depth_lidar

def lidar_depth_map_from_token(nusc,
                               nusc_explorer,
                               current_sample_token):
    '''
    Picks current_sample_token as reference and projects lidar points onto the image plane.

    Arg(s):
        nusc : NuScenes Object
            nuScenes object instance
        nusc_explorer : NuScenesExplorer Object
            nuScenes explorer object instance
        current_sample_token : str
            token for accessing the current sample data
    Returns:
        numpy[float32] : H x W depth
    '''

    current_sample = nusc.get('sample', current_sample_token)
    lidar_token = current_sample['data']['LIDAR_TOP']
    main_camera_token = current_sample['data']['CAM_FRONT']

    # project the lidar frame into the camera frame
    main_points_lidar, main_depth_lidar, main_image = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=lidar_token,
        camera_token=main_camera_token)

    depth_map = points_to_depth_map(main_points_lidar, main_depth_lidar, main_image)

    return depth_map

def points_to_depth_map(points, depth, image):
    '''
    Plots the depth values onto the image plane

    Arg(s):
        points : numpy[float32]
            2 x N matrix in x, y
        depth : numpy[float32]
            N scales for z
        image : numpy[float32]
            H x W x 3 image for reference frame size
    Returns:
        numpy[float32] : H x W image with depth plotted
    '''

    # Plot points onto the image
    image = np.asarray(image)
    depth_map = np.zeros((image.shape[0], image.shape[1]))

    points_quantized = np.round(points).astype(int)

    for pt_idx in range(0, points_quantized.shape[1]):
        x = points_quantized[0, pt_idx]
        y = points_quantized[1, pt_idx]
        depth_map[y, x] = depth[pt_idx]

    return depth_map

def process_scene(_args):
    '''
    Processes one scene from first sample to last sample

    Arg(s):
        args : tuple(Object, Object, str, int, str, str, int, int, str, bool)
            tag_trainvaltest : str
                train, val, test split
            tag_daynight : str
                day, night split
            scene_id : str
                identifier for one scene
            first_sample_token : str
                token to identify first sample in the scene for fetching
            last_sample_token : str
                token to identify last sample in the scene for fetching
            n_forward : int
                number of forward (future) frames to reproject
            n_backward : int
                number of backward (previous) frames to reproject
            output_dirpath : str
                root of output directory
            paths_only : bool
                if set, then only produce paths
    Returns:
        list[str] : paths to camera image
        list[str] : paths to lidar depth map
        list[str] : paths to ground truth (merged lidar) depth map
    '''

    tag_trainvaltest, \
        tag_daynight, \
        scene_id, \
        first_sample_token, \
        last_sample_token, \
        n_forward, \
        n_backward, \
        output_dirpath, \
        paths_only = _args

    if tag_trainvaltest == 'train' or tag_trainvaltest == 'val':
        nusc = nusc_trainval
        nusc_explorer = nusc_explorer_trainval
    elif tag_trainvaltest == 'test':
        nusc = nusc_test
        nusc_explorer = nusc_explorer_test
    else:
        raise ValueError('Unsupport tag: {}'.format(tag_trainvaltest))

    # Instantiate the first sample id
    sample_id = 0
    sample_token = first_sample_token

    camera_image_paths = []
    lidar_paths = []
    ground_truth_paths = []
    camera_intrinsics_paths = []
    camera_absolute_pose_paths = []

    camera_intrinsics_dirpath = os.path.join(args.nuscenes_data_derived_dirpath, 'intrinsics')
    os.makedirs(camera_intrinsics_dirpath, exist_ok=True)

    detection_dirpath = os.path.join(
        args.nuscenes_data_derived_dirpath,
        'sample_data_detection')

    detection_json_path = os.path.join(detection_dirpath, scene_id + '.json')

    if not paths_only:
        f = open(detection_json_path, 'r')
        detections = json.loads(f.read())
    else:
        assert os.path.exists(detection_dirpath)
        assert os.path.exists(detection_json_path)

    print('Processing {}'.format(scene_id))

    # Iterate through all samples up to the last sample
    while sample_token != last_sample_token:

        # Fetch a single sample
        current_sample = nusc.get('sample', sample_token)
        scene_token = current_sample['scene_token']
        camera_token = current_sample['data']['CAM_FRONT']
        camera_sample = nusc.get('sample_data', camera_token)

        '''
        Set up paths
        '''
        camera_image_path = os.path.join(nusc.dataroot, camera_sample['filename'])

        dirpath, filename = os.path.split(camera_image_path)
        dirpath = dirpath.replace(nusc.dataroot, output_dirpath)
        filename, ext = os.path.splitext(filename)

        # Create image path
        image_dirpath = dirpath.replace(
            'samples',
            os.path.join('image', scene_id))
        image_filename = filename + ext

        image_path = os.path.join(
            image_dirpath,
            image_filename)

        # Create lidar path
        lidar_dirpath = dirpath.replace(
            'samples',
            os.path.join('lidar', scene_id))
        lidar_filename = filename + '.png'

        lidar_path = os.path.join(
            lidar_dirpath,
            lidar_filename)

        # Create ground truth path
        ground_truth_dirpath = dirpath.replace(
            'samples',
            os.path.join('ground_truth', scene_id))
        ground_truth_filename = filename + '.png'

        ground_truth_path = os.path.join(
            ground_truth_dirpath,
            ground_truth_filename)

        # Create camera intrinsics path
        camera_intrinsics_path = os.path.join(
            camera_intrinsics_dirpath,
            '{}_{}.npy'.format(str(scene_id).zfill(4), scene_token))

        # Create camera absolute pose path
        camera_absolute_pose_dirpath = dirpath.replace(
            'samples',
            os.path.join('absolute_pose', scene_id))
        camera_absolute_pose_filename = filename + '.npy'

        camera_absolute_pose_path = os.path.join(
            camera_absolute_pose_dirpath,
            camera_absolute_pose_filename)

        # In case multiple threads create same directory
        dirpaths = [
            image_dirpath,
            lidar_dirpath,
            ground_truth_dirpath,
            camera_absolute_pose_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                try:
                    os.makedirs(dirpath)
                except Exception:
                    pass

        '''
        Store file paths
        '''
        camera_image_paths.append(image_path)
        lidar_paths.append(lidar_path)
        ground_truth_paths.append(ground_truth_path)
        camera_intrinsics_paths.append(camera_intrinsics_path)
        camera_absolute_pose_paths.append(camera_absolute_pose_path)

        if not paths_only:
            # Get intrinsics for camera
            _, _, camera_intrinsics = nusc.get_sample_data(
                camera_token,
                box_vis_level=BoxVisibility.ANY,
                use_flat_vehicle_coordinates=False)

            if not os.path.exists(camera_intrinsics_path):
                np.save(camera_intrinsics_path, camera_intrinsics)

            '''
            Get camera data
            '''
            camera_image = data_utils.load_image(camera_image_path)
            shutil.copy(camera_image_path, image_path)

            '''
            Get absolute pose
            '''
            current_camera_token = current_sample['data']['CAM_FRONT']

            camera_absolute_pose = get_extrinsics_matrix(
                nusc=nusc,
                sensor_token=current_camera_token)

            np.save(camera_absolute_pose_path, camera_absolute_pose)

            '''
            Get lidar points projected to an image and save as PNG
            '''
            lidar_depth = lidar_depth_map_from_token(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token)

            data_utils.save_depth(lidar_depth, lidar_path)

            '''
            Merge forward and backward point clouds for lidar
            '''
            # Merges n_forward and n_backward number of point clouds to frame at sample token
            points_lidar, depth_lidar = merge_lidar_point_clouds(
                nusc=nusc,
                nusc_explorer=nusc_explorer,
                current_sample_token=sample_token,
                n_forward=n_forward,
                n_backward=n_backward,
                detections=detections)

            '''
            Project point cloud onto the image plane and save as PNG
            '''
            # Merges n_forward and n_backward number of point clouds to frame at sample token
            # but in this case we need the lidar image so that we can save it
            ground_truth = points_to_depth_map(points_lidar, depth_lidar, camera_image)

            # Filtering the dense lidar points using photometric error
            if args.filter_threshold_photometric_reconstruction > 0:

                # Convert to torch to utilize existing backprojection function
                ground_truth = torch.from_numpy(ground_truth).unsqueeze(0).float()
                camera_image = torch.from_numpy(camera_image).permute(2, 0, 1).unsqueeze(0).float()
                camera_intrinsics = torch.from_numpy(camera_intrinsics).unsqueeze(0).float()

                # Backproject ground truth points to 3D
                shape = camera_image.shape
                points = data_utils.backproject_to_camera(ground_truth, camera_intrinsics, shape)

                # Get the previous and next keyframe tokens
                prev_sample_token = current_sample['prev']
                next_sample_token = current_sample['next']

                # Preallocate error maps
                error_map_next = torch.ones_like(ground_truth)
                error_map_prev = torch.ones_like(ground_truth)

                if prev_sample_token != '':
                    prev_sample = nusc.get('sample', prev_sample_token)
                    prev_filename = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])['filename']
                    prev_camera_image_path = os.path.join(nusc.dataroot, prev_filename)
                    assert os.path.exists(prev_camera_image_path)

                    prev_camera_image = \
                        torch.from_numpy(data_utils.load_image(prev_camera_image_path, data_format='CHW')).unsqueeze(0)
                    prev_camera_token = prev_sample['data']['CAM_FRONT']

                    current_to_prev_pose = get_relative_pose(
                        nusc,
                        source_sensor_token=current_camera_token,
                        target_sensor_token=prev_camera_token)

                    current_to_prev_pose = torch.from_numpy(current_to_prev_pose)

                    target_xy_prev = data_utils.project_to_pixel(
                        points=points,
                        pose=current_to_prev_pose,
                        intrinsics=camera_intrinsics,
                        shape=shape)

                    reprojected_image_from_prev = data_utils.grid_sample(
                        image=prev_camera_image,
                        target_xy=target_xy_prev,
                        shape=shape,
                        padding_mode='zeros')

                    # Mean error across channels
                    error_map_prev = torch.mean(torch.abs(camera_image - reprojected_image_from_prev), dim=1)
                else:
                    error_map_prev = None

                if next_sample_token != '':
                    next_sample = nusc.get('sample', next_sample_token)
                    next_filename = nusc.get('sample_data', next_sample['data']['CAM_FRONT'])['filename']
                    next_camera_image_path = os.path.join(nusc.dataroot, next_filename)
                    assert os.path.exists(next_camera_image_path)

                    next_camera_image = \
                        torch.from_numpy(data_utils.load_image(next_camera_image_path, data_format='CHW')).unsqueeze(0)
                    next_camera_token = next_sample['data']['CAM_FRONT']

                    current_to_next_pose = get_relative_pose(
                        nusc,
                        source_sensor_token=current_camera_token,
                        target_sensor_token=next_camera_token)

                    current_to_next_pose = torch.from_numpy(current_to_next_pose)
                    target_xy_next = data_utils.project_to_pixel(
                        points=points,
                        pose=current_to_next_pose,
                        intrinsics=camera_intrinsics,
                        shape=shape)

                    reprojected_image_from_next = data_utils.grid_sample(
                        image=next_camera_image,
                        target_xy=target_xy_next,
                        shape=shape,
                        padding_mode='zeros')

                    # Mean error across channels
                    error_map_next = torch.mean(torch.abs(camera_image - reprojected_image_from_next), dim=1)
                else:
                    error_map_next = None

                # Check if we are on either ends of a sequence, if not then we choose the minimum error
                if error_map_prev is None:
                    error_map = error_map_next
                elif error_map_next is None:
                    error_map = error_map_prev
                elif error_map_prev is not None and error_map_next is not None:
                    error_map, error_map_min_indices = torch.min(torch.cat([error_map_prev, error_map_next]), dim=0, keepdim=True)
                else:
                    raise ValueError('Both forward and backward reprojected error maps are None.')

                # Convert error map to numpy format
                error_map = error_map.numpy()

                photometric_valid_mask = np.where(
                    error_map < args.filter_threshold_photometric_reconstruction,
                    np.ones_like(error_map),
                    np.zeros_like(error_map))

                ground_truth = ground_truth * photometric_valid_mask
                ground_truth = np.squeeze(ground_truth.numpy())

            # Filter outliers from ground truth based on differences in neighboring points
            ground_truth = torch.from_numpy(ground_truth).unsqueeze(0).float()
            validity_map_ground_truth = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                torch.zeros_like(ground_truth))

            ground_truth, _ = outlier_removal.remove_outliers(
                sparse_depth=ground_truth,
                validity_map=validity_map_ground_truth)

            # Save depth map as PNG
            data_utils.save_depth(ground_truth, ground_truth_path)

        '''
        Move to next sample in scene
        '''
        sample_id = sample_id + 1
        sample_token = current_sample['next']

    print('Finished {} samples in {}'.format(sample_id, scene_id))

    return (tag_trainvaltest,
            tag_daynight,
            camera_image_paths,
            lidar_paths,
            ground_truth_paths,
            camera_intrinsics_paths,
            camera_absolute_pose_paths)


'''
Main function
'''
if __name__ == '__main__':

    use_multithread = args.n_thread > 1 and not args.debug

    pool_inputs = []
    pool_results = []

    # Training all, day, and night paths
    train_camera_image_paths = []
    train_lidar_paths = []
    train_ground_truth_paths = []
    train_intrinsics_paths = []
    train_absolute_pose_paths = []

    train_day_camera_image_paths = []
    train_day_lidar_paths = []
    train_day_ground_truth_paths = []
    train_day_intrinsics_paths = []
    train_day_absolute_pose_paths = []

    train_night_camera_image_paths = []
    train_night_lidar_paths = []
    train_night_ground_truth_paths = []
    train_night_intrinsics_paths = []
    train_night_absolute_pose_paths = []

    # Validation all, day, and night paths
    val_camera_image_paths = []
    val_lidar_paths = []
    val_ground_truth_paths = []
    val_intrinsics_paths = []
    val_absolute_pose_paths = []

    val_day_camera_image_paths = []
    val_day_lidar_paths = []
    val_day_ground_truth_paths = []
    val_day_intrinsics_paths = []
    val_day_absolute_pose_paths = []

    val_night_camera_image_paths = []
    val_night_lidar_paths = []
    val_night_ground_truth_paths = []
    val_night_intrinsics_paths = []
    val_night_absolute_pose_paths = []

    # Testing all, day, and night paths
    test_camera_image_paths = []
    test_lidar_paths = []
    test_ground_truth_paths = []
    test_intrinsics_paths = []
    test_absolute_pose_paths = []

    test_day_camera_image_paths = []
    test_day_lidar_paths = []
    test_day_ground_truth_paths = []
    test_day_intrinsics_paths = []
    test_day_absolute_pose_paths = []

    test_night_camera_image_paths = []
    test_night_lidar_paths = []
    test_night_ground_truth_paths = []
    test_night_intrinsics_paths = []
    test_night_absolute_pose_paths = []

    train_ids, val_ids, test_ids = get_train_val_test_split_scene_ids(
        TRAIN_DATA_SPLIT_FILEPATH,
        VAL_DATA_SPLIT_FILEPATH,
        TEST_DATA_SPLIT_FILEPATH)

    day_ids , night_ids = get_train_val_test_daynight_split_scene_ids()

    scene_ids = train_ids + val_ids + test_ids

    n_train = len(train_ids)
    n_val = len(val_ids)
    n_test = len(test_ids)
    print('Total Scenes to process: {}'.format(len(scene_ids)))
    print('Training: {}  Validation: {}  Testing: {}'.format(n_train, n_val, n_test))

    print('Number of daytime scene ids: {}'.format(len(day_ids)))
    print('Number of nighttime scene ids: {}'.format(len(night_ids)))

    # Add all tasks for processing each scene to pool inputs
    for idx in range(0, MAX_TRAIN_SCENES + MAX_TEST_SCENES):

        if idx < MAX_TRAIN_SCENES:
            nusc = nusc_explorer_trainval.nusc
        else:
            idx = idx - MAX_TRAIN_SCENES
            nusc = nusc_explorer_test.nusc

        current_scene = nusc.scene[idx]
        scene_id = current_scene['name']

        if scene_id in train_ids:
            tag_trainvaltest = 'train'
        elif scene_id in val_ids:
            tag_trainvaltest = 'val'
        elif scene_id in test_ids:
            tag_trainvaltest = 'test'
        else:
            raise ValueError('{} cannot be found in train or val split'.format(scene_id))

        if scene_id in night_ids:
            tag_daynight = 'night'
        elif scene_id in day_ids:
            tag_daynight = 'day'
        else:
            raise ValueError('{} do not have a daynight tag'.format(scene_id))

        first_sample_token = current_scene['first_sample_token']
        last_sample_token = current_scene['last_sample_token']

        inputs = [
            tag_trainvaltest,
            tag_daynight,
            scene_id,
            first_sample_token,
            last_sample_token,
            args.n_forward_frames_to_reproject,
            args.n_backward_frames_to_reproject,
            args.nuscenes_data_derived_dirpath,
            args.paths_only,
        ]

        pool_inputs.append(inputs)

        if not use_multithread:
            pool_results.append(process_scene(inputs))

    if use_multithread:
        torch.set_num_threads(1)
        # Create pool of threads
        with mp.Pool(args.n_thread) as pool:
            # Will fork n_thread to process scene
            pool_results = pool.map(process_scene, pool_inputs)

    # Unpack output paths
    for results in pool_results:

        tag_trainvaltest, \
            tag_daynight, \
            camera_image_scene_paths, \
            lidar_scene_paths, \
            ground_truth_scene_paths, \
            camera_intrinsics_paths, \
            camera_absolute_pose_paths = results

        if tag_trainvaltest == 'train':
            train_camera_image_paths.extend(camera_image_scene_paths)
            train_lidar_paths.extend(lidar_scene_paths)
            train_ground_truth_paths.extend(ground_truth_scene_paths)
            train_intrinsics_paths.extend(camera_intrinsics_paths)
            train_absolute_pose_paths.extend(camera_absolute_pose_paths)

            if tag_daynight == 'day':
                train_day_camera_image_paths.extend(camera_image_scene_paths)
                train_day_lidar_paths.extend(lidar_scene_paths)
                train_day_ground_truth_paths.extend(ground_truth_scene_paths)
                train_day_intrinsics_paths.extend(camera_intrinsics_paths)
                train_day_absolute_pose_paths.extend(camera_absolute_pose_paths)
            elif tag_daynight == 'night':
                train_night_camera_image_paths.extend(camera_image_scene_paths)
                train_night_lidar_paths.extend(lidar_scene_paths)
                train_night_ground_truth_paths.extend(ground_truth_scene_paths)
                train_night_intrinsics_paths.extend(camera_intrinsics_paths)
                train_night_absolute_pose_paths.extend(camera_absolute_pose_paths)
            else:
                raise ValueError('Found invalid daynight tag: {}'.format(tag_daynight))

        elif tag_trainvaltest == 'val':
            val_camera_image_paths.extend(camera_image_scene_paths)
            val_lidar_paths.extend(lidar_scene_paths)
            val_ground_truth_paths.extend(ground_truth_scene_paths)
            val_intrinsics_paths.extend(camera_intrinsics_paths)
            val_absolute_pose_paths.extend(camera_absolute_pose_paths)

            if tag_daynight == 'day':
                val_day_camera_image_paths.extend(camera_image_scene_paths)
                val_day_lidar_paths.extend(lidar_scene_paths)
                val_day_ground_truth_paths.extend(ground_truth_scene_paths)
                val_day_intrinsics_paths.extend(camera_intrinsics_paths)
                val_day_absolute_pose_paths.extend(camera_absolute_pose_paths)
            elif tag_daynight == 'night':
                val_night_camera_image_paths.extend(camera_image_scene_paths)
                val_night_lidar_paths.extend(lidar_scene_paths)
                val_night_ground_truth_paths.extend(ground_truth_scene_paths)
                val_night_intrinsics_paths.extend(camera_intrinsics_paths)
                val_night_absolute_pose_paths.extend(camera_absolute_pose_paths)
            else:
                raise ValueError('Found invalid daynight tag: {}'.format(tag_daynight))

        elif tag_trainvaltest == 'test':
            test_camera_image_paths.extend(camera_image_scene_paths)
            test_lidar_paths.extend(lidar_scene_paths)
            test_ground_truth_paths.extend(ground_truth_scene_paths)
            test_intrinsics_paths.extend(camera_intrinsics_paths)
            test_absolute_pose_paths.extend(camera_absolute_pose_paths)

            if tag_daynight == 'day':
                test_day_camera_image_paths.extend(camera_image_scene_paths)
                test_day_lidar_paths.extend(lidar_scene_paths)
                test_day_ground_truth_paths.extend(ground_truth_scene_paths)
                test_day_intrinsics_paths.extend(camera_intrinsics_paths)
                test_day_absolute_pose_paths.extend(camera_absolute_pose_paths)
            elif tag_daynight == 'night':
                test_night_camera_image_paths.extend(camera_image_scene_paths)
                test_night_lidar_paths.extend(lidar_scene_paths)
                test_night_ground_truth_paths.extend(ground_truth_scene_paths)
                test_night_intrinsics_paths.extend(camera_intrinsics_paths)
                test_night_absolute_pose_paths.extend(camera_absolute_pose_paths)
            else:
                raise ValueError('Found invalid daynight tag: {}'.format(tag_daynight))

        else:
            raise ValueError('Found invalid tag: {}'.format(tag_trainvaltest))

    '''
    Subsample from validation and testing set
    '''
    # Validation set
    val_camera_image_subset_paths = val_camera_image_paths[::2]
    val_lidar_subset_paths = val_lidar_paths[::2]
    val_ground_truth_subset_paths = val_ground_truth_paths[::2]
    val_intrinsics_subset_paths = val_intrinsics_paths[::2]
    val_absolute_pose_subset_paths = val_absolute_pose_paths[::2]

    val_day_camera_image_subset_paths = val_day_camera_image_paths[::2]
    val_day_lidar_subset_paths = val_day_lidar_paths[::2]
    val_day_ground_truth_subset_paths = val_day_ground_truth_paths[::2]
    val_day_intrinsics_subset_paths = val_day_intrinsics_paths[::2]
    val_day_absolute_pose_subset_paths = val_day_absolute_pose_paths[::2]

    val_night_camera_image_subset_paths = val_night_camera_image_paths[::2]
    val_night_lidar_subset_paths = val_night_lidar_paths[::2]
    val_night_ground_truth_subset_paths = val_night_ground_truth_paths[::2]
    val_night_intrinsics_subset_paths = val_night_intrinsics_paths[::2]
    val_night_absolute_pose_subset_paths = val_night_absolute_pose_paths[::2]

    # Testing set
    test_camera_image_subset_paths = test_camera_image_paths[::2]
    test_lidar_subset_paths = test_lidar_paths[::2]
    test_ground_truth_subset_paths = test_ground_truth_paths[::2]
    test_intrinsics_subset_paths = test_intrinsics_paths[::2]
    test_absolute_pose_subset_paths = test_absolute_pose_paths[::2]

    test_day_camera_image_subset_paths = test_day_camera_image_paths[::2]
    test_day_lidar_subset_paths = test_day_lidar_paths[::2]
    test_day_ground_truth_subset_paths = test_day_ground_truth_paths[::2]
    test_day_intrinsics_subset_paths = test_day_intrinsics_paths[::2]
    test_day_absolute_pose_subset_paths = test_day_absolute_pose_paths[::2]

    test_night_camera_image_subset_paths = test_night_camera_image_paths[::2]
    test_night_lidar_subset_paths = test_night_lidar_paths[::2]
    test_night_ground_truth_subset_paths = test_night_ground_truth_paths[::2]
    test_night_intrinsics_subset_paths = test_night_intrinsics_paths[::2]
    test_night_absolute_pose_subset_paths = test_night_absolute_pose_paths[::2]

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
                    'ground truth',
                    train_absolute_pose_paths,
                    TRAIN_ABSOLUTE_POSE_FILEPATH
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
                    'day ground truth',
                    train_day_absolute_pose_paths,
                    TRAIN_DAY_ABSOLUTE_POSE_FILEPATH
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
                    'night ground truth',
                    train_night_absolute_pose_paths,
                    TRAIN_NIGHT_ABSOLUTE_POSE_FILEPATH
                ]
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
                    'ground truth',
                    val_absolute_pose_paths,
                    VAL_ABSOLUTE_POSE_FILEPATH
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
                    val_day_absolute_pose_paths,
                    VAL_DAY_ABSOLUTE_POSE_SUBSET_FILEPATH
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
                ]
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
                ]
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
                ]
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
