import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, cv2, argparse, random
import multiprocessing as mp
import numpy as np
from natsort import natsorted
from sklearn.cluster import MiniBatchKMeans
sys.path.insert(0, './')
import utils.src.data_utils as data_utils


N_CLUSTER = 1500
O_HEIGHT = 968
O_WIDTH = 1296
R_HEIGHT = 480
R_WIDTH = 640
N_HEIGHT = 448
N_WIDTH = 608
MIN_POINTS = 1100
TEMPORAL_WINDOW = 3
RANDOM_SEED = 1
TEST_SET_SUBSAMPLE_FACTOR = 10

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


SCANNET_ROOT_DIRPATH = os.path.join('data', 'scannet')
SCANNET_TRAIN_DIRPATH = os.path.join(SCANNET_ROOT_DIRPATH, 'scans')
SCANNET_TEST_DIRPATH = os.path.join(SCANNET_ROOT_DIRPATH, 'scans_test')

SCANNET_DERIVED_DIRPATH = os.path.join('data', 'scannet_derived')

TRAIN_REF_DIRPATH = os.path.join('training', 'scannet')
VAL_REF_DIRPATH = os.path.join('validation', 'scannet')
TEST_REF_DIRPATH = os.path.join('testing', 'scannet')

TRAIN_SUPERVISED_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'supervised')
TRAIN_UNSUPERVISED_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'unsupervised')

# Define output paths for supervised training
TRAIN_SUPERVISED_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'scannet_train_image_{}.txt')
TRAIN_SUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'scannet_train_sparse_depth_{}.txt')
TRAIN_SUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'scannet_train_ground_truth_{}.txt')
TRAIN_SUPERVISED_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'scannet_train_intrinsics_{}.txt')

# Define output paths for for unsupervised training
TRAIN_UNSUPERVISED_IMAGES_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'scannet_train_images_{}.txt')
TRAIN_UNSUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'scannet_train_sparse_depth_{}.txt')
TRAIN_UNSUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'scannet_train_ground_truth_{}.txt')
TRAIN_UNSUPERVISED_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'scannet_train_intrinsics_{}.txt')

# Test file paths
TEST_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'scannet_test_image_{}.txt')
TEST_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'scannet_test_sparse_depth_{}.txt')
TEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'scannet_test_ground_truth_{}.txt')
TEST_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'scannet_test_intrinsics_{}.txt')


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            ground truth path at time t=0
            sparse depth sampler type
            target number of sparse points to sample
            minimum number of sparse points to sample
            height of image and ground truth after cropping away black edge regions
            width of image and ground truth after cropping away black edge regions
            save image as image triplet if set
    Returns:
        str : output image path at time t=0
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output ground truth path at time t=0
    '''

    image0_path, \
        image1_path, \
        image2_path, \
        ground_truth_path, \
        sparse_depth_distro_type, \
        n_points, \
        min_points, \
        n_height, \
        n_width, \
        save_image_triplet = inputs

    # Load image (for corner detection) to generate valid map
    image0 = cv2.imread(image0_path)
    image0 = np.float32(cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY))

    # Load dense depth
    ground_truth = data_utils.load_depth(ground_truth_path, multiplier=1000.0)
    assert ground_truth.shape == (R_HEIGHT, R_WIDTH)

    image0 = cv2.resize(image0, (R_WIDTH, R_HEIGHT), interpolation=cv2.INTER_AREA)

    # Crop away black borders
    d_height = R_HEIGHT - n_height
    d_width = R_WIDTH - n_width

    y_start = d_height // 2
    x_start = d_width // 2
    y_end = y_start + n_height
    x_end = x_start + n_width

    image0 = image0[y_start:y_end, x_start:x_end]
    ground_truth = ground_truth[y_start:y_end, x_start:x_end]

    if sparse_depth_distro_type == 'corner':
        N_INIT_CORNER = 30000

        # Run Harris corner detector
        corners = cv2.cornerHarris(image0, blockSize=5, ksize=3, k=0.04)

        # Remove the corners that are located on invalid depth locations
        corners = corners * np.where(ground_truth > 0.0, 1.0, 0.0)

        # Vectorize corner map to 1D vector and select N_INIT_CORNER corner locations
        corners = corners.ravel()
        corner_locations = np.argsort(corners)[0:N_INIT_CORNER]

        # Get locations of corners as indices as (x, y)
        corner_locations = np.unravel_index(
            corner_locations,
            (image0.shape[0], image0.shape[1]))

        # Convert to (y, x) convention
        corner_locations = \
            np.transpose(np.array([corner_locations[0], corner_locations[1]]))

        # Cluster them into n_points (number of output points)
        kmeans = MiniBatchKMeans(
            n_clusters=n_points,
            max_iter=2,
            n_init=1,
            init_size=None,
            random_state=RANDOM_SEED,
            reassignment_ratio=1e-11)
        kmeans.fit(corner_locations)

        # Use k-Means means as corners
        selected_indices = kmeans.cluster_centers_.astype(np.uint16)

    elif sparse_depth_distro_type == 'uniform':
        indices = \
            np.array([[h, w] for h in range(n_height) for w in range(n_width)])

        # Randomly select n_points number of points
        selected_indices = \
            np.random.permutation(range(n_height * n_width))[0:n_points]
        selected_indices = indices[selected_indices]

    else:
        raise ValueError('Unsupported sparse depth distribution type: {}'.format(
            sparse_depth_distro_type))

    # Convert the indices into validity map
    validity_map = np.zeros_like(image0).astype(np.int16)
    validity_map[selected_indices[:, 0], selected_indices[:, 1]] = 1.0

    # Build validity map from selected points, keep only ones greater than 0
    validity_map = np.where(validity_map * ground_truth > 0.0, 1.0, 0.0)

    # Get sparse depth based on validity map
    sparse_depth = validity_map * ground_truth

    # Shape check
    error_flag = False

    if np.squeeze(sparse_depth).shape != (n_height, n_width):
        error_flag = True
        print('FAILED: np.squeeze(sparse_depth).shape != ({}, {})'.format(n_height, n_width))

    # Depth value check
    if np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0:
        error_flag = True
        print('FAILED: np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0')

    if np.sum(np.where(validity_map > 0.0, 1.0, 0.0)) < min_points:
        error_flag = True
        print('FAILED: np.sum(np.where(validity_map > 0.0, 1.0, 0.0)) < MIN_POINTS', np.sum(np.where(validity_map > 0.0, 1.0, 0.0)))

    if np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < min_points:
        error_flag = True
        print('FAILED: np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < MIN_POINTS')

    # NaN check
    if np.any(np.isnan(sparse_depth)):
        error_flag = True
        print('FAILED: np.any(np.isnan(sparse_depth))')

    if not error_flag:

        image0 = cv2.imread(image0_path)

        image0 = cv2.resize(image0, (R_WIDTH, R_HEIGHT), interpolation=cv2.INTER_AREA)
        image0 = image0[y_start:y_end, x_start:x_end, :]

        if save_image_triplet:
            # Read images and concatenate together
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            image1 = cv2.resize(image1, (R_WIDTH, R_HEIGHT), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (R_WIDTH, R_HEIGHT), interpolation=cv2.INTER_AREA)

            image1 = image1[y_start:y_end, x_start:x_end, :]
            image2 = image2[y_start:y_end, x_start:x_end, :]

            imagec = np.concatenate([image1, image0, image2], axis=1)
        else:
            imagec = None

        # Example: data/scannet/scans_test/scene0799_00/export/color/1.jpg
        image_output_path = image0_path \
            .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH) \
            .replace(os.path.join('export', 'color'), 'image')

        if imagec is None:
            images_output_path = None
        else:
            images_output_path = image0_path \
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH) \
                .replace(os.path.join('export', 'color'), 'images')

        sparse_depth_output_path = ground_truth_path \
            .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH) \
            .replace(os.path.join('export', 'depth'), 'sparse_depth')
        ground_truth_output_path = ground_truth_path \
            .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH) \
            .replace(os.path.join('export', 'depth'), 'ground_truth')

        image_output_dirpath = os.path.dirname(image_output_path)

        if images_output_path is not None:
            images_output_dirpath = os.path.dirname(images_output_path)
        else:
            images_output_dirpath = None

        sparse_depth_output_dirpath = os.path.dirname(sparse_depth_output_path)
        ground_truth_output_dirpath = os.path.dirname(ground_truth_output_path)

        # Create output directories
        output_dirpaths = [
            image_output_dirpath,
            images_output_dirpath,
            sparse_depth_output_dirpath,
            ground_truth_output_dirpath
        ]

        for dirpath in output_dirpaths:
            if dirpath is not None and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

        # Write to file
        cv2.imwrite(image_output_path, image0)

        if imagec is not None:
            cv2.imwrite(images_output_path, imagec)

        data_utils.save_depth(sparse_depth, sparse_depth_output_path, multiplier=256.0)
        data_utils.save_depth(ground_truth, ground_truth_output_path, multiplier=256.0)
    else:
        print('Found error in {}'.format(ground_truth_path))
        image_output_path = 'error'
        images_output_path = 'error'
        sparse_depth_output_path = 'error'
        ground_truth_output_path = 'error'

    return (image_output_path,
            images_output_path,
            sparse_depth_output_path,
            ground_truth_output_path)

def setup_dataset_scannet_training(sparse_depth_distro_type,
                                   n_points,
                                   min_points,
                                   n_height,
                                   n_width,
                                   temporal_window,
                                   fast_forward):
    '''
    Fetch image and ground truth paths for training

    Arg(s):
        sparse_depth_distro_type : bool
            sampling strategy for sparse points
        n_points : int
            number of sparse points to sample
        min_points : int
            minimum number of sparse points to accept a sample into training set
        n_height : int
            height of image and ground truth after cropping away black edges
        n_width : int
            width of image and ground truth after cropping away black edges
        temporal_window : int
            window to sample image triplet on left middle and right of window
        fast_forward : bool
            if set, then fast forward through sequences that have already been processed
    '''

    # Define output paths for supervised training
    train_supervised_image_output_paths = []
    train_supervised_sparse_depth_output_paths = []
    train_supervised_ground_truth_output_paths = []
    train_supervised_intrinsics_output_paths = []

    # Define output paths for unsupervised training
    train_unsupervised_images_output_paths = []
    train_unsupervised_sparse_depth_output_paths = []
    train_unsupervised_ground_truth_output_paths = []
    train_unsupervised_intrinsics_output_paths = []

    w = int(temporal_window // 2)

    train_sequence_dirpaths = natsorted(glob.glob(os.path.join(SCANNET_TRAIN_DIRPATH, '*/')))

    for train_sequence_dirpath in train_sequence_dirpaths:

        image_paths = natsorted(glob.glob(
            os.path.join(train_sequence_dirpath, 'export', 'color', '*.jpg')))
        ground_truth_paths = natsorted(glob.glob(
            os.path.join(train_sequence_dirpath, 'export', 'depth', '*.png')))

        n_sample = len(image_paths)

        assert n_sample == len(ground_truth_paths)

        intrinsics_path = \
            os.path.join(train_sequence_dirpath, 'export', 'intrinsic', 'intrinsic_color.txt')

        # Load and convert intrinsics to 3 x 3
        intrinsics = np.loadtxt(intrinsics_path)
        intrinsics = intrinsics[:3, :3]

        # Adjust based on resizing and cropping
        scale_factor_x = R_WIDTH / O_WIDTH
        scale_factor_y = R_HEIGHT / O_HEIGHT

        d_height = R_HEIGHT - n_height
        d_width = R_WIDTH - n_width
        offset_x = d_width // 2
        offset_y = d_height // 2

        intrinsics[0, 0] = intrinsics[0, 0] * scale_factor_x
        intrinsics[1, 1] = intrinsics[1, 1] * scale_factor_y
        intrinsics[0, 2] = intrinsics[0, 2] * scale_factor_x - offset_x
        intrinsics[1, 2] = intrinsics[1, 2] * scale_factor_x - offset_y

        intrinsics_output_path = intrinsics_path \
            .replace(os.path.join('intrinsic', 'intrinsic_color.txt'), 'intrinsics.npy')

        np.save(intrinsics_output_path, intrinsics)

        # Get all existing paths
        image_output_dirpath = os.path.dirname(
            image_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'color'), 'image'))
        images_output_dirpath = os.path.dirname(
            image_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'color'), 'images'))
        sparse_depth_output_dirpath = os.path.dirname(
            ground_truth_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'depth'), 'sparse_depth'))
        ground_truth_output_dirpath = os.path.dirname(
            ground_truth_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'depth'), 'ground_truth'))

        image_output_paths = natsorted(glob.glob(os.path.join(image_output_dirpath, '*.jpg')))
        images_output_paths = natsorted(glob.glob(os.path.join(images_output_dirpath, '*.jpg')))
        sparse_depth_output_paths = natsorted(glob.glob(os.path.join(sparse_depth_output_dirpath, '*.png')))
        ground_truth_output_paths = natsorted(glob.glob(os.path.join(ground_truth_output_dirpath, '*.png')))

        is_exists_output_dirpaths = \
            os.path.exists(image_output_dirpath) and \
            os.path.exists(images_output_dirpath) and \
            os.path.exists(sparse_depth_output_dirpath) and \
            os.path.exists(ground_truth_output_dirpath) and \
            len(image_output_paths) == len(sparse_depth_output_paths) and \
            len(image_output_paths) == len(ground_truth_output_paths)

        if fast_forward and is_exists_output_dirpaths:

            print('Found {} samples for supervised and {} samples for unsupervised training in: {}'.format(
                len(image_output_paths), len(images_output_paths), train_sequence_dirpath))

            # Append all training paths
            train_supervised_image_output_paths.extend(image_output_paths)
            train_supervised_sparse_depth_output_paths.extend(sparse_depth_output_paths)
            train_supervised_ground_truth_output_paths.extend(ground_truth_output_paths)
            train_supervised_intrinsics_output_paths.extend([intrinsics_output_path] * len(image_output_paths))

            images_output_filenames = [
                os.path.splitext(os.path.basename(path))[0]
                for path in images_output_paths
            ]

            for depth_idx, sparse_depth_output_path in enumerate(sparse_depth_output_paths):
                sparse_depth_output_filename = os.path.splitext(os.path.basename(sparse_depth_output_path))[0]

                try:
                    images_idx = images_output_filenames.index(sparse_depth_output_filename)
                except ValueError:
                    continue

                ground_truth_output_path = ground_truth_output_paths[depth_idx]
                images_output_path = images_output_paths[images_idx]

                train_unsupervised_images_output_paths.append(images_output_path)
                train_unsupervised_sparse_depth_output_paths.append(sparse_depth_output_path)
                train_unsupervised_ground_truth_output_paths.append(ground_truth_output_path)
                train_unsupervised_intrinsics_output_paths.extend([intrinsics_output_path] * len(images_output_paths))
        else:
            print('Processing testing {} samples in: {}'.format(n_sample, train_sequence_dirpath))

            pool_inputs = []

            for idx in range(n_sample):

                image_filename = os.path.splitext(os.path.basename(image_paths[idx]))[0]
                ground_truth_filename = os.path.splitext(os.path.basename(ground_truth_paths[idx]))[0]
                assert image_filename == ground_truth_filename

                if idx in range(w, n_sample - w):
                    pool_inputs.append((
                        image_paths[idx],
                        image_paths[idx-w],
                        image_paths[idx+w],
                        ground_truth_paths[idx],
                        sparse_depth_distro_type,
                        n_points,
                        min_points,
                        n_height,
                        n_width,
                        True))
                else:
                    pool_inputs.append((
                        image_paths[idx],
                        image_paths[idx],
                        image_paths[idx],
                        ground_truth_paths[idx],
                        sparse_depth_distro_type,
                        n_points,
                        min_points,
                        n_height,
                        n_width,
                        False))

            with mp.Pool() as pool:
                pool_results = pool.map(process_frame, pool_inputs)

                for result in pool_results:
                    image_output_path, \
                        images_output_path, \
                        sparse_depth_output_path, \
                        ground_truth_output_path = result

                    error_encountered = \
                        image_output_path == 'error' or \
                        images_output_path == 'error' or \
                        sparse_depth_output_path == 'error' or \
                        ground_truth_output_path == 'error'

                    if error_encountered:
                        continue

                    # Collect train filepaths
                    if images_output_path is not None:
                        train_unsupervised_images_output_paths.append(images_output_path)
                        train_unsupervised_sparse_depth_output_paths.append(sparse_depth_output_path)
                        train_unsupervised_ground_truth_output_paths.append(ground_truth_output_path)
                        train_unsupervised_intrinsics_output_paths.append(intrinsics_output_path)

                    train_supervised_image_output_paths.append(image_output_path)
                    train_supervised_sparse_depth_output_paths.append(sparse_depth_output_path)
                    train_supervised_ground_truth_output_paths.append(ground_truth_output_path)
                    train_supervised_intrinsics_output_paths.append(intrinsics_output_path)

    '''
    Write training output paths
    '''
    train_supervised_image_output_filepath = \
        TRAIN_SUPERVISED_IMAGE_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    train_supervised_sparse_depth_output_filepath = \
        TRAIN_SUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    train_supervised_ground_truth_output_filepath = \
        TRAIN_SUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    train_supervised_intrinsics_output_filepath = \
        TRAIN_SUPERVISED_INTRINSICS_OUTPUT_FILEPATH.format(sparse_depth_distro_type)

    train_unsupervised_images_output_filepath = \
        TRAIN_UNSUPERVISED_IMAGES_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    train_unsupervised_sparse_depth_output_filepath = \
        TRAIN_UNSUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    train_unsupervised_ground_truth_output_filepath = \
        TRAIN_UNSUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    train_unsupervised_intrinsics_output_filepath = \
        TRAIN_UNSUPERVISED_INTRINSICS_OUTPUT_FILEPATH.format(sparse_depth_distro_type)

    # Storing paths for supervised training
    print('Storing {} training supervised image file paths into: {}'.format(
        len(train_supervised_image_output_paths), train_supervised_image_output_filepath))
    data_utils.write_paths(train_supervised_image_output_filepath, train_supervised_image_output_paths)

    print('Storing {} training supervised sparse depth file paths into: {}'.format(
        len(train_supervised_sparse_depth_output_paths), train_supervised_sparse_depth_output_filepath))
    data_utils.write_paths(train_supervised_sparse_depth_output_filepath, train_supervised_sparse_depth_output_paths)

    print('Storing {} training supervised ground truth file paths into: {}'.format(
        len(train_supervised_ground_truth_output_paths), train_supervised_ground_truth_output_filepath))
    data_utils.write_paths(train_supervised_ground_truth_output_filepath, train_supervised_ground_truth_output_paths)

    print('Storing {} training supervised intrinsics file paths into: {}'.format(
        len(train_supervised_intrinsics_output_paths), train_supervised_intrinsics_output_filepath))
    data_utils.write_paths(train_supervised_intrinsics_output_filepath, train_supervised_intrinsics_output_paths)

    # Storing paths for unsupervised training
    print('Storing {} training image file paths into: {}'.format(
        len(train_unsupervised_images_output_paths), train_unsupervised_images_output_filepath))
    data_utils.write_paths(train_unsupervised_images_output_filepath, train_unsupervised_images_output_paths)

    print('Storing {} training sparse depth file paths into: {}'.format(
        len(train_unsupervised_sparse_depth_output_paths), train_unsupervised_sparse_depth_output_filepath))
    data_utils.write_paths(train_unsupervised_sparse_depth_output_filepath, train_unsupervised_sparse_depth_output_paths)

    print('Storing {} training ground truth file paths into: {}'.format(
        len(train_unsupervised_ground_truth_output_paths), train_unsupervised_ground_truth_output_filepath))
    data_utils.write_paths(train_unsupervised_ground_truth_output_filepath, train_unsupervised_ground_truth_output_paths)

    print('Storing {} training intrinsics file paths into: {}'.format(
        len(train_unsupervised_intrinsics_output_paths), train_unsupervised_intrinsics_output_filepath))
    data_utils.write_paths(train_unsupervised_intrinsics_output_filepath, train_unsupervised_intrinsics_output_paths)


def setup_dataset_scannet_testing(sparse_depth_distro_type,
                                  n_points,
                                  n_height,
                                  n_width,
                                  fast_forward):
    '''
    Fetch image and ground truth paths for testing

    Arg(s):
        sparse_depth_distro_type : bool
            sampling strategy for sparse points
        n_points : int
            number of sparse points to sample
        n_height : int
            height of image and ground truth after cropping away black edges
        n_width : int
            width of image and ground truth after cropping away black edges
        fast_forward : bool
            if set, then fast forward through sequences that have already been processed
    '''

    # Define output paths
    test_image_output_paths = []
    test_sparse_depth_output_paths = []
    test_ground_truth_output_paths = []
    test_intrinsics_output_paths = []

    test_sequence_dirpaths = natsorted(glob.glob(os.path.join(SCANNET_TEST_DIRPATH, '*/')))

    for test_sequence_dirpath in test_sequence_dirpaths:

        test_image_paths = natsorted(glob.glob(
            os.path.join(test_sequence_dirpath, 'export', 'color', '*.jpg')))
        test_ground_truth_paths = natsorted(glob.glob(
            os.path.join(test_sequence_dirpath, 'export', 'depth', '*.png')))

        n_sample = len(test_image_paths)

        assert n_sample == len(test_ground_truth_paths)

        test_intrinsics_path = \
            os.path.join(test_sequence_dirpath, 'export', 'intrinsic', 'intrinsic_color.txt')

        # Load and convert intrinsics to 3 x 3
        intrinsics = np.loadtxt(test_intrinsics_path)
        intrinsics = intrinsics[:3, :3]

        # Adjust based on resizing and cropping
        scale_factor_x = R_WIDTH / O_WIDTH
        scale_factor_y = R_HEIGHT / O_HEIGHT

        d_height = R_HEIGHT - n_height
        d_width = R_WIDTH - n_width
        offset_x = d_width // 2
        offset_y = d_height // 2

        intrinsics[0, 0] = intrinsics[0, 0] * scale_factor_x
        intrinsics[1, 1] = intrinsics[1, 1] * scale_factor_y
        intrinsics[0, 2] = intrinsics[0, 2] * scale_factor_x - offset_x
        intrinsics[1, 2] = intrinsics[1, 2] * scale_factor_x - offset_y

        test_intrinsics_output_path = test_intrinsics_path \
            .replace(os.path.join('intrinsic', 'intrinsic_color.txt'), 'intrinsics.npy')

        np.save(test_intrinsics_output_path, intrinsics)

        # Get all existing paths
        image_output_dirpath = os.path.dirname(
            test_image_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'color'), 'image'))
        sparse_depth_output_dirpath = os.path.dirname(
            test_ground_truth_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'depth'), 'sparse_depth'))
        ground_truth_output_dirpath = os.path.dirname(
            test_ground_truth_paths[0]
                .replace(SCANNET_ROOT_DIRPATH, SCANNET_DERIVED_DIRPATH)
                .replace(os.path.join('export', 'depth'), 'ground_truth'))

        image_output_paths = natsorted(glob.glob(os.path.join(image_output_dirpath, '*.jpg')))
        sparse_depth_output_paths = natsorted(glob.glob(os.path.join(sparse_depth_output_dirpath, '*.png')))
        ground_truth_output_paths = natsorted(glob.glob(os.path.join(ground_truth_output_dirpath, '*.png')))

        is_exists_output_dirpaths = \
            os.path.exists(image_output_dirpath) and \
            os.path.exists(sparse_depth_output_dirpath) and \
            os.path.exists(ground_truth_output_dirpath) and \
            len(image_output_paths) == len(sparse_depth_output_paths) and \
            len(image_output_paths) == len(ground_truth_output_paths)

        if fast_forward and is_exists_output_dirpaths:

            print('Found {} samples for testing in: {}'.format(len(image_output_paths), test_sequence_dirpath))

            # Append all testing paths
            test_image_output_paths.extend(image_output_paths)
            test_sparse_depth_output_paths.extend(sparse_depth_output_paths)
            test_ground_truth_output_paths.extend(ground_truth_output_paths)
            test_intrinsics_output_paths.extend([test_intrinsics_output_path] * len(image_output_paths))
        else:
            print('Processing testing {} samples in: {}'.format(n_sample, test_sequence_dirpath))

            # Subsample test set by factor
            test_image_paths = test_image_paths[::TEST_SET_SUBSAMPLE_FACTOR]
            test_ground_truth_paths = test_ground_truth_paths[::TEST_SET_SUBSAMPLE_FACTOR]

            pool_inputs = []

            for image_path, ground_truth_path in zip(test_image_paths, test_ground_truth_paths):
                image_filename = os.path.splitext(os.path.basename(image_path))[0]
                ground_truth_filename = os.path.splitext(os.path.basename(ground_truth_path))[0]
                assert image_filename == ground_truth_filename

                pool_inputs.append((
                    image_path,
                    image_path,
                    image_path,
                    ground_truth_path,
                    sparse_depth_distro_type,
                    n_points,
                    100,
                    n_height,
                    n_width,
                    False))

            with mp.Pool() as pool:
                pool_results = pool.map(process_frame, pool_inputs)

                for result in pool_results:
                    image_output_path, \
                        _, \
                        sparse_depth_output_path, \
                        ground_truth_output_path = result

                    error_encountered = \
                        image_output_path == 'error' or \
                        sparse_depth_output_path == 'error' or \
                        ground_truth_output_path == 'error'

                    if error_encountered:
                        continue

                    # Collect test filepaths
                    test_image_output_paths.append(image_output_path)
                    test_sparse_depth_output_paths.append(sparse_depth_output_path)
                    test_ground_truth_output_paths.append(ground_truth_output_path)
                    test_intrinsics_output_paths.append(test_intrinsics_output_path)

    '''
    Write testing output paths
    '''
    test_image_output_filepath = TEST_IMAGE_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    test_sparse_depth_output_filepath = TEST_SPARSE_DEPTH_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    test_ground_truth_output_filepath = TEST_GROUND_TRUTH_OUTPUT_FILEPATH.format(sparse_depth_distro_type)
    test_intrinsics_output_filepath = TEST_INTRINSICS_OUTPUT_FILEPATH.format(sparse_depth_distro_type)

    print('Storing {} testing image file paths into: {}'.format(
        len(test_image_output_paths),  test_image_output_filepath))
    data_utils.write_paths(test_image_output_filepath, test_image_output_paths)

    print('Storing {} testing sparse depth file paths into: {}'.format(
        len(test_sparse_depth_output_paths), test_sparse_depth_output_filepath))
    data_utils.write_paths(test_sparse_depth_output_filepath, test_sparse_depth_output_paths)

    print('Storing {} testing dense depth file paths into: {}'.format(
        len(test_ground_truth_output_paths), test_ground_truth_output_filepath))
    data_utils.write_paths(test_ground_truth_output_filepath, test_ground_truth_output_paths)

    print('Storing {} testing intrinsics file paths into: {}'.format(
        len(test_intrinsics_output_paths),  test_intrinsics_output_filepath))
    data_utils.write_paths( test_intrinsics_output_filepath, test_intrinsics_output_paths)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--sparse_depth_distro_type', type=str, default='corner')
    parser.add_argument('--n_points',                 type=int, default=N_CLUSTER)
    parser.add_argument('--min_points',               type=int, default=MIN_POINTS)
    parser.add_argument('--n_height',                 type=int, default=N_HEIGHT)
    parser.add_argument('--n_width',                  type=int, default=N_WIDTH)
    parser.add_argument('--temporal_window',          type=int, default=TEMPORAL_WINDOW)
    parser.add_argument('--fast_forward',             action='store_true')

    args = parser.parse_args()

    # Create output directories first
    dirpaths = [
        SCANNET_DERIVED_DIRPATH,
        TRAIN_REF_DIRPATH,
        VAL_REF_DIRPATH,
        TEST_REF_DIRPATH,
        TRAIN_SUPERVISED_REF_DIRPATH,
        TRAIN_UNSUPERVISED_REF_DIRPATH
    ]

    for dirpath in dirpaths:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

    # Set up dataset for training
    setup_dataset_scannet_training(
        sparse_depth_distro_type=args.sparse_depth_distro_type,
        n_points=args.n_points,
        min_points=args.min_points,
        n_height=args.n_height,
        n_width=args.n_width,
        temporal_window=args.temporal_window,
        fast_forward=args.fast_forward)

    # Set up dataset for testing
    setup_dataset_scannet_testing(
        sparse_depth_distro_type=args.sparse_depth_distro_type,
        n_points=args.n_points,
        n_height=args.n_height,
        n_width=args.n_width,
        fast_forward=args.fast_forward)
