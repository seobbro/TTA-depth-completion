import sys, os, glob, shutil, argparse, random, cv2
import multiprocessing as mp
import numpy as np
import pandas as pd
sys.path.insert(0, 'src')
import data_utils

# Set random seed
np.random.seed(42)
random.seed(42)

N_VAL_SAMPLE = 5

'''
Paths for KITTI dataset
'''
KITTI_ROOT_DIRPATH = os.path.join('data', 'kitti_depth_completion')
KITTI_TRAIN_SPARSE_DEPTH_DIRPATH = os.path.join(
    KITTI_ROOT_DIRPATH, 'train_val_split', 'sparse_depth', 'train')
# To be concatenated to sequence path
KITTI_SPARSE_DEPTH_REFPATH = os.path.join('proj_depth', 'velodyne_raw')


'''
Paths for Virtual KITTI dataset
'''
VKITTI_ROOT_DIRPATH = os.path.join('data', 'virtual_kitti')
VKITTI_TRAIN_DEPTH_REFPATH = 'vkitti_2.0.3'
# Note: we only need to use the clone directory since lighting change only affects RGB
VKITTI_TRAIN_DEPTH_DIRPATH = os.path.join(VKITTI_ROOT_DIRPATH, VKITTI_TRAIN_DEPTH_REFPATH)


'''
Output directory
'''
OUTPUT_ROOT_DIRPATH = os.path.join('data', 'virtual_kitti_kitti_derived')
TRAIN_SUPERVISED_REF_DIRPATH = os.path.join('training', 'virtual_kitti', 'supervised')
TRAIN_UNSUPERVISED_REF_DIRPATH = os.path.join('training', 'virtual_kitti', 'unsupervised')
TEST_REF_DIRPATH = os.path.join('testing', 'virtual_kitti')

# All paths for supervised training
ALL_SUPERVISED_IMAGE_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_all_image-{}.txt')
ALL_SUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_all_sparse_depth-{}.txt')
ALL_SUPERVISED_VALIDITY_MAP_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_all_validity_map-{}.txt')
ALL_SUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_all_ground_truth-{}.txt')
ALL_SUPERVISED_INTRINSICS_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_all_intrinsics-{}.txt')

# Paths for supervised training
TRAIN_SUPERVISED_IMAGE_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_train_image-{}.txt')
TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_train_sparse_depth-{}.txt')
TRAIN_SUPERVISED_VALIDITY_MAP_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_train_validity_map-{}.txt')
TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_train_ground_truth-{}.txt')
TRAIN_SUPERVISED_INTRINSICS_FILEPATH = os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'vkitti_train_intrinsics-{}.txt')

# All paths for unsupervised training
ALL_UNSUPERVISED_IMAGE_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_all_image-{}.txt')
ALL_UNSUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_all_sparse_depth-{}.txt')
ALL_UNSUPERVISED_VALIDITY_MAP_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_all_validity_map-{}.txt')
ALL_UNSUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_all_ground_truth-{}.txt')
ALL_UNSUPERVISED_INTRINSICS_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_all_intrinsics-{}.txt')

# Paths for unsupervised training
TRAIN_UNSUPERVISED_IMAGE_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_train_image-{}.txt')
TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_train_sparse_depth-{}.txt')
TRAIN_UNSUPERVISED_VALIDITY_MAP_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_train_validity_map-{}.txt')
TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_train_ground_truth-{}.txt')
TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH = os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'vkitti_train_intrinsics-{}.txt')

# Paths for testing
TEST_IMAGE_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'vkitti_test_image-{}.txt')
TEST_SPARSE_DEPTH_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'vkitti_test_sparse_depth-{}.txt')
TEST_VALIDITY_MAP_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'vkitti_test_validity_map-{}.txt')
TEST_GROUND_TRUTH_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'vkitti_test_ground_truth-{}.txt')
TEST_INTRINSICS_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'vkitti_test_intrinsics-{}.txt')

# Data split
TEST_SPLIT_SEQUENCES = [
    'Scene02'
]


def process_frame(args):
    '''
    Processes one frame

    Arg(s):
        args : tuple[str, str, str, str, list[str], list[str], bool]
            vkitti_image_path at time t=0,
            vkitti_image_path at time t=-1,
            vkitti_image_path at time t=1,
            vkitti_intrinsics,
            vkitti_ground_truth_path,
            kitti_sparse_depth_paths,
            output_dirpaths,
            save_image_triplet,
            paths_only
    Returns:
        list[str]: paths to vkitti image
        list[str]: paths to vkitti image triplets
        list[str]: paths to vkitti sparse depth
        list[str]: paths to vkitti validity map
        list[str]: paths to vkitti intrinsics matrix
        list[str]: paths to vkitti ground truth
    '''

    vkitti_image0_path, \
        vkitti_image1_path, \
        vkitti_image2_path, \
        vkitti_intrinsics, \
        vkitti_ground_truth_path, \
        kitti_sparse_depth_paths, \
        output_dirpaths, \
        save_image_triplet, \
        paths_only = args

    # Load virtual KITTI groundtruth depth
    vkitti_ground_truth = data_utils.load_depth(vkitti_ground_truth_path, multiplier=100.0)

    # Set up filename and output paths
    filename = os.path.basename(vkitti_ground_truth_path)
    filename, ext = os.path.splitext(filename)

    n_sample = len(kitti_sparse_depth_paths)

    vkitti_image_dirpath, \
        vkitti_images_dirpath, \
        vkitti_sparse_depth_dirpath, \
        vkitti_validity_map_dirpath, \
        vkitti_intrinsics_dirpath, \
        vkitti_ground_truth_dirpath = output_dirpaths

    # Output filepaths
    vkitti_image_paths = \
        [os.path.join(vkitti_image_dirpath, os.path.basename(vkitti_image0_path))] * n_sample

    imagec = None

    if save_image_triplet:
        vkitti_images_paths = \
            [os.path.join(vkitti_images_dirpath, os.path.basename(vkitti_image0_path))] * n_sample

        # Read images and concatenate together
        image0 = cv2.imread(vkitti_image0_path)
        image1 = cv2.imread(vkitti_image1_path)
        image2 = cv2.imread(vkitti_image2_path)

        imagec = np.concatenate([image1, image0, image2], axis=1)

    else:
        vkitti_images_paths = None

    # Save intrinsics and duplicate paths
    vkitti_intrinsics_path = os.path.join(
        vkitti_intrinsics_dirpath,
        os.path.splitext(os.path.basename(vkitti_image0_path))[0] + '.npy')
    np.save(vkitti_intrinsics_path, vkitti_intrinsics)

    vkitti_intrinsics_paths = [vkitti_intrinsics_path] * n_sample

    # Set up sparse depth, validity map, and ground truth paths
    vkitti_sparse_depth_paths = []
    vkitti_validity_map_paths = []
    vkitti_ground_truth_paths = \
        [os.path.join(vkitti_ground_truth_dirpath, filename + ext)] * n_sample

    # Get height and width of VKITTI image for rescaling
    n_height_vkitti, n_width_vkitti = vkitti_ground_truth.shape

    # Define vectorize function to convert list to vector for indexing
    int_vector = np.vectorize(np.int_)

    for idx, kitti_sparse_depth_path in enumerate(kitti_sparse_depth_paths):

        if not paths_only:
            vkitti_validity_map = np.zeros_like(vkitti_ground_truth, dtype=np.float32)

            # Load KITTI validity map
            kitti_sparse_depth = data_utils.load_depth(kitti_sparse_depth_path)
            kitti_validity_map = np.where(
                kitti_sparse_depth > 0.0,
                1.0,
                0.0)

            n_height_kitti, n_width_kitti = kitti_validity_map.shape

            # Get xy positions of points from lidar depth map and get validity map
            y_nonzero, x_nonzero = np.nonzero(kitti_validity_map)

            x_nonzero_indices = int_vector((x_nonzero / n_width_kitti) * n_width_vkitti)
            y_nonzero_indices = int_vector((y_nonzero / n_height_kitti) * n_height_vkitti)

            vkitti_validity_map[y_nonzero_indices, x_nonzero_indices] = 1.0

            vkitti_sparse_depth = vkitti_ground_truth * vkitti_validity_map

        # Append KITTI sequence and filename to VKIITTI filename
        kitti_path_parts = kitti_sparse_depth_path.split(os.sep)
        output_filename = filename + '-{}'.format(kitti_path_parts[-5] + '_' + kitti_path_parts[-1])

        # Store output paths
        vkitti_sparse_depth_paths.append(os.path.join(vkitti_sparse_depth_dirpath, output_filename))
        vkitti_validity_map_paths.append(os.path.join(vkitti_validity_map_dirpath, output_filename))

        # Save to as PNG to disk
        if not paths_only:
            data_utils.save_depth(vkitti_sparse_depth, vkitti_sparse_depth_paths[-1])
            data_utils.save_validity_map(kitti_validity_map, vkitti_validity_map_paths[-1])

        # Only save groundtruth depth once
        if idx == 0 and not paths_only:
            data_utils.save_depth(vkitti_ground_truth, vkitti_ground_truth_paths[-1])
            shutil.copy(vkitti_image0_path, vkitti_image_paths[-1])

            if imagec is not None:
                cv2.imwrite(vkitti_images_paths[-1], imagec)

    return (vkitti_image_paths,
            vkitti_images_paths,
            vkitti_sparse_depth_paths,
            vkitti_validity_map_paths,
            vkitti_intrinsics_paths,
            vkitti_ground_truth_paths)


parser = argparse.ArgumentParser()

parser.add_argument('--n_sample_per_image',
    type=int, default=50, help='Number of nuscenes sample to use for each Virtual KITTI image')
parser.add_argument('--paths_only',
    action='store_true', help='If set, then only produce paths')
parser.add_argument('--n_thread',
    type=int, default=10, help='Number of threads to use in parallel pool')
parser.add_argument('--temporal_window',
    type=int, default=3, help='Temporal window to use for image triplet')

args = parser.parse_args()


'''
Generate sparse, semi-dense, dense depth with validity map
'''
for dirpath in [TRAIN_SUPERVISED_REF_DIRPATH, TRAIN_UNSUPERVISED_REF_DIRPATH, TEST_REF_DIRPATH]:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# Obtain all KITTI paths
kitti_sparse_depth_paths = glob.glob(
    os.path.join(KITTI_TRAIN_SPARSE_DEPTH_DIRPATH, '*', KITTI_SPARSE_DEPTH_REFPATH, '*', '*.png'))

kitti_sparse_depth_paths = sorted(kitti_sparse_depth_paths)

# Obtain Virtual KITTI sequence directory paths
vkitti_sequence_dirpaths = sorted(glob.glob(os.path.join(VKITTI_TRAIN_DEPTH_DIRPATH, '*/')))

# Allocate lists to hold all paths
all_supervised_image_paths = []
all_supervised_sparse_depth_paths = []
all_supervised_validity_map_paths = []
all_supervised_intrinsics_paths = []
all_supervised_ground_truth_paths = []

train_supervised_image_paths = []
train_supervised_sparse_depth_paths = []
train_supervised_validity_map_paths = []
train_supervised_intrinsics_paths = []
train_supervised_ground_truth_paths = []

all_unsupervised_images_paths = []
all_unsupervised_sparse_depth_paths = []
all_unsupervised_validity_map_paths = []
all_unsupervised_intrinsics_paths = []
all_unsupervised_ground_truth_paths = []

train_unsupervised_images_paths = []
train_unsupervised_sparse_depth_paths = []
train_unsupervised_validity_map_paths = []
train_unsupervised_intrinsics_paths = []
train_unsupervised_ground_truth_paths = []

test_image_paths = []
test_sparse_depth_paths = []
test_validity_map_paths = []
test_intrinsics_paths = []
test_ground_truth_paths = []

conditions = ['clone', 'rain', 'fog', '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right']

for vkitti_condition in conditions:

    # Allocate lists to hold paths for each condition
    all_supervised_image_condition_paths = []
    all_supervised_sparse_depth_condition_paths = []
    all_supervised_validity_map_condition_paths = []
    all_supervised_intrinsics_condition_paths = []
    all_supervised_ground_truth_condition_paths = []

    train_supervised_image_condition_paths = []
    train_supervised_sparse_depth_condition_paths = []
    train_supervised_validity_map_condition_paths = []
    train_supervised_intrinsics_condition_paths = []
    train_supervised_ground_truth_condition_paths = []

    all_unsupervised_images_condition_paths = []
    all_unsupervised_sparse_depth_condition_paths = []
    all_unsupervised_validity_map_condition_paths = []
    all_unsupervised_intrinsics_condition_paths = []
    all_unsupervised_ground_truth_condition_paths = []

    train_unsupervised_images_condition_paths = []
    train_unsupervised_sparse_depth_condition_paths = []
    train_unsupervised_validity_map_condition_paths = []
    train_unsupervised_intrinsics_condition_paths = []
    train_unsupervised_ground_truth_condition_paths = []

    test_image_condition_paths = []
    test_sparse_depth_condition_paths = []
    test_validity_map_condition_paths = []
    test_intrinsics_condition_paths = []
    test_ground_truth_condition_paths = []

    for vkitti_sequence_dirpath in vkitti_sequence_dirpaths:
        print('Processing Virtual KITTI sequence: {}'.format(vkitti_sequence_dirpath))

        # Select Virtual KITTI image for sequence: data/virtual_kitti/vkitti_2.0.3_depth/Scene01/clone/frames/rgb/
        vkitti_sequence_image_dirpath = os.path.join(vkitti_sequence_dirpath, vkitti_condition, 'frames', 'rgb')

        # Select Virtual KITTI depth for sequence: data/virtual_kitti/vkitti_2.0.3_depth/Scene01/clone/frames/depth/
        vkitti_sequence_depth_dirpath = os.path.join(vkitti_sequence_dirpath, vkitti_condition, 'frames', 'depth')

        # Select Virtual KITTI intrinsics for sequence: data/virtual_kitti/vkitti_2.0.3_depth/Scene01/clone/intrinsic.txt
        vkitti_sequence_intrinsics_path = os.path.join(vkitti_sequence_dirpath, vkitti_condition, 'intrinsic.txt')

        # Read intrinsics file: frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        intrinsics = pd.read_csv(vkitti_sequence_intrinsics_path, delim_whitespace=True)

        # Construct output directory for rgb: data/virtual_kitti_derived/vkitti_2.0.3_depth/Scene01/clone/frames/rgb/
        output_sequence_image_dirpath = \
            vkitti_sequence_image_dirpath.replace(VKITTI_ROOT_DIRPATH, OUTPUT_ROOT_DIRPATH)

        # Construct output directory for depth: data/virtual_kitti_derived/vkitti_2.0.3_depth/Scene01/clone/frames/depth/
        output_sequence_depth_dirpath = \
            vkitti_sequence_depth_dirpath.replace(VKITTI_ROOT_DIRPATH, OUTPUT_ROOT_DIRPATH)

        # Construct output directory for intrinsics: data/virtual_kitti_derived/vkitti_2.0.3_depth/Scene01/clone/intrinsic
        output_sequence_intrinsics_dirpath = \
            vkitti_sequence_intrinsics_path.replace(VKITTI_ROOT_DIRPATH, OUTPUT_ROOT_DIRPATH)[:-4]

        n_output = 0

        camera_dirpaths = ['Camera_0', 'Camera_1']

        w = int(args.temporal_window // 2)

        for vkitti_camera_dirpath in camera_dirpaths:

            vkitti_sequence_image_filepaths = sorted(glob.glob(
                os.path.join(vkitti_sequence_image_dirpath, vkitti_camera_dirpath, '*.jpg')))

            vkitti_sequence_depth_filepaths = sorted(glob.glob(
                os.path.join(vkitti_sequence_depth_dirpath, vkitti_camera_dirpath, '*.png')))

            n_filepath_image = len(vkitti_sequence_image_filepaths)
            n_filepath_depth = len(vkitti_sequence_depth_filepaths)

            assert n_filepath_image == n_filepath_depth, \
                'Number of filepaths: image={} depth={}'.format(n_filepath_image, n_filepath_depth)

            # Construct output paths
            output_image_dirpath = os.path.join(
                output_sequence_image_dirpath, vkitti_camera_dirpath, 'image')
            output_images_dirpath = os.path.join(
                output_sequence_image_dirpath, vkitti_camera_dirpath, 'images')
            output_sparse_depth_dirpath = os.path.join(
                output_sequence_depth_dirpath, vkitti_camera_dirpath, 'sparse_depth')
            output_validity_map_dirpath = os.path.join(
                output_sequence_depth_dirpath, vkitti_camera_dirpath, 'validity_map')
            output_intrinsics_dirpath = os.path.join(
                output_sequence_intrinsics_dirpath, vkitti_camera_dirpath)
            output_ground_truth_dirpath = os.path.join(
                output_sequence_depth_dirpath, vkitti_camera_dirpath, 'ground_truth')

            # Get camera id
            camera_id = int(vkitti_camera_dirpath[-1])
            intrinsics_camera = intrinsics.loc[intrinsics['cameraID'] == camera_id]

            output_dirpaths = [
                output_image_dirpath,
                output_images_dirpath,
                output_sparse_depth_dirpath,
                output_validity_map_dirpath,
                output_intrinsics_dirpath,
                output_ground_truth_dirpath
            ]

            for output_dirpath in output_dirpaths:
                if not os.path.exists(output_dirpath):
                    os.makedirs(output_dirpath)

            pool_inputs = []
            n_sample = len(vkitti_sequence_image_filepaths)

            for idx in range(n_sample):
                vkitti_image0_path = vkitti_sequence_image_filepaths[idx]

                # Create intrinsics matrix using K[0,0] K[1,1] K[0,2] K[1,2]
                vkitti_intrinsics_params = intrinsics_camera.loc[intrinsics_camera['frame'] == idx]
                vkitti_intrinsics_params = np.reshape(vkitti_intrinsics_params.to_numpy(), -1)[2:]
                vkitti_intrinsics = np.eye(3)
                vkitti_intrinsics[0, 0] = vkitti_intrinsics_params[0]
                vkitti_intrinsics[1, 1] = vkitti_intrinsics_params[1]
                vkitti_intrinsics[0, 2] = vkitti_intrinsics_params[2]
                vkitti_intrinsics[1, 2] = vkitti_intrinsics_params[3]

                if idx in range(w, n_sample - w):
                    vkitti_image1_path = vkitti_sequence_image_filepaths[idx-1]
                    vkitti_image2_path = vkitti_sequence_image_filepaths[idx+1]
                    save_image_triplet = True
                else:
                    vkitti_image1_path = None
                    vkitti_image2_path = None
                    save_image_triplet = False

                vkitti_ground_truth_path = vkitti_sequence_depth_filepaths[idx]

                # Shuffle the KITTI samples and choose N of them
                kitti_subset_sparse_depth_paths = \
                    np.random.permutation(kitti_sparse_depth_paths)[0:args.n_sample_per_image]

                pool_inputs.append((
                    vkitti_image0_path,
                    vkitti_image1_path,
                    vkitti_image2_path,
                    vkitti_intrinsics,
                    vkitti_ground_truth_path,
                    kitti_subset_sparse_depth_paths,
                    output_dirpaths,
                    save_image_triplet,
                    args.paths_only))

            with mp.Pool(args.n_thread) as pool:
                pool_results = pool.map(process_frame, pool_inputs)

            for result in pool_results:
                image_paths, \
                    images_paths, \
                    sparse_depth_paths, \
                    validity_map_paths, \
                    intrinsics_paths, \
                    ground_truth_paths = result

                n_output = n_output + len(sparse_depth_paths)

                # If we have image triplet
                if images_paths is not None:
                    all_unsupervised_images_condition_paths.extend(images_paths)
                    all_unsupervised_sparse_depth_condition_paths.extend(sparse_depth_paths)
                    all_unsupervised_validity_map_condition_paths.extend(validity_map_paths)
                    all_unsupervised_intrinsics_condition_paths.extend(intrinsics_paths)
                    all_unsupervised_ground_truth_condition_paths.extend(ground_truth_paths)

                # Append all images for supervised training
                all_supervised_image_condition_paths.extend(image_paths)
                all_supervised_sparse_depth_condition_paths.extend(sparse_depth_paths)
                all_supervised_validity_map_condition_paths.extend(validity_map_paths)
                all_supervised_intrinsics_condition_paths.extend(intrinsics_paths)
                all_supervised_ground_truth_condition_paths.extend(ground_truth_paths)

                # Check if the image belong to test sequence
                is_test = False

                for seq in TEST_SPLIT_SEQUENCES:
                    if seq in image_paths[0]:
                        is_test = True
                        break

                if is_test:
                    test_image_condition_paths.extend(image_paths[0:min(N_VAL_SAMPLE, args.n_sample_per_image)])
                    test_sparse_depth_condition_paths.extend(sparse_depth_paths[0:min(N_VAL_SAMPLE, args.n_sample_per_image)])
                    test_validity_map_condition_paths.extend(validity_map_paths[0:min(N_VAL_SAMPLE, args.n_sample_per_image)])
                    test_intrinsics_condition_paths.extend(intrinsics_paths[0:min(N_VAL_SAMPLE, args.n_sample_per_image)])
                    test_ground_truth_condition_paths.extend(ground_truth_paths[0:min(N_VAL_SAMPLE, args.n_sample_per_image)])
                else:
                    if images_paths is not None:
                        train_unsupervised_images_condition_paths.extend(images_paths)
                        train_unsupervised_sparse_depth_condition_paths.extend(sparse_depth_paths)
                        train_unsupervised_validity_map_condition_paths.extend(validity_map_paths)
                        train_unsupervised_intrinsics_condition_paths.extend(intrinsics_paths)
                        train_unsupervised_ground_truth_condition_paths.extend(ground_truth_paths)

                    train_supervised_image_condition_paths.extend(image_paths)
                    train_supervised_sparse_depth_condition_paths.extend(sparse_depth_paths)
                    train_supervised_validity_map_condition_paths.extend(validity_map_paths)
                    train_supervised_intrinsics_condition_paths.extend(intrinsics_paths)
                    train_supervised_ground_truth_condition_paths.extend(ground_truth_paths)

        print('Generated {} total samples for {}'.format(n_output, vkitti_sequence_dirpath))

    # Keep track of all condition paths for sequences
    all_supervised_image_paths.extend(all_supervised_image_condition_paths)
    all_supervised_sparse_depth_paths.extend(all_supervised_sparse_depth_condition_paths)
    all_supervised_validity_map_paths.extend(all_supervised_validity_map_condition_paths)
    all_supervised_intrinsics_paths.extend(all_supervised_intrinsics_condition_paths)
    all_supervised_ground_truth_paths.extend(all_supervised_ground_truth_condition_paths)

    train_supervised_image_paths.extend(train_supervised_image_condition_paths)
    train_supervised_sparse_depth_paths.extend(train_supervised_sparse_depth_condition_paths)
    train_supervised_validity_map_paths.extend(train_supervised_validity_map_condition_paths)
    train_supervised_intrinsics_paths.extend(train_supervised_intrinsics_condition_paths)
    train_supervised_ground_truth_paths.extend(train_supervised_ground_truth_condition_paths)

    all_unsupervised_images_paths.extend(all_unsupervised_images_condition_paths)
    all_unsupervised_sparse_depth_paths.extend(all_unsupervised_sparse_depth_condition_paths)
    all_unsupervised_validity_map_paths.extend(all_unsupervised_validity_map_condition_paths)
    all_unsupervised_intrinsics_paths.extend(all_unsupervised_intrinsics_condition_paths)
    all_unsupervised_ground_truth_paths.extend(all_unsupervised_ground_truth_condition_paths)

    train_unsupervised_images_paths.extend(train_unsupervised_images_condition_paths)
    train_unsupervised_sparse_depth_paths.extend(train_unsupervised_sparse_depth_condition_paths)
    train_unsupervised_validity_map_paths.extend(train_unsupervised_validity_map_condition_paths)
    train_unsupervised_intrinsics_paths.extend(train_unsupervised_intrinsics_condition_paths)
    train_unsupervised_ground_truth_paths.extend(train_unsupervised_ground_truth_condition_paths)

    test_image_paths.extend(test_image_condition_paths)
    test_sparse_depth_paths.extend(test_sparse_depth_condition_paths)
    test_validity_map_paths.extend(test_validity_map_condition_paths)
    test_intrinsics_paths.extend(test_intrinsics_condition_paths)
    test_ground_truth_paths.extend(test_ground_truth_condition_paths)

    '''
    Write all supervised training paths for a condition to disk
    '''
    print('Writing all {} supervised training image paths to {}'.format(
        len(all_supervised_image_condition_paths),
        ALL_SUPERVISED_IMAGE_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_SUPERVISED_IMAGE_FILEPATH.format(vkitti_condition),
        all_supervised_image_condition_paths)

    print('Writing all {} supervised training sparse depth paths to {}'.format(
        len(all_supervised_sparse_depth_condition_paths),
        ALL_SUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_SUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition),
        all_supervised_sparse_depth_condition_paths)

    print('Writing all {} supervised training validity map paths to {}'.format(
        len(all_supervised_validity_map_condition_paths),
        ALL_SUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_SUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition),
        all_supervised_validity_map_condition_paths)

    print('Writing all {} supervised training intrinsics paths to {}'.format(
        len(all_supervised_intrinsics_condition_paths),
        ALL_SUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_SUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition),
        all_supervised_intrinsics_condition_paths)

    print('Writing all {} supervised training groundtruth depth paths to {}'.format(
        len(all_supervised_ground_truth_condition_paths),
        ALL_SUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_SUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition),
        all_supervised_ground_truth_condition_paths)

    '''
    Write supervised training paths for a condition to disk
    '''
    print('Writing {} supervised training image paths to {}'.format(
        len(train_supervised_image_condition_paths),
        TRAIN_SUPERVISED_IMAGE_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_SUPERVISED_IMAGE_FILEPATH.format(vkitti_condition),
        train_supervised_image_condition_paths)

    print('Writing {} supervised training sparse depth paths to {}'.format(
        len(train_supervised_sparse_depth_condition_paths),
        TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition),
        train_supervised_sparse_depth_condition_paths)

    print('Writing {} supervised training validity map paths to {}'.format(
        len(train_supervised_validity_map_condition_paths),
        TRAIN_SUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_SUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition),
        train_supervised_validity_map_condition_paths)

    print('Writing {} supervised training intrinsics paths to {}'.format(
        len(train_supervised_intrinsics_condition_paths),
        TRAIN_SUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_SUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition),
        train_supervised_intrinsics_condition_paths)

    print('Writing {} supervised training groundtruth depth paths to {}'.format(
        len(train_supervised_ground_truth_condition_paths),
        TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition),
        train_supervised_ground_truth_condition_paths)

    '''
    Write all unsupervised training paths for a condition to disk
    '''
    print('Writing all unsupervised {} training image paths to {}'.format(
        len(all_unsupervised_images_condition_paths),
        ALL_UNSUPERVISED_IMAGE_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_UNSUPERVISED_IMAGE_FILEPATH.format(vkitti_condition),
        all_unsupervised_images_condition_paths)

    print('Writing all {} unsupervised training sparse depth paths to {}'.format(
        len(all_unsupervised_sparse_depth_condition_paths),
        ALL_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition),
        all_unsupervised_sparse_depth_condition_paths)

    print('Writing all {} unsupervised training validity map paths to {}'.format(
        len(all_unsupervised_validity_map_condition_paths),
        ALL_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition),
        all_unsupervised_validity_map_condition_paths)

    print('Writing all {} unsupervised training intrinsics paths to {}'.format(
        len(all_unsupervised_intrinsics_condition_paths),
        ALL_UNSUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_UNSUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition),
        all_unsupervised_intrinsics_condition_paths)

    print('Writing all {} unsupervised training groundtruth depth paths to {}'.format(
        len(all_unsupervised_ground_truth_condition_paths),
        ALL_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        ALL_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition),
        all_unsupervised_ground_truth_condition_paths)

    '''
    Write unsupervised training paths for a condition to disk
    '''
    print('Writing {} unsupervised training image paths to {}'.format(
        len(train_unsupervised_images_condition_paths),
        TRAIN_UNSUPERVISED_IMAGE_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_IMAGE_FILEPATH.format(vkitti_condition),
        train_unsupervised_images_condition_paths)

    print('Writing {} unsupervised training sparse depth paths to {}'.format(
        len(train_unsupervised_sparse_depth_condition_paths),
        TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format(vkitti_condition),
        train_unsupervised_sparse_depth_condition_paths)

    print('Writing {} unsupervised training validity map paths to {}'.format(
        len(train_unsupervised_validity_map_condition_paths),
        TRAIN_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format(vkitti_condition),
        train_unsupervised_validity_map_condition_paths)

    print('Writing {} unsupervised training intrinsics paths to {}'.format(
        len(train_unsupervised_intrinsics_condition_paths),
        TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH.format(vkitti_condition),
        train_unsupervised_intrinsics_condition_paths)

    print('Writing {} unsupervised training groundtruth depth paths to {}'.format(
        len(train_unsupervised_ground_truth_condition_paths),
        TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format(vkitti_condition),
        train_unsupervised_ground_truth_condition_paths)

    '''
    Write testing paths for a condition to disk
    '''
    print('Writing {} testing image paths to {}'.format(
        len(test_image_condition_paths),
        TEST_IMAGE_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TEST_IMAGE_FILEPATH.format(vkitti_condition),
        test_image_condition_paths)

    print('Writing {} testing sparse depth paths to {}'.format(
        len(test_sparse_depth_condition_paths),
        TEST_SPARSE_DEPTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TEST_SPARSE_DEPTH_FILEPATH.format(vkitti_condition),
        test_sparse_depth_condition_paths)

    print('Writing {} testing validity map paths to {}'.format(
        len(test_validity_map_condition_paths),
        TEST_VALIDITY_MAP_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TEST_VALIDITY_MAP_FILEPATH.format(vkitti_condition),
        test_validity_map_condition_paths)

    print('Writing {} testing intrinsics paths to {}'.format(
        len(test_intrinsics_condition_paths),
        TEST_INTRINSICS_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TEST_INTRINSICS_FILEPATH.format(vkitti_condition),
        test_intrinsics_condition_paths)

    print('Writing {} testing groundtruth depth paths to {}'.format(
        len(test_ground_truth_condition_paths),
        TEST_GROUND_TRUTH_FILEPATH.format(vkitti_condition)))
    data_utils.write_paths(
        TEST_GROUND_TRUTH_FILEPATH.format(vkitti_condition),
        test_ground_truth_condition_paths)


'''
Write all supervised training paths to disk
'''
print('Writing all {} supervised training image paths to {}'.format(
    len(all_supervised_image_paths),
    ALL_SUPERVISED_IMAGE_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_SUPERVISED_IMAGE_FILEPATH.format('all'),
    all_supervised_image_paths)

print('Writing all {} supervised training sparse depth paths to {}'.format(
    len(all_supervised_sparse_depth_paths),
    ALL_SUPERVISED_SPARSE_DEPTH_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_SUPERVISED_SPARSE_DEPTH_FILEPATH.format('all'),
    all_supervised_sparse_depth_paths)

print('Writing all {} supervised training validity map paths to {}'.format(
    len(all_supervised_validity_map_paths),
    ALL_SUPERVISED_VALIDITY_MAP_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_SUPERVISED_VALIDITY_MAP_FILEPATH.format('all'),
    all_supervised_validity_map_paths)

print('Writing all {} supervised training intrinsics paths to {}'.format(
    len(all_supervised_intrinsics_paths),
    ALL_SUPERVISED_INTRINSICS_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_SUPERVISED_INTRINSICS_FILEPATH.format('all'),
    all_supervised_intrinsics_paths)

print('Writing all {} supervised training groundtruth depth paths to {}'.format(
    len(all_supervised_ground_truth_paths),
    ALL_SUPERVISED_GROUND_TRUTH_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_SUPERVISED_GROUND_TRUTH_FILEPATH.format('all'),
    all_supervised_ground_truth_paths)

'''
Write supervised training paths to disk
'''
print('Writing {} supervised training image paths to {}'.format(
    len(train_supervised_image_paths),
    TRAIN_SUPERVISED_IMAGE_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_SUPERVISED_IMAGE_FILEPATH.format('all'),
    train_supervised_image_paths)

print('Writing {} supervised training sparse depth paths to {}'.format(
    len(train_supervised_sparse_depth_paths),
    TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_SUPERVISED_SPARSE_DEPTH_FILEPATH.format('all'),
    train_supervised_sparse_depth_paths)

print('Writing {} supervised training validity map paths to {}'.format(
    len(train_supervised_validity_map_paths),
    TRAIN_SUPERVISED_VALIDITY_MAP_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_SUPERVISED_VALIDITY_MAP_FILEPATH.format('all'),
    train_supervised_validity_map_paths)

print('Writing {} supervised training intrinsics paths to {}'.format(
    len(train_supervised_intrinsics_paths),
    TRAIN_SUPERVISED_INTRINSICS_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_SUPERVISED_INTRINSICS_FILEPATH.format('all'),
    train_supervised_intrinsics_paths)

print('Writing {} supervised training groundtruth depth paths to {}'.format(
    len(train_supervised_ground_truth_paths),
    TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_SUPERVISED_GROUND_TRUTH_FILEPATH.format('all'),
    train_supervised_ground_truth_paths)

'''
Write all unsupervised training paths to disk
'''
print('Writing all {} unsupervised training image paths to {}'.format(
    len(all_unsupervised_images_paths),
    ALL_UNSUPERVISED_IMAGE_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_UNSUPERVISED_IMAGE_FILEPATH.format('all'),
    all_unsupervised_images_paths)

print('Writing all {} unsupervised training sparse depth paths to {}'.format(
    len(all_unsupervised_sparse_depth_paths),
    ALL_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format('all'),
    all_unsupervised_sparse_depth_paths)

print('Writing all {} unsupervised training validity map paths to {}'.format(
    len(all_unsupervised_validity_map_paths),
    ALL_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format('all'),
    all_unsupervised_validity_map_paths)

print('Writing all {} unsupervised training intrinsics paths to {}'.format(
    len(all_unsupervised_intrinsics_paths),
    ALL_UNSUPERVISED_INTRINSICS_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_UNSUPERVISED_INTRINSICS_FILEPATH.format('all'),
    all_unsupervised_intrinsics_paths)

print('Writing all {} unsupervised training groundtruth depth paths to {}'.format(
    len(all_unsupervised_ground_truth_paths),
    ALL_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format('all')))
data_utils.write_paths(
    ALL_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format('all'),
    all_unsupervised_ground_truth_paths)

'''
Write unsupervised training paths to disk
'''
print('Writing {} unsupervised training image paths to {}'.format(
    len(train_unsupervised_images_paths),
    TRAIN_UNSUPERVISED_IMAGE_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_UNSUPERVISED_IMAGE_FILEPATH.format('all'),
    train_unsupervised_images_paths)

print('Writing {} unsupervised training sparse depth paths to {}'.format(
    len(train_unsupervised_sparse_depth_paths),
    TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_UNSUPERVISED_SPARSE_DEPTH_FILEPATH.format('all'),
    train_unsupervised_sparse_depth_paths)

print('Writing {} unsupervised training validity map paths to {}'.format(
    len(train_unsupervised_validity_map_paths),
    TRAIN_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_UNSUPERVISED_VALIDITY_MAP_FILEPATH.format('all'),
    train_unsupervised_validity_map_paths)

print('Writing {} unsupervised training intrinsics paths to {}'.format(
    len(train_unsupervised_intrinsics_paths),
    TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_UNSUPERVISED_INTRINSICS_FILEPATH.format('all'),
    train_unsupervised_intrinsics_paths)

print('Writing {} unsupervised training groundtruth depth paths to {}'.format(
    len(train_unsupervised_ground_truth_paths),
    TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format('all')))
data_utils.write_paths(
    TRAIN_UNSUPERVISED_GROUND_TRUTH_FILEPATH.format('all'),
    train_unsupervised_ground_truth_paths)

'''
Write testing paths to disk
'''
print('Writing {} testing image paths to {}'.format(
    len(test_image_paths),
    TEST_IMAGE_FILEPATH.format('all')))
data_utils.write_paths(
    TEST_IMAGE_FILEPATH.format('all'),
    test_image_paths)

print('Writing {} testing sparse depth paths to {}'.format(
    len(test_sparse_depth_paths),
    TEST_SPARSE_DEPTH_FILEPATH.format('all')))
data_utils.write_paths(
    TEST_SPARSE_DEPTH_FILEPATH.format('all'),
    test_sparse_depth_paths)

print('Writing {} testing validity map paths to {}'.format(
    len(test_validity_map_paths),
    TEST_VALIDITY_MAP_FILEPATH.format('all')))
data_utils.write_paths(
    TEST_VALIDITY_MAP_FILEPATH.format('all'),
    test_validity_map_paths)

print('Writing {} testing intrinsics paths to {}'.format(
    len(test_intrinsics_paths),
    TEST_INTRINSICS_FILEPATH.format('all')))
data_utils.write_paths(
    TEST_INTRINSICS_FILEPATH.format('all'),
    test_intrinsics_paths)

print('Writing {} testing groundtruth depth paths to {}'.format(
    len(test_ground_truth_paths),
    TEST_GROUND_TRUTH_FILEPATH.format('all')))
data_utils.write_paths(
    TEST_GROUND_TRUTH_FILEPATH.format('all'),
    test_ground_truth_paths)
