import os, sys, argparse, json
import torch
from torchvision.models import detection
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from setup_dataset_nuscenes import get_train_val_test_split_scene_ids

sys.path.insert(0, 'src')
sys.path.append(os.getcwd())
import src.data_utils as data_utils


MAX_TRAIN_SCENES = 850
MAX_TEST_SCENES = 150


"""
python3 -W ignore setup/save_detections.py --nuscenes_data_root_dirpath /media/staging/common/datasets/nuscenes --nuscenes_data_derived_dirpath /media/staging/common/datasets/nuscenes_derived_v2
"""


'''
Data split filepaths
'''
DATA_SPLIT_DIRPATH = os.path.join('setup', 'nuscenes')
TRAIN_DATA_SPLIT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, 'train_scene_ids.txt')
VAL_DATA_SPLIT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, 'val_scene_ids.txt')
TEST_DATA_SPLIT_FILEPATH = os.path.join(DATA_SPLIT_DIRPATH, 'test_scene_ids.txt')


'''
Set up input arguments
'''
parser = argparse.ArgumentParser()


parser.add_argument('--nuscenes_data_root_dirpath',
    type=str, required=True, help='Path to nuscenes dataset')
parser.add_argument('--nuscenes_data_derived_dirpath',
    type=str, required=True, help='Path to derived dataset')
parser.add_argument('--batch_size',
    type=int, default=2, help='Size of batch to process for detection model')
parser.add_argument('--moving_object_class_ids',
    nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9], help='List of moving object class ids according to COCO 2017 dataset')
parser.add_argument('--threshold_score',
    type=float, default=0.35, help='Threshold of detection score to accept bounding box as detection')

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


def get_IoU(box1, box2):
    '''
    Returns intersection over union between two bounding boxes

    Arg(s):
        box1 : tuple[float]
            bounding box
        box2 : tuple[float32]
            bounding box
    Returns:
        float : intersection over union score
    '''

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    topx = max(xmin1, xmin2)
    topy = max(ymin1, ymin2)

    bottomx = min(xmax1, xmax2)
    bottomy = min(ymax1, ymax2)

    if bottomx > topx and bottomy > topy:
        inter_area = (bottomx - topx) * (bottomy - topy)
        union_area = \
            (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - inter_area
        return (inter_area / union_area)
    else:
        return 0

def get_moving_object_bboxes(images,
                             object_detection_model,
                             moving_object_class_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                             threshold_score=0.35):
    '''
    Return bounding boxes for moving objects in image

    Arg(s):
        images : numpy[float32]
            list of H x W x 3 images
        object_detection_model : torch.nn.Module
            object detection model instance i.e. DETR, retinanet_resnet50_fpn
        moving_object_class_ids : list[int]
            list of moving object class ids according to COCO 2017 dataset
        threshold_score : float
            score threshold to accept detection
    Returns:
        list[tuple[float]] : list of bounding box tuples
    '''

    # Convert list of H x W x 3 images to 3 x H x W tensors
    # torchvision.model expects a list of tensors, each of shape [C, H, W] and
    # will return list of dictionary of torch.Tensors
    images = [
        torch.permute(torch.from_numpy(image), (2, 0, 1)).cuda() for image in images
    ]

    boxes_moving = []

    # Forward through the model
    with torch.no_grad():
        bbox_detections = object_detection_model(images)

    for bbox_detection in bbox_detections:

        # Extract all bounding boxes with confidence higher than 0.35
        # RetinaNet already performs NMS
        # By default implementation NMS should sort in decreasing order
        boxes = []
        for i in range(0, len(bbox_detection['boxes'])):

            idx = int(bbox_detection['labels'][i])

            if idx in moving_object_class_ids and bbox_detection['scores'][i] > threshold_score:
                box = bbox_detection['boxes'][i].detach().cpu().numpy().astype('int')
                boxes.append(box.tolist())

        # Filter out boxes that have overlap based on IOU
        filtered_boxes = []

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                box1 = boxes[i]
                box2 = boxes[j]

                if get_IoU(box1, box2) < 0.5:
                    if box1 not in filtered_boxes:
                        filtered_boxes.append(box1)
                    if box2 not in filtered_boxes:
                        filtered_boxes.append(box2)

        boxes_moving.append(filtered_boxes)

    return boxes_moving

def detect_moving_objects(nusc,
                          current_sample_token,
                          object_detection_model,
                          batch_size,
                          moving_object_class_ids,
                          threshold_score):
    '''
    Arg(s):
        nusc : Object
            nuScenes data object
        current_sample_token : str
            token to access sensor data in nuscenes sample_data object
        batch_size : int
            size of batch to process for detection model
        moving_object_class_ids : list[int]
            list of moving object class ids
        threshold_score',
            threshold of detection score to accept bounding box as detection
    Returns:
        dict[str, tuple[float]] : dictionary of tokens to bounding boxes
    '''

    # Get the sample
    current_sample = nusc.get('sample', current_sample_token)

    # Get camera sample for non-keyframe
    next_camera_sample_nonkf = nusc.get('sample_data', current_sample['data']['CAM_FRONT'])

    # List of images and tokens being processed
    images = []
    tokens = []
    bboxes = {}

    # Get token for sample
    next_camera_nonkf_token = next_camera_sample_nonkf['token']

    # Create path to to non-key frame image file
    next_camera_sample_nonkf_path = os.path.join(
        nusc.dataroot,
        next_camera_sample_nonkf['filename'])

    # Load image for sample
    next_camera_nonkf_image = data_utils.load_image(
        next_camera_sample_nonkf_path,
        normalize=True,
        data_format='HWC')

    # Batch images together
    images.append(next_camera_nonkf_image)
    tokens.append(next_camera_nonkf_token)

    # While there is still more samples
    while next_camera_sample_nonkf['next'] != '':

        # Move to next sample
        next_camera_sample_nonkf = nusc.get('sample_data', next_camera_sample_nonkf['next'])

        # Get token for sample
        next_camera_nonkf_token = next_camera_sample_nonkf['token']

        # Create path to to non-key frame image file
        next_camera_sample_nonkf_path = os.path.join(
            nusc.dataroot,
            next_camera_sample_nonkf['filename'])

        # Load image for sample
        next_camera_nonkf_image = data_utils.load_image(
            next_camera_sample_nonkf_path,
            normalize=True,
            data_format='HWC')

        # Batch images together
        images.append(next_camera_nonkf_image)
        tokens.append(next_camera_nonkf_token)

        # Process each batch
        if len(images) == batch_size:
            # Detect moving objects
            next_camera_nonkf_detections = get_moving_object_bboxes(
                images,
                object_detection_model=object_detection_model,
                moving_object_class_ids=moving_object_class_ids,
                threshold_score=threshold_score)

            # Store moving object bounding boxes
            for token, boxes in zip(tokens, next_camera_nonkf_detections):
                bboxes[token] = boxes

            # Empty out buffer
            images = []
            tokens = []

    # In case we do not fill the batch size
    if len(images) > 0:
        # Detect moving objects
        next_camera_nonkf_detections = get_moving_object_bboxes(
            images,
            object_detection_model=object_detection_model,
            moving_object_class_ids=moving_object_class_ids,
            threshold_score=threshold_score)

        # Store moving object bounding boxes
        for token, boxes in zip(tokens, next_camera_nonkf_detections):
            bboxes[token] = boxes

    return bboxes

def write_to_json(data, json_path):
    '''
    Saves data to json

    Arg(s):
        data : Object
            data to be written to json
    '''

    # Create output directory if not exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    json_object = json.dumps(data)
    f = open(json_path, 'w')
    f.write(json_object)
    f.close()

def process_scene(_args):
    '''
    Processes one scene and saves bounding boxes to json file

    Arg(s):
        _args : tuple[str, str, str, torchvision.model, int, list[int], float]
            scene_id,
            first_sample_token,
            object_detection_model,
            batch_size,
            moving_object_class_ids,
            threshold_score
    '''

    scene_id, \
        first_sample_token, \
        object_detection_model, \
        batch_size, \
        moving_object_class_ids, \
        threshold_score = _args

    print('Processing {}'.format(scene_id))

    # Create output directory
    output_dirpath = os.path.join(args.nuscenes_data_derived_dirpath, 'sample_data_detection')
    os.makedirs(output_dirpath, exist_ok=True)

    bboxes = {}

    # Detect bounding boxes in all non-keyframe samples
    bboxes = detect_moving_objects(
        nusc=nusc,
        current_sample_token=first_sample_token,
        object_detection_model=object_detection_model,
        batch_size=batch_size,
        moving_object_class_ids=moving_object_class_ids,
        threshold_score=threshold_score)

    json_output_path = os.path.join(output_dirpath, scene_id + '.json')
    write_to_json(bboxes, json_output_path)

    print('Finished {} samples in {}'.format(len(bboxes), scene_id))


if __name__ == '__main__':

    train_ids, val_ids, test_ids = get_train_val_test_split_scene_ids(
        TRAIN_DATA_SPLIT_FILEPATH,
        VAL_DATA_SPLIT_FILEPATH,
        TEST_DATA_SPLIT_FILEPATH)

    scene_ids = train_ids + val_ids + test_ids
    n_train = len(train_ids)
    n_val = len(val_ids)
    n_test = len(test_ids)

    print('Total Scenes to process: {}'.format(len(scene_ids)))
    print('Training: {}  Validation: {}  Testing: {}'.format(n_train, n_val, n_test))

    # Instantiate object detection model
    object_detection_model = detection.retinanet_resnet50_fpn(pretrained=True).cuda()
    object_detection_model.eval()

    # Add all tasks for processing each scene to pool inputs
    for idx in range(0, MAX_TRAIN_SCENES + MAX_TEST_SCENES):

        # Depending on trainval or test set, we will choose a different explorer
        if idx < MAX_TRAIN_SCENES:
            nusc = nusc_explorer_trainval.nusc
        else:
            idx = idx - MAX_TRAIN_SCENES
            nusc = nusc_explorer_test.nusc

        current_scene = nusc.scene[idx]
        scene_id = current_scene['name']

        first_sample_token = current_scene['first_sample_token']

        inputs = [
            scene_id,
            first_sample_token,
            object_detection_model,
            args.batch_size,
            args.moving_object_class_ids,
            args.threshold_score
        ]

        # Detects moving objects and saves as json
        process_scene(inputs)
