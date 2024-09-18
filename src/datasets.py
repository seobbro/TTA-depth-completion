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

import torch
import numpy as np
import data_utils
import os
# import json
import random
from PIL import Image
import torchvision.transforms.functional as TF

def load_triplet_image(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t - 1
        numpy[float32] : image at t
        numpy[float32] : image at t + 1
    '''

    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    if data_format == 'CHW':
        dim = -1
    elif data_format == 'HWC':
        dim = -2
    else:
        print("data_format is wrong")
        assert()

    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=dim)
    return image1, image0, image2

def horizontal_flip(images_arr):
    '''
    Perform horizontal flip on each sample

    Arg(s):
        images_arr : list[np.array[float32]]
            list of N x C x H x W tensors
    Returns:
        list[np.array[float32]] : list of transformed N x C x H x W image tensors
    '''

    for i, image in enumerate(images_arr):
        if len(image.shape) != 3:
            raise ValueError('Can only flip C x H x W images in dataloader.')

        flipped_image = np.flip(image, axis=-1)
        images_arr[i] = flipped_image

    return images_arr

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = np.array([[0.0, 0.0, -x_start],
                                           [0.0, 0.0, -y_start],
                                           [0.0, 0.0, 0.0     ]])
        intrinsics = np.array(
            [in_ + offset_principal_point for in_ in intrinsics]
        )

        return outputs, intrinsics
    else:
        return outputs


def deterministic_crop(inputs, shape, num_crops, intrinsics=None, crop_type=['bottom']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''
    # Check the inputs: image, sparse_depth, ground_truth
    assert len(inputs) == 3
    assert intrinsics is not None

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height
    x_start = d_width // 2
    # Original code
    # if 'horizontal' in crop_type:

    #     # Select from one of the pre-defined anchored locations
    #     if 'anchored' in crop_type:
    #         # Create anchor positions
    #         crop_anchors = [
    #             0.0, 0.50, 1.0
    #         ]

    #         widths = [
    #             anchor * d_width for anchor in crop_anchors
    #         ]
    #         x_start = int(widths[np.random.randint(low=0, high=len(widths))])

    #     # Randomly select a crop location
    #     else:
    #         x_start = np.random.randint(low=0, high=d_width+1)
    unit = d_height // num_crops

    x_start_list = [i * unit for i in range(num_crops)]
    image_list = []
    sparse_depth_list = []
    ground_truth_list = []
    intrinsics_list = []

    for i in range(num_crops):
        # Crop each input into (n_height, n_width)
        y_end = y_start + n_height
        x_end = x_start_list[i] + n_width
        outputs = [
            T[:, y_start:y_end, x_start_list[i]:x_end] for T in inputs
        ]
        # Adjust intrinsics
        if intrinsics is not None:
            offset_principal_point = np.array([[0.0, 0.0, -x_start],
                                            [0.0, 0.0, -y_start],
                                            [0.0, 0.0, 0.0     ]])

            # intrinsics_temp = np.array(
            #     [in_ + offset_principal_point for in_ in intrinsics]
            # )
            intrinsics_temp = intrinsics + offset_principal_point

        image_list.append(outputs[0])
        sparse_depth_list.append(outputs[1])
        ground_truth_list.append(outputs[2])
        intrinsics_list.append(intrinsics_temp)
    images = np.stack(image_list, axis=0)
    sparse_depths = np.stack(sparse_depth_list, axis=0)
    ground_truths = np.stack(ground_truth_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)
    # print(intrinsics.shape)
    outputs = [images, sparse_depths, ground_truths]
    return outputs, intrinsics

#
#
# class DepthCompletionInferenceDataset(torch.utils.data.Dataset):
#     '''
#     Dataset for fetching:
#         (1) image
#         (2) sparse depth
#         (3) intrinsic camera calibration matrix
#     Arg(s):
#         image_paths : list[str]
#             paths to images
#         sparse_depth_paths : list[str]
#             paths to sparse depth maps
#         intrinsics_paths : list[str]
#             paths to intrinsic camera calibration matrix
#         load_image_triplets : bool
#             Whether or not inference images are stored as triplets or single
#     '''
#
#     def __init__(self,
#                  image_paths,
#                  sparse_depth_paths,
#                  intrinsics_paths,
#                  ground_truth_paths=None,
#                  load_image_triplets=False):
#
#         self.n_sample = len(image_paths)
#
#         self.image_paths = image_paths
#         self.sparse_depth_paths = sparse_depth_paths
#
#         if intrinsics_paths is not None:
#             self.intrinsics_paths = intrinsics_paths
#         else:
#             self.intrinsics_paths = [None] * self.n_sample
#
#         for paths in [sparse_depth_paths, intrinsics_paths]:
#             assert len(paths) == self.n_sample
#
#         self.is_available_ground_truth = \
#            ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])
#
#         if self.is_available_ground_truth:
#             self.ground_truth_paths = ground_truth_paths
#
#         self.data_format = 'CHW'
#         self.load_image_triplets = load_image_triplets
#
#     def __getitem__(self, index):
#
#         # Load image
#         if self.load_image_triplets:
#             _, image, _ = load_triplet_image(
#                 path=self.image_paths[index],
#                 normalize=False,
#                 data_format=self.data_format)
#         else:
#             image = data_utils.load_image(
#                 path=self.image_paths[index],
#                 normalize=False,
#                 data_format=self.data_format)
#
#         # Load sparse depth
#         sparse_depth = data_utils.load_depth(
#             path=self.sparse_depth_paths[index],
#             data_format=self.data_format)
#
#         # Load camera intrinsics
#         if self.intrinsics_paths[index] is not None:
#             intrinsics = np.load(self.intrinsics_paths[index])
#         else:
#             intrinsics = np.eye(N=3)
#
#         inputs = [
#             image,
#             sparse_depth,
#             intrinsics
#         ]
#
#         # Load ground truth if available
#         if self.is_available_ground_truth:
#             ground_truth = data_utils.load_depth(
#                 self.ground_truth_paths[index],
#                 data_format=self.data_format)
#             inputs.append(ground_truth)
#
#         # Convert to float32
#         inputs = [
#             T.astype(np.float32)
#             for T in inputs
#         ]
#
#         # Return image, sparse_depth, intrinsics, and if available, ground_truth
#         return inputs
#
#     def __len__(self):
#         return self.n_sample


class DepthCompletionInferenceDataset_(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.ground_truth_paths = ground_truth_paths
        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):

        # Load image
        image = data_utils.load_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)
        # Load sparse depth

        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load ground truth
        ground_truth, validity_map_gt = data_utils.load_depth_with_validity_map(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])

        # Convert to float32
        image, sparse_depth, ground_truth, intrinsics, validity_map_gt = [
            T.astype(np.float32)
            for T in [image, sparse_depth, ground_truth, intrinsics, validity_map_gt]
        ]
        return image, sparse_depth, ground_truth, intrinsics, validity_map_gt

    def __len__(self):
        return self.n_sample


class DepthCompletionTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix
    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 ground_truth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths, ground_truth_paths]:
            assert len(paths) == self.n_sample

        if random_crop_shape is not None:
            self.random_crop_shape = random_crop_shape
            self.random_crop_type = ['horizontal', 'vertical']
        elif 'kitti' in image_paths[0] and random_crop_shape is None:
            self.random_crop_shape = (320, 768)
            self.random_crop_type = ['horizontal', 'vertical']
        elif 'void' in image_paths[0] and random_crop_shape is None:
            self.random_crop_shape = (448, 576)
            self.random_crop_type = ['horizontal', 'vertical']
        else:
            print('dataset should be updated')
            exit()
        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.ground_truth_paths = ground_truth_paths
        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):
        # Load image
        # if self.load_image_triplets:
        #     _, image, _ = load_triplet_image(
        #         path=self.image_paths[index],
        #         normalize=False,
        #         data_format=self.data_format)
        # else:
        #     image = data_utils.load_image(
        #         path=self.image_paths[index],
        #         normalize=False,
        #         data_format=self.data_format)
        _, image, _ = load_triplet_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)
        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        ground_truth, validity_map = data_utils.load_depth_with_validity_map(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])
        # [image, sparse_depth, ground_truth, validity_map], [intrinsics] = random_crop(
        #     inputs=[image, sparse_depth, ground_truth, validity_map],
        #     shape=self.random_crop_shape,
        #     intrinsics=[intrinsics],
        #     crop_type=self.random_crop_type)
        # print(image.shape, sparse_depth.shape,ground_truth.shape, validity_map.shape)

        # Convert to float32
        image, sparse_depth, ground_truth, validity_map, intrinsics = [
            T.astype(np.float32)
            for T in [image, sparse_depth, ground_truth, validity_map, intrinsics]
        ]

        return image, sparse_depth, ground_truth, intrinsics, validity_map

    def __len__(self):
        return self.n_sample

class DepthCompletionTrainingDataset_transforms(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 ground_truth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths, ground_truth_paths]:
            assert len(paths) == self.n_sample

        if 'kitti' in image_paths[0]:
            self.top_crop = 100
            self.height,  self.width = 240, 1216
        else:
            self.top_crop = 0

        # if random_crop_shape is not None:
        #     self.random_crop_shape = random_crop_shape
        #     self.random_crop_type = ['horizontal', 'vertical']
        # elif 'kitti' in image_paths[0] and random_crop_shape is None:
        #     self.random_crop_shape = (320, 768)
        #     self.random_crop_type = ['horizontal', 'vertical']
        # elif 'void' in image_paths[0] and random_crop_shape is None:
        #     self.random_crop_shape = (448, 576)
        #     self.random_crop_type = ['horizontal', 'vertical']
        # else:
        #     print('dataset should be updated')
        #     exit()

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        if intrinsics_paths is not None:
            self.intrinsics_paths = intrinsics_paths
        else:
            self.intrinsics_paths = [None] * self.n_sample

        self.ground_truth_paths = ground_truth_paths

        self.data_format = 'CHW'

        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):
        # Load image
        # if self.load_image_triplets:
        #     _, image, _ = load_triplet_image(
        #         path=self.image_paths[index],
        #         normalize=False,
        #         data_format=self.data_format)
        # else:
        #     image = data_utils.load_image(
        #         path=self.image_paths[index],
        #         normalize=False,
        #         data_format=self.data_format)

        image, sparse_depth, ground_truth, intrinsics = self._load_data(index)

        # _, image, _ = load_triplet_image(
        #     path=self.image_paths[index],
        #     normalize=False,
        #     data_format=self.data_format)
        # # Load sparse depth
        # sparse_depth = data_utils.load_depth(
        #     path=self.sparse_depth_paths[index],
        #     data_format=self.data_format)
        #
        # ground_truth, validity_map = data_utils.load_depth_with_validity_map(
        #     path=self.ground_truth_paths[index],
        #     data_format=self.data_format)
        #
        # # Load camera intrinsics
        # intrinsics = np.load(self.intrinsics_paths[index])

        # Augmentations
        width, height = image.size

        if self.top_crop > 0:
            image = TF.crop(image, self.top_crop, 0,
                          height - self.top_crop, width)
            sparse_depth = TF.crop(sparse_depth, self.top_crop, 0,
                            height - self.top_crop, width)
            ground_truth = TF.crop(ground_truth, self.top_crop, 0,
                         height - self.top_crop, width)
            intrinsics[1, 2] = intrinsics[1, 2] - self.top_crop

        _scale = np.random.uniform(1.0, 1.5)
        scale = np.int(height * _scale)
        degree = np.random.uniform(-5.0, 5.0)
        flip = np.random.uniform(0.0, 1.0)

        # Horizontal flip
        if flip > 0.5:
            image = TF.hflip(image)
            sparse_depth = TF.hflip(sparse_depth)
            ground_truth = TF.hflip(ground_truth)
            intrinsics[0, 2] = width - intrinsics[0, 2]
        # Rotation
        image = TF.rotate(image, angle=degree, resample=Image.BICUBIC)
        sparse_depth = TF.rotate(sparse_depth, angle=degree, resample=Image.NEAREST)
        ground_truth = TF.rotate(ground_truth, angle=degree, resample=Image.NEAREST)

        # Color jitter
        brightness = np.random.uniform(0.6, 1.4)
        contrast = np.random.uniform(0.6, 1.4)
        saturation = np.random.uniform(0.6, 1.4)

        image = TF.adjust_brightness(image, brightness)
        image = TF.adjust_contrast(image, contrast)
        image = TF.adjust_saturation(image, saturation)

        # Resize
        image = TF.resize(image, scale, Image.BICUBIC)
        sparse_depth = TF.resize(sparse_depth, scale, Image.NEAREST)
        ground_truth = TF.resize(ground_truth, scale, Image.NEAREST)

        intrinsics[0, 0] = intrinsics[0, 0] * _scale
        intrinsics[1, 1] = intrinsics[1, 1] * _scale
        intrinsics[0, 2] = intrinsics[0, 2] * _scale
        intrinsics[1, 2] = intrinsics[1, 2] * _scale

        width, height = image.size

        assert self.height <= height and self.width <= width, \
            "patch size is larger than the input size"

        h_start = random.randint(0, height - self.height)
        w_start = random.randint(0, width - self.width)

        image = TF.crop(image, h_start, w_start, self.height, self.width)
        sparse_depth = TF.crop(sparse_depth, h_start, w_start, self.height, self.width)
        ground_truth = TF.crop(ground_truth, h_start, w_start, self.height, self.width)

        intrinsics[0, 2] = intrinsics[0, 2] - w_start
        intrinsics[1, 2] = intrinsics[1, 2] - h_start

        image = TF.to_tensor(image)
        image = TF.normalize(
            image,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225), inplace=True)

        sparse_depth = TF.to_tensor(np.array(sparse_depth))
        sparse_depth = sparse_depth / _scale

        ground_truth = TF.to_tensor(np.array(ground_truth))
        ground_truth = ground_truth / _scale
        # [image, sparse_depth, ground_truth, validity_map], [intrinsics] = random_crop(
        #     inputs=[image, sparse_depth, ground_truth, validity_map],
        #     shape=self.random_crop_shape,
        #     intrinsics=[intrinsics],
        #     crop_type=self.random_crop_type)
        # print(image.shape, sparse_depth.shape,ground_truth.shape, validity_map.shape)

        # Convert to float32
        # image, sparse_depth, ground_truth, intrinsics = [
        #     T.astype(np.float32)
        #     for T in [image, sparse_depth, ground_truth,intrinsics]
        # ]

        return image, sparse_depth, ground_truth, intrinsics

    def __len__(self):
        return self.n_sample

    def _load_data(self, index, normalize=True, data_format='HWC'):
        '''
        Loads an RGB image

        Arg(s):
            path : str
                path to RGB image
            normalize : bool
                if set, then normalize image between [0, 1]
            data_format : str
                'CHW', or 'HWC'
        Returns:
            numpy[float32] : H x W x C or C x H x W image
        '''

        # image = Image.open(self.image_paths[index])
        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format='HWC')

        # load_triplet_image(self.image_paths[index], normalize=False, data_format='HWC')

        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        ground_truth = data_utils.load_depth(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])

        image = Image.fromarray(np.squeeze(image).astype('float32'), mode='RGB')
        sparse_depth = Image.fromarray(np.squeeze(sparse_depth).astype('float32'), mode='F')
        ground_truth = Image.fromarray(np.squeeze(ground_truth).astype('float32'), mode='F')

        w1, h1 = image.size
        w2, h2 = sparse_depth.size
        w3, h3 = ground_truth.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3
        #
        # # Load image
        # if self.training:
        #     = paths
        #     image = Image.open(path).convert('RGB')
        #
        # # Convert to numpy
        # image = np.asarray(image, np.float32)
        #
        # if data_format == 'HWC':
        #     pass
        # elif data_format == 'CHW':
        #     image = np.transpose(image, (2, 0, 1))
        # else:
        #     raise ValueError('Unsupported data format: {}'.format(data_format))
        #
        # # Normalize
        # image = image / 255.0 if normalize else image
        return image, sparse_depth, ground_truth, intrinsics

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


class DepthCompletionTrainingDataset_with_Distillation(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 ground_truth_paths,
                 intrinsics_paths,
                 output_paths=None,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        for paths in [sparse_depth_paths, intrinsics_paths, ground_truth_paths]:
            assert len(paths) == self.n_sample

        if random_crop_shape is not None:
            self.random_crop_shape = random_crop_shape
            self.random_crop_type = ['horizontal', 'vertical']
        elif 'kitti' in image_paths[0] and random_crop_shape is None:
            self.random_crop_shape = (320, 768)
            self.random_crop_type = ['horizontal', 'vertical']
        elif 'void' in image_paths[0] and random_crop_shape is None:
            self.random_crop_shape = (448, 576)
            self.random_crop_type = ['horizontal', 'vertical']
        else:
            print('dataset should be updated')
            exit()
        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths
        self.ground_truth_paths = ground_truth_paths
        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):
        # Load image
        # if self.load_image_triplets:
        #     _, image, _ = load_triplet_image(
        #         path=self.image_paths[index],
        #         normalize=False,
        #         data_format=self.data_format)
        # else:
        #     image = data_utils.load_image(
        #         path=self.image_paths[index],
        #         normalize=False,
        #         data_format=self.data_format)
        _, image, _ = load_triplet_image(
            path=self.image_paths[index],
            normalize=False,
            data_format=self.data_format)
        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        ground_truth, validity_map = data_utils.load_depth_with_validity_map(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index])
        [image, sparse_depth, ground_truth, validity_map], [intrinsics] = random_crop(
            inputs=[image, sparse_depth, ground_truth, validity_map],
            shape=self.random_crop_shape,
            intrinsics=[intrinsics],
            crop_type=self.random_crop_type)
        # print(image.shape, sparse_depth.shape,ground_truth.shape, validity_map.shape)

        # Convert to float32
        image, sparse_depth, ground_truth, validity_map, intrinsics = [
            T.astype(np.float32)
            for T in [image, sparse_depth, ground_truth, validity_map, intrinsics]
        ]

        return image, sparse_depth, ground_truth, intrinsics, validity_map

    def __len__(self):
        return self.n_sample


class MonitoredDistillationTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image at t
        (2) left camera image at t - 1
        (3) left camera image at t + 1
        (4) if stereo is available, stereo camera image
        (5) sparse depth map at t
        (6) teacher output from ensemble at t
        (7) if stereo is available, intrinsic camera calibration matrix
        (8) if stereo is available, focal length and baseline

    Arg(s):
        image0_paths : list[str]
            paths to left camera images
        image1_paths : list[str]
            paths to right camera images
        sparse_depth0_paths : list[str]
            paths to left camera sparse depth maps
        sparse_depth1_paths : list[str]
            paths to right camera sparse depth maps
        ground_truth0_paths : list[str]
            paths to left camera ground truth depth maps
        ground_truth1_paths : list[str]
            paths to right camera ground truth depth maps
        ensemble_teacher_output0_paths : list[list[str]]
            list of lists of paths to left camera teacher output for ensemble
        ensemble_teacher_output1_paths : list[list[str]]
            list of lists of paths to right camera teacher output for ensemble
        intrinsics0_paths : list[str]
            paths to intrinsic left camera calibration matrix
        intrinsics1_paths : list[str]
            paths to intrinsic right camera calibration matrix
        focal_length_baseline0_paths : list[str]
            paths to focal length and baseline for left camera
        focal_length_baseline1_paths : list[str]
            paths to focal length and baseline for right camera
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        random_swap : bool
            Whether to perform random swapping as data augmentation
    '''

    def __init__(self,
                 image0_paths,
                 image1_paths,
                 sparse_depth0_paths,
                 sparse_depth1_paths,
                 ground_truth0_paths,
                 ground_truth1_paths,
                 ensemble_teacher_output0_paths,
                 ensemble_teacher_output1_paths,
                 intrinsics0_paths,
                 intrinsics1_paths,
                 focal_length_baseline0_paths,
                 focal_length_baseline1_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 random_swap=False):

        self.n_sample = len(image0_paths)

        # Make sure that all paths in stereo stream is present
        self.stereo_available = \
            image1_paths is not None and \
            sparse_depth1_paths is not None and \
            ground_truth1_paths is not None and \
            intrinsics1_paths is not None and \
            focal_length_baseline0_paths is not None and \
            focal_length_baseline1_paths is not None and \
            ensemble_teacher_output1_paths is not None and \
            None not in image1_paths and \
            None not in sparse_depth1_paths and \
            None not in ground_truth1_paths and \
            None not in intrinsics1_paths and \
            None not in focal_length_baseline0_paths and \
            None not in focal_length_baseline1_paths and \
            None not in ensemble_teacher_output1_paths

        # If it is missing then populate them with None
        if not self.stereo_available:
            image1_paths = [None] * self.n_sample
            sparse_depth1_paths = [None] * self.n_sample
            ground_truth1_paths = [None] * self.n_sample
            intrinsics1_paths = [None] * self.n_sample
            focal_length_baseline0_paths = [None] * self.n_sample
            focal_length_baseline1_paths = [None] * self.n_sample
            ensemble_teacher_output1_paths = \
                [[None] * self.n_sample] * len(ensemble_teacher_output0_paths)

        input_paths = [
            image1_paths,
            sparse_depth0_paths,
            sparse_depth1_paths,
            ground_truth0_paths,
            ground_truth1_paths,
            intrinsics0_paths,
            intrinsics1_paths,
            focal_length_baseline0_paths,
            focal_length_baseline1_paths
        ]

        input_paths = input_paths + \
            ensemble_teacher_output0_paths + \
            ensemble_teacher_output1_paths

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.image0_paths = image0_paths
        self.image1_paths = image1_paths

        self.sparse_depth0_paths = sparse_depth0_paths
        self.sparse_depth1_paths = sparse_depth1_paths

        self.ground_truth0_paths = ground_truth0_paths
        self.ground_truth1_paths = ground_truth1_paths

        self.intrinsics0_paths = intrinsics0_paths
        self.intrinsics1_paths = intrinsics1_paths

        self.focal_length_baseline0_paths = focal_length_baseline0_paths
        self.focal_length_baseline1_paths = focal_length_baseline1_paths

        self.ensemble_teacher_output0_paths = ensemble_teacher_output0_paths
        self.ensemble_teacher_output1_paths = ensemble_teacher_output1_paths

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.do_random_swap = random_swap and self.stereo_available

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Swap and flip a stereo video stream
        do_swap = True if self.do_random_swap and np.random.uniform() < 0.5 else False

        if do_swap:
            # Swap paths for 0 and 1 indices
            image0_path = self.image1_paths[index]
            sparse_depth0_path = self.sparse_depth1_paths[index]
            ground_truth0_path = self.ground_truth1_paths[index]
            ensemble_teacher_output0_paths = self.ensemble_teacher_output1_paths
            intrinsics0_path = self.intrinsics1_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline1_paths[index]
            image3_path = self.image0_paths[index]
        else:
            # Keep paths consistent
            image0_path = self.image0_paths[index]
            sparse_depth0_path = self.sparse_depth0_paths[index]
            ground_truth0_path = self.ground_truth0_paths[index]
            ensemble_teacher_output0_paths = self.ensemble_teacher_output0_paths
            intrinsics0_path = self.intrinsics0_paths[index]
            focal_length_baseline0_path = self.focal_length_baseline0_paths[index]

            image3_path = self.image1_paths[index]

        # Load images at times: t-1, t, t+1
        image1, image0, image2 = load_triplet_image(
            path=image0_path,
            normalize=False,
            data_format=self.data_format)

        # Load sparse depth map at time t
        sparse_depth0 = data_utils.load_depth(
            path=sparse_depth0_path,
            data_format=self.data_format)

        # Load ground_truth map at time t
        ground_truth0 = data_utils.load_depth(
            path=ground_truth0_path,
            data_format=self.data_format)

        # Load teacher output from ensemble
        teacher_output0 = []

        for paths in ensemble_teacher_output0_paths:
            teacher_output0.append(
                data_utils.load_depth(
                    path=paths[index],
                    data_format=self.data_format))

        teacher_output0 = np.concatenate(teacher_output0, axis=0)

        # Load camera intrinsics
        intrinsics0 = np.load(intrinsics0_path)

        # Load stereo pair for image0
        if self.stereo_available:
            _, image3, _ = load_triplet_image(
                path=image3_path,
                normalize=False,
                data_format=self.data_format)

            # Load camera intrinsics
            focal_length_baseline0 = np.load(focal_length_baseline0_path)
        else:
            image3 = image0.copy()
            focal_length_baseline0 = np.array([0, 0])

        inputs = [
            image0,
            image1,
            image2,
            image3,
            sparse_depth0,
            ground_truth0,
            teacher_output0,
        ]

        # If we swapped L and R, also need to horizontally flip images
        if do_swap:
            inputs = horizontal_flip(inputs)

        # Crop input images and depth maps and adjust intrinsics
        if self.do_random_crop:
            inputs, [intrinsics0] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics0],
                crop_type=self.random_crop_type)

        # Convert inputs to float32
        inputs = inputs + [intrinsics0, focal_length_baseline0]

        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        return inputs

    def __len__(self):
        return self.n_sample


'''
Merging
'''

class DepthCompletionSupervisedTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image (N x 3 x H x W)
        (2) sparse depth (N x 1 x H x W)
        (3) ground truth (N x 2 x H x W)
        (4) camera intrinsics (N x 3 x 3)

    Arg(s):
        image_paths : list[str]
            paths to camera images
        sparse_depth_paths : list[str]
            paths to camera sparse depth maps
        ground_truth_paths : list[str]
            list of paths to ground truth depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 ground_truth_paths,
                 intrinsics_paths,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        if intrinsics_paths is not None:
            self.intrinsics_paths = intrinsics_paths
        else:
            self.intrinsics_paths = [None] * self.n_sample

        self.ground_truth_paths = ground_truth_paths

        input_paths = [
            self.sparse_depth_paths,
            self.intrinsics_paths,
            self.ground_truth_paths,
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.load_image_triplets = load_image_triplets

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_triplet_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth map
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load ground truth depth map
        ground_truth_depth = data_utils.load_depth(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index])
        else:
            intrinsics = np.eye(N=3)

        # Sanity checks with shape assertions
        spatial_dims = image.shape[1:]
        assert sparse_depth.shape[1:] == spatial_dims
        assert ground_truth_depth.shape[1:] == spatial_dims

        inputs = [
            image,
            sparse_depth,
            ground_truth_depth
        ]

        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # # Crop input images and depth maps and adjust intrinsics

        # image, sparse_depth, ground_truth_depth = inputs
        # _scale = np.random.uniform(1.0, 1.5)

        # width, height = image.size

        # scale = np.int(height * _scale)

        # image = TF.resize(image, scale, Image.BICUBIC)
        # sparse_depth = TF.resize(sparse_depth, scale, Image.NEAREST)
        # ground_truth_depth = TF.resize(ground_truth_depth, scale, Image.NEAREST)
        # TF.crop(image, self.random_crop_shape)

        # random_crop
        # Add intrinsics to inputs
        inputs.append(intrinsics)
        # Convert inputs to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]
        # del inputs
        # return np.array(image), np.array(sparse_depth), np.array(ground_truth_depth), np.array(intrinsics)
        return inputs

    def __len__(self):
        return self.n_sample

class DepthCompletionSupervisedTrainingDataset_ConCat(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) camera image (N x 3 x H x W)
        (2) sparse depth (N x 1 x H x W)
        (3) ground truth (N x 2 x H x W)
        (4) camera intrinsics (N x 3 x 3)

    Arg(s):
        image_paths : list[str]
            paths to camera images
        sparse_depth_paths : list[str]
            paths to camera sparse depth maps
        ground_truth_paths : list[str]
            list of paths to ground truth depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        random_crop_shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 ground_truth_paths,
                 intrinsics_paths,
                 inner_iter=1,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        if intrinsics_paths is not None:
            self.intrinsics_paths = intrinsics_paths
        else:
            self.intrinsics_paths = [None] * self.n_sample

        self.ground_truth_paths = ground_truth_paths

        input_paths = [
            self.sparse_depth_paths,
            self.intrinsics_paths,
            self.ground_truth_paths,
        ]

        for paths in input_paths:
            assert len(paths) == self.n_sample

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.load_image_triplets = load_image_triplets

        self.data_format = 'CHW'

        self.inner_iter = inner_iter

    def __getitem__(self, index):
        # Load image
        if self.load_image_triplets:
            _, image, _ = load_triplet_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth map
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load ground truth depth map
        ground_truth_depth = data_utils.load_depth(
            path=self.ground_truth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index])
        else:
            intrinsics = np.eye(N=3)

        # Sanity checks with shape assertions
        spatial_dims = image.shape[1:]

        assert sparse_depth.shape[1:] == spatial_dims
        assert ground_truth_depth.shape[1:] == spatial_dims

        inputs = [
            image,
            sparse_depth,
            ground_truth_depth
        ]
        # print(image.shape)

        input_list, intrinsic = deterministic_crop(inputs=inputs,
            shape=self.random_crop_shape,
            num_crops=self.inner_iter,
            intrinsics=intrinsics,
            crop_type=self.random_crop_type)
        [image, sparse_depth, ground_truth] = input_list
        # images_list = []
        # sparse_depths_list = []
        # ground_truths_list = []
        # intrinsics_list = []
        # for _ in range(self.inner_iter):
        #     # Crop input images and depth maps and adjust intrinsics
        #     if self.do_random_crop:
        #         inputs, intrinsics = random_crop(
        #             inputs=inputs,
        #             shape=self.random_crop_shape,
        #             intrinsics=intrinsics,
        #             crop_type=self.random_crop_type)

        #     # Add intrinsics to inputs
        #     inputs.append(intrinsics)

        #     # Convert inputs to float32
        #     inputs = [
        #         T.astype(np.float32)
        #         for T in inputs
        #     ]
        #     images_list.append(inputs[0])
        #     sparse_depths_list.append(inputs[1])
        #     ground_truths_list.append(inputs[2])
        #     intrinsics_list.append(inputs[3])

        inputs = [image, sparse_depth, ground_truth]
        inputs.append(intrinsic)
        # print(image.shape, sparse_depth.shape, ground_truth.shape, intrinsic.shape)
        return inputs

    def __len__(self):
        return self.n_sample

class DepthCompletionInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        if intrinsics_paths is not None:
            self.intrinsics_paths = intrinsics_paths
        else:
            self.intrinsics_paths = [None] * self.n_sample

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths

        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_triplet_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index], allow_pickle=True)
        else:
            intrinsics = np.eye(N=3)

        inputs = [
            image,
            sparse_depth,
            intrinsics
        ]

        # Load ground truth if available
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return image, sparse_depth, intrinsics, and if available, ground_truth
        del inputs
        if self.is_available_ground_truth:
            # return inputs
            return image.astype(np.float32), sparse_depth.astype(np.float32), intrinsics.astype(np.float32), ground_truth.astype(np.float32)
        else:
            return image.astype(np.float32), sparse_depth.astype(np.float32), intrinsics.astype(np.float32)

    def __len__(self):
        return self.n_sample

class DepthCompletionInferenceDataset_Adapt(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) intrinsic camera calibration matrix

    Arg(s):
        image_paths : list[str]
            paths to images
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to intrinsic camera calibration matrix
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        load_image_triplets : bool
            Whether or not inference images are stored as triplets or single
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths=None,
                 random_crop_shape=None,
                 random_crop_type=None,
                 load_image_triplets=False):

        self.n_sample = len(image_paths)

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths

        if intrinsics_paths is not None:
            self.intrinsics_paths = intrinsics_paths
        else:
            self.intrinsics_paths = [None] * self.n_sample

        for paths in [sparse_depth_paths, intrinsics_paths]:
            assert len(paths) == self.n_sample

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths

        self.data_format = 'CHW'
        self.load_image_triplets = load_image_triplets

        self.random_crop_shape = random_crop_shape
        self.random_crop_type = random_crop_type

        self.do_random_crop = \
            random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_triplet_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                path=self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load sparse depth
        sparse_depth = data_utils.load_depth(
            path=self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index], allow_pickle=True)
        else:
            intrinsics = np.eye(N=3)

        spatial_dims = image.shape[1:]
        assert sparse_depth.shape[1:] == spatial_dims

        inputs = [
            image,
            sparse_depth,
        ]
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            assert ground_truth.shape[1:] == spatial_dims
            inputs.append(ground_truth)

        if self.do_random_crop:
            inputs, [intrinsics] = random_crop(
                inputs=inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        # Load ground truth if available
        inputs.append(intrinsics)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return image, sparse_depth, intrinsics, and if available, ground_truth
        return inputs

    def __len__(self):
        return self.n_sample
