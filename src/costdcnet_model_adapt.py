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

import os, sys, argparse
sys.path.insert(0, os.path.join('external_src', 'costdcnet'))
sys.path.insert(0, os.path.join('external_src', 'costdcnet', 'models'))
from CostDCNet_adapt import CostDCNet as CostDCNetBaseModel
import torch
import torch.nn as nn
import MinkowskiEngine as ME


class CostDCNetModel_Adapt(object):
    '''
    Class for interfacing with NLSPN model

    Arg(s):
        device : torch.device
            device to run model on
        max_depth : float
            value to clamp ground truths to in computing loss
        use_pretrained : bool
            if set, then configure using legacy settings
    '''
    def __init__(self,
                 device=torch.device('cuda'),
                 max_depth=10.0,
                 ):

        # Settings to reproduce NLSPN numbers on KITTI
        args = argparse.Namespace(
            time=False,
            res=16,  # for NYU v2
            up_scale=4,  # for NYU v2
            max=max_depth,
            device=device)
        self.args = args

        # Instantiate depth completion model
        self.model = CostDCNetBaseModel(args)

        # Move to device
        self.device = device
        self.to(self.device)

    def forward(self,
                image,
                sparse_depth,
                intrinsics=None,
                crop_mask=None,
                loss_type=None):
        '''
        Forwards inputs through the network

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
            sparse_depth : tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : tensor[float32]
                N x 3 x 3 Camera intrinsics
            loss_type : forward types
                {'pretrain' : get output depth,
                 'prepare' : get embedding for self-supervised learning,
                 'adapt' : get both output depth and embeddings for self-supervised learning}
        Returns:
            loss_type == 'pretrain':
                torch.Tensor[float32] N x 1 x H x W dense depth map
            loss_type == 'prepare':
                torch.Tensor[float32] N x (H * W) x C prediction embedding
                torch.Tensor[float32].detach() N x (H * W) x C projection embedding
            loss_type == 'adapt':
                torch.Tensor[float32] N x 1 x H x W dense depth map
                torch.Tensor[float32] N x (H * W) x C prediction embedding
                torch.Tensor[float32].detach() N x (H * W) x C projection embedding
        '''
        batch_size, _, og_height, og_width = image.shape
        if not self.model.training:
            image, sparse_depth = self.transform_inputs(image, sparse_depth)

        image, sparse_depth, intrinsics = \
            self.pad_inputs(image, sparse_depth, intrinsics)

        output_depth = self.model(
            image=image,
            sparse_depth=sparse_depth,
            crop_mask=crop_mask,
            loss_type=loss_type)

        if 'head' in loss_type:
            pass
        else:
            if isinstance(output_depth, list):
                output_depth[0] = self.recover_inputs(output_depth[0], og_height, og_width)
            else:
                output_depth = self.recover_inputs(output_depth, og_height, og_width)

        return output_depth

    def transform_inputs(self, image, sparse_depth):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        return image, sparse_depth

    def pad_inputs(self, image, sparse_depth, intrinsics):

        n_height, n_width = image.shape[-2:]

        do_padding = False
        # Pad the images and expand at 0-th dimension to get batch
        if n_height % 16 != 0:
            times = n_height // 16
            padding_top = (times + 1) * 16 - n_height
            do_padding = True
        else:
            padding_top = 0

        if n_width % 16 != 0:
            times = n_width // 16
            padding_right = (times + 1) * 16 - n_width
            do_padding = True
        else:
            padding_right = 0

        if do_padding:
            # Pad the images and expand at 0-th dimension to get batch
            image0 = torch.nn.functional.pad(
                image,
                (0, padding_right, padding_top, 0, 0, 0),
                mode='constant',
                value=0)

            sparse_depth0 = torch.nn.functional.pad(
                sparse_depth,
                (0, padding_right, padding_top, 0, 0, 0),
                mode='constant',
                value=0)

            image1 = torch.nn.functional.pad(
                image,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            sparse_depth1 = torch.nn.functional.pad(
                sparse_depth,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            image = torch.cat([image0, image1], dim=0)
            sparse_depth = torch.cat([sparse_depth0, sparse_depth1], dim=0)
            intrinsics = torch.cat([intrinsics, intrinsics], dim=0)

        return image, sparse_depth, intrinsics

    def recover_inputs(self, output_depth, n_height, n_width):
        height, width = output_depth.size()[-2:]
        do_padding = False if (n_height == height and n_width == width) else True
        if do_padding:
            padding_top = height - n_height
            padding_right = width - n_width
            output0, output1 = torch.chunk(output_depth, chunks=2, dim=0)
            if padding_right == 0 :
                output0 = output0[:, :, padding_top:, :]
                output1 = output1[:, :, :-padding_top, :]
            elif padding_top == 0:
                output0 = output0[:, :, :, :-padding_right]
                output1 = output1[:, :, :, padding_right:]
            else:
                output0 = output0[:, :, padding_top:, :-padding_right]
                output1 = output1[:, :, :-padding_top, padding_right:]

            output_depth = torch.cat([
                torch.unsqueeze(output0, dim=1),
                torch.unsqueeze(output1, dim=1)],
                dim=1)

            output_depth = torch.mean(output_depth, dim=1, keepdim=False)

        return output_depth

    def _prepare_head(self, mode):
        '''
        Initialize the self-supervised MLP heads
        '''
        return self.model._prepare_head(mode=mode)

    def get_offset(self):
        '''
        Get offset values
        '''
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            self.model.module.get_offset()
        else:
            self.model.get_offset()

    def compute_loss(self,
                     input_rgb=None,
                     output_depth=None,
                     validity_map=None,
                     ground_truth=None,
                     dataset_name='void',
                     loss_type='pretrain'):
        '''

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
            l1_weight : float
                weight of l1 loss
            l2_weight : float
                weight of l2 loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        if dataset_name == 'void':
            l1_weight=1.0
            l2_weight=0.0
        elif dataset_name in ['kitti', 'waymo']:
            l1_weight=1.0
            l2_weight=1.0
        m = validity_map
        num_valid = torch.sum(m)
        d = torch.abs(ground_truth - output_depth) * m
        d = torch.sum(d, dim=[1, 2, 3])
        loss_l1 = d / (num_valid + 1e-8)
        loss_l1 = loss_l1.sum()

        d2 = torch.pow(ground_truth - output_depth, 2) * m
        d2 = torch.sum(d2, dim=[1, 2, 3])
        loss_l2 = d2 / (num_valid + 1e-8)
        loss_l2 = loss_l2.sum()
        loss = l1_weight * loss_l1 + l2_weight * loss_l2

        # Store loss info
        loss_info = {
            'loss_l1': loss_l1.detach().item(),
            'loss_l2': loss_l2.detach().item(),
            'loss': loss.detach().item()
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = []
        parameters = torch.nn.ParameterList(self.model.parameters())

        return parameters

    def prepare_parameters(self, mode=''):
        '''
        Return the self-supervised MLP parameters updated in preprare stage
        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''
        self.model._prepare_head(mode=mode)
        parameters = []
        if 'init' in mode:
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                    m.track_running_stats = False
        elif 'head' in mode:
            for np, p in self.model.named_parameters():
                if ('proj' in np or 'pred' in np) and ('_t' not in np):
                    parameters.append(p)
            return parameters

        elif 'selfsup' in mode:
            if 'meta' in mode:
                for np, p in self.model.named_parameters():
                    if 'meta' in np:
                        parameters.append(p)
                    elif ('proj' in np or 'pred' in np) and ('_t' not in np):
                        parameters.append(p)
            else:
                for np, p in self.model.named_parameters():
                    if ('proj' in np or 'pred' in np) and ('_t' not in np):
                        parameters.append(p)
        else:
            raise NotImplementedError

        parameters = torch.nn.ParameterList(parameters)
        return parameters

    def adapt_parameters(self, mode=None):
        '''
        Returns the list of adapt parameters in the model

        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''
        parameters = []
        names = []
        if mode is None:
            for np, p in self.model.named_parameters():
                if 'proj' not in np and 'pred' not in np and 'matching' not in np:
                    parameters.append(p)
                    names.append(np)

        elif mode == 'meta_rgb':
            for np, p in self.model.named_parameters():
                if 'meta' in np or 'fe1_rgb' in np:
                    parameters.append(p)
                    names.append(p)
                elif 'conv' in np:
                    if 'conv1_dep' in np:
                        parameters.append(p)
                        names.append(np)
                        # pass
                    else:
                        parameters.append(p)
                        names.append(np)

        elif mode == 'meta_bn':
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    p.requires_grad_(True)
                    parameters.append(p)
                    names.append(p)
                else:
                    p.requires_grad_(False)
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or \
                        isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm) or \
                        isinstance(m, ME.MinkowskiSyncBatchNorm):
                    if ('pred' not in nm or 'proj' not in nm):
                        m.requires_grad_(True)
                        m.track_running_stats = False
                        m.running_mean = None
                        m.running_var = None
                        for np, p in m.named_parameters():
                            if np in ['weight', 'bias']:  # weight is scale, bias is shift
                                p.requires_grad_(True)
                                parameters.append(p)
                                names.append(f"{nm}.{np}")

        elif mode == 'rgbd_meanval':
            for np, p in self.model.named_parameters():
                if (('meta' in np) and 'rgb' in np) or ('conv' in np and 'conv1' not in np) or 'prop_layer' in np:
                    parameters.append(p)
                    names.append(p)

        elif mode == 'rgbd_meanval_depth':
            for np, p in self.model.named_parameters():
                if 'dep' in np and 'meta' in np:
                    parameters.append(p)
                    names.append(p)

        parameters = torch.nn.ParameterList(parameters)
        return parameters

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def train_meta(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

        # freeze bn layers
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and 'meta' not in nm:  # weight is scale, bias is shift
                        p.requires_grad_(False)
                        # p.track_running_stats(False)
                        m.eval()

    def train_prepare(self):
        '''
        Sets model to training mode
        '''
        self.model.train()
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                if ('proj' not in nm and 'pred' not in nm) or '_t' in nm:
                    m.track_running_stats = False
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            p.requires_grad_(False)
                            m.eval()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model = torch.nn.DataParallel(self.model)

    def set_device(self, rank):
        self.model.module.set_device(rank)

    def distributed_data_parallel(self, rank):
        '''
        Allows multi-gpu split along batch
        '''
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)

    def restore_model(self, restore_path, optimizer=None, learning_schedule=None, learning_rates=None, n_step_per_epoch=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer if optimizer is passed in
        '''
        checkpoint_dict = torch.load(restore_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint_dict['net'])
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.load_state_dict(checkpoint_dict['net'])
        else:
            self.model.load_state_dict(checkpoint_dict['net'])

        if 'meanvar' in checkpoint_dict.keys():
            for k in checkpoint_dict['meanvar'].keys():
                if k != 'length':
                    checkpoint_dict['meanvar'][k] = checkpoint_dict['meanvar'][k].cuda()
            self.model.glob_mean = checkpoint_dict['meanvar']

            self.mean_var_dict = checkpoint_dict['meanvar']

        if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if 'train_step' in checkpoint_dict.keys():
            train_step = checkpoint_dict['train_step']
            return optimizer, train_step
        else:
            return optimizer

    def save_model(self, checkpoint_path, step, optimizer, meanvar=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint = {
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            checkpoint = {
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'net': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        if meanvar is not None:
            checkpoint['meanvar'] = meanvar
        torch.save(checkpoint, checkpoint_path)

    def convert_syncbn(self, apex):
        '''
        Convert BN layers to SyncBN layers.
        SyncBN merge the BN layer weights in every backward step.
        '''
        if apex:
            apex.parallel.convert_syncbn_model(self.model)
        else:
            from torch.nn import SyncBatchNorm
            SyncBatchNorm.convert_sync_batchnorm(self.model)
