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

import os, sys, argparse
from data_utils import inpainting
sys.path.insert(0, os.path.join('external_src', 'NLSPN'))
sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src'))
sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src', 'model'))
from nlspnmodel_adapt import NLSPNModel_Adapt as NLSPNBaseModel
import torch
import torch.nn as nn
import gc
class NLSPNModel_Adapt(object):
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
                 max_depth=100.0,
                 inpainting=False,
                 use_pretrained=False,
                 dataset_name=None,
                 from_scratch=False,
                 offset=False):

        # Settings to reproduce NLSPN numbers on KITTI
        if dataset_name in ['waymo', 'nuscenes']:
            n_prop_time = 18
        else:
            n_prop_time = 18
        args = argparse.Namespace(
            affinity='TGASS',
            affinity_gamma=0.5,
            conf_prop=True,
            from_scratch=from_scratch,
            legacy=offset,
            lr=0.001,
            max_depth=max_depth,
            network='resnet34',
            preserve_input=True,
            prop_kernel=3,
            prop_time=n_prop_time,  # For Nuscenes I set 24 for now.
            test_only=True)

        # Instantiate depth completion model
        self.model = NLSPNBaseModel(args)
        self.use_pretrained = use_pretrained
        self.max_depth = max_depth

        # Move to device
        self.device = device
        self.to(self.device)
        self.mean_var_dict = {}
        self.mean_var_dict['length'] = 0
        self.mean_var_dict['mean_dep'] = []
        self.mean_var_dict['cov_dep'] = []
        self.mean_var_dict['mean_proj'] = []
        self.mean_var_dict['cov_proj'] = []
        self.max_prototypes = 100000
        self.glob_mean = None
        self.adapt = False

    def forward(self,
                image,
                sparse_depth,
                intrinsics=None,
                crop_mask=None,
                loss_type=None):
        # TODO: Needs doc string
        '''
        Forwards inputs through the network

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
            sparse_depth : tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
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
        output_depth = self.model(
            image=image,
            sparse_depth=sparse_depth,
            crop_mask=crop_mask,
            loss_type=loss_type)
        # Fill in any holes with inpainting
        if not self.model.training and 'head' not in loss_type:
            output_depth = output_depth.detach().clone().cpu().numpy()
            output_depth = inpainting(output_depth)
            output_depth = torch.from_numpy(output_depth).to(self.device)
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
                     l1_weight=1.0,
                     l2_weight=1.0,
                     loss_type='pretrain'):
        '''
        Compute loss as NLSPN does: 1.0 * L1 + 1.0 * L2

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

        return self.nlspn_loss(
            output_depth=output_depth,
            ground_truth=ground_truth)

    def nlspn_loss(self,
                   output_depth,
                   ground_truth,
                   l1_weight=1.0,
                   l2_weight=1.0,):
        '''
        Compute loss as NLSPN does: 1.0 * L1 + 1.0 * L2

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
        # They clamp their predictions
        output_depth = torch.clamp(output_depth, min=0, max=self.max_depth)
        ground_truth = torch.clamp(ground_truth, min=0, max=self.max_depth)
        mask = (ground_truth > 0.0001).type_as(output_depth).detach()

        # Obtain valid values
        # Compute individual losses
        # l1 loss
        d = torch.abs(output_depth - ground_truth) * mask
        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])
        loss_l1 = d / (num_valid + 1e-8)
        loss_l1 = loss_l1.sum()

        d2 = torch.pow(output_depth - ground_truth, 2) * mask
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
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                parameters.append(param)

        parameters = torch.nn.ParameterList(parameters)

        return parameters

    def prepare_parameters(self, mode=''):
        '''
        Return the self-supervised MLP parameters updated in preprare stage
        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''
        self.model._prepare_head(mode=mode)
        parameters = []
        names = []

        if 'init' in mode:
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or \
                    isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                    if 'meta' not in nm:
                        m.requires_grad_(False)
        elif 'head' in mode:
            for np, p in self.model.named_parameters():
                if ('proj' in np or 'pred' in np) and ('_t' not in np):
                    parameters.append(p)
            return parameters
        elif 'selfsup' in mode:
            if 'meta' in mode:
                for np, p in self.model.named_parameters():
                    if 'meta' in np or (('proj' in np or 'pred' in np) and '_t' not in np):
                        parameters.append(p)
            else:
                for np, p in self.model.named_parameters():
                    if ('proj' in np or 'pred' in np) and ('_t' not in np):
                        parameters.append(p)
        elif 'meta' in mode:
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
                else:
                    p.requires_grad_(False)
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

        elif mode == 'meta':
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
                    names.append(p)
                    p.requires_grad_(True)
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)):
                    m.requires_grad_(False)
                    m.track_running_stats = False

        elif mode == 'meta_fix':
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
                    names.append(p)
            for nm, m in self.model.named_modules():
                if isinstance(m, (nn.modules.batchnorm.BatchNorm2d, torch.nn.modules.batchnorm.SyncBatchNorm)):
                    m.track_running_stats = False

        elif mode == 'meta_bn':
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    p.requires_grad_(True)
                    parameters.append(p)
                    names.append(p)
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias'] and 'pred' not in np and 'proj' not in np:  # weight is scale, bias is shift
                            parameters.append(p)
                            names.append(f"{nm}.{np}")

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
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                for np, p in m.named_parameters():
                    if 'meta' not in nm:
                        m.eval()

    def train_prepare(self):
        '''
        Sets model to training mode
        '''
        self.model.train()
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                if ('proj' not in nm and 'pred' not in nm) or '_t' in nm:
                    m.eval()

    def init_bn_stats(self):
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                if 'prop_layer' not in nm:
                    m.track_running_stats = True
                    m.requires_grad_(True)
                    m.train()
                else:
                    m.requires_grad_(False)
        for np, p in self.model.named_parameters():
            if 'prop_layer' not in np:
                p.requires_grad_(True)
            elif 'conv_offset' not in np and 'aff_scale_const' not in np:
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)

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

        if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
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
