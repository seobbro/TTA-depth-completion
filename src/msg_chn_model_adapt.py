src/msg_chn_model_adapt.pyimport torch
import os, sys
import loss_utils
sys.path.insert(0, os.path.join('external_src', 'MSG_CHN'))
sys.path.insert(0, os.path.join('external_src', 'MSG_CHN', 'workspace', 'exp_msg_chn'))
from network_exp_msg_chn_adapt import network_adapt
from loss_utils import smoothness_loss_func
import torch
import torch.nn as nn

class MsgChnModel_Adapt(object):
    '''
    Class for interfacing with MSGCHN model

    Arg(s):
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        device : torch.device
            device to run model on
    '''

    def __init__(self, max_predict_depth=5.0, inpainting=False, device=torch.device('cuda')):
        # Initialize model
        self.model = network_adapt(inpainting)

        self.max_predict_depth = max_predict_depth

        # Move to device
        self.device = device
        self.to(self.device)

    def forward(self, image, sparse_depth, intrinsics=None, crop_mask=None, loss_type='pretrain'):
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
            loss_type == 'pretrain'
                torch.Tensor[float32] N x 1 x H x W dense depth map
            loss_type == 'prepare'
                torch.Tensor[float32] N x (H * W) x C prediction embedding
                torch.Tensor[float32].detach() N x (H * W) x C projection embedding
        '''

        # Assume that training mode provides the same crop size of the image
        if self.model.training:
            if 'adapt' in loss_type:
                n_height, n_width = image.shape[-2:]

                do_padding = False

                # Pad to width and height such that it is divisible by 16
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

                output_depth, pred, proj = self.model.forward(image=image, sparse_depth=sparse_depth, crop_mask=crop_mask, loss_type=loss_type)

                if do_padding:
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

                return output_depth, pred, proj
            else:
                return self.model.forward(image=image, sparse_depth=sparse_depth, crop_mask=crop_mask, loss_type=loss_type)

        # For eval, we assume the different input sizes
        else:
            n_height, n_width = image.shape[-2:]

            do_padding = False

            # Pad to width and height such that it is divisible by 16
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

            output_depth = self.model.forward(image=image, sparse_depth=sparse_depth, crop_mask=crop_mask, loss_type=loss_type)
            if 'meta' not in loss_type or 'selfsup' not in loss_type:
                output_depth = output_depth[0]
            if do_padding:
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

    def compute_loss(self,
                     input_rgb=None,
                     output_depth=None,
                     validity_map=None,
                     ground_truth=None,
                     l1_weight=1.0,
                     l2_weight=1.0,
                     loss_type='pretrain'):
        '''
        Compute L2 (MSE) Loss

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground_truth depth
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        return self.msg_chn_loss(output_depth, ground_truth, validity_map)

    def msg_chn_loss(self,
                     output_depth,
                     ground_truth,
                     validity_map):
        '''
        Training the model with supervised objective
        Input:
            output_depth: torch.Tensor[float32]
                B x 1 x H x W dense depth prediction from the model
            ground_truth: torch.Tensor[float32]
                B x 1 x H x W dense ground truth
            validity_map: torch.Tensor[float32]
                B x 1 x H x W validity map of ground truth

        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''
        # Clamp ground truth values
        w_scale0 = 1.0
        w_scale1 = 0.0
        w_scale2 = 0.0
                
        ground_truth = torch.clamp(
            ground_truth,
            min=0.0,
            max=self.max_predict_depth)

        validity_map = torch.where(ground_truth > 0,
            torch.ones_like(ground_truth),
            ground_truth)
        
        # Compute loss
        loss = w_scale0 * loss_utils.l2_loss(src=output_depth[0],tgt=ground_truth,w=validity_map) + \
            w_scale1 * loss_utils.l2_loss(src=output_depth[1],tgt=ground_truth,w=validity_map) + \
            w_scale2 * loss_utils.l2_loss(src=output_depth[2],tgt=ground_truth,w=validity_map)

        loss_info = {
            'loss': loss
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''

        parameters = list(self.model.parameters())
        return parameters

    def bn_adapt_parameters(self):
        print('MSG_CHN does not contain bn')
        raise NotImplementedError

    def _prepare_head(self, mode=''):
        '''
        Initialize the self-supervised MLP heads
        '''
        self.model._prepare_head(mode)

    def prepare_parameters(self, mode=''):
        '''
        Return the self-supervised MLP parameters updated in preprare stage
        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''
        parameters = []
        names = []
        self.model._prepare_head(mode=mode)

        if 'head' in mode:
            self.model._prepare_head(mode=mode)

            for np, p in self.model.named_parameters():
                if ('proj' in np or 'pred' in np) and ('_t' not in np):
                    parameters.append(p)
                    names.append(np)
            return parameters

        if 'selfsup' in mode:
            if 'meta' in mode:
                for np, p in self.model.named_parameters():
                    if 'meta' in np or ('proj' in np or 'pred' in np and '_t' not in np):
                        parameters.append(p)
                        names.append(np)
                    else:
                        p.requires_grad_(False)
            else:
                for np, p in self.model.named_parameters():
                    if 'proj' in np or 'pred' in np:
                        parameters.append(p)
                        names.append(np)
                    else:
                        p.requires_grad_(False)

        elif 'meta' in mode:
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
                    names.append(np)
                else:
                    p.requires_grad_(False)
        else:
            self.model._prepare_head()
            parameters = []
            names = []
            for np, p in self.model.named_parameters():
                if 'proj' in np or 'pred' in np:
                    parameters.append(p)
                    names.append(p)
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
        elif mode == 'cotta':
            for nm, m in self.model.named_modules():
                if isinstance(m, nn.modules.batchnorm.BatchNorm2d) or isinstance(m, torch.nn.modules.batchnorm.SyncBatchNorm):
                    m.requires_grad_(True)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            parameters.append(p)
                            names.append(f"{nm}.{np}")
                else:
                    m.requires_grad_(True)
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            parameters.append(p)
                            names.append(f"{nm}.{np}")


        elif mode == 'encoder':
            for np, p in self.model.named_parameters():
                if 'proj' not in np and 'pred' not in np and 'conv' in np:
                    if 'conv1_dep' in np:
                        pass
                    else:
                        parameters.append(p)
                        names.append(np)
        elif mode == 'rgb':
            for np, p in self.model.named_parameters():
                if 'rgb' in np:
                    parameters.append(p)
                    names.append(np)
        elif mode == 'rgbd_conv':
            for np, p in self.model.named_parameters():
                if 'rgb_process_layer' in np or 'meta' in np:
                    parameters.append(p)

        elif mode == 'meta':
            for np, p in self.model.named_parameters():
                if 'meta' in np:
                    parameters.append(p)
                    names.append(p)

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

    def train_prepare(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    
    def train_meta(self):
        '''
        Sets model to training mode
        '''

        self.model.train()
    
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

        self.device = device
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

    def restore_model(self, restore_path, optimizer=None):
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
            self.model.module.load_state_dict(checkpoint_dict['net'])
        else:
            self.model.load_state_dict(checkpoint_dict['net'])

        # Load optimizer if it's not None:
        if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if 'train_step' in checkpoint_dict.keys():
            train_step = checkpoint_dict['train_step']
        else:
            train_step = 0
        
        return optimizer, train_step

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
