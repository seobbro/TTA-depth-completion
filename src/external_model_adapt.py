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
import torch
import torchvision
import log_utils
import torch.nn.functional as F
from loss_utils import smoothness_loss_func, sparse_depth_consistency_loss_func, robustness_loss_func


# TODO: Implement cropping in external_model_adapt.py, not in tta_main.py
# from loss_utils import patchfy

class ExternalModel_Adapt(object):
    '''
    Wrapper class for all external depth completion models

    Arg(s):
        model_name : str
            depth completion model to use
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        use_pretrained : bool
            if set, then load pretrained for NLSPN
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 min_predict_depth,
                 max_predict_depth,
                 max_input_depth=None,
                 offset=False,
                 from_scratch=False,
                 dataset_name=None,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device
        self.max_predict_depth = max_predict_depth
        self.max_input_depth = max_input_depth
        if model_name == 'nlspn':
            from nlspn_model_adapt import NLSPNModel_Adapt
            self.model = NLSPNModel_Adapt(
                device=device,
                max_depth=max_predict_depth,
                offset=offset,
                dataset_name=dataset_name,
                from_scratch=from_scratch)

        elif model_name == 'msg_chn':
            from msg_chn_model_adapt import MsgChnModel_Adapt
            self.model = MsgChnModel_Adapt(
                device=device,
                max_predict_depth=max_predict_depth)

        elif 'costdcnet' in model_name:
            from costdcnet_model_adapt import CostDCNetModel_Adapt
            self.model = CostDCNetModel_Adapt(device=device, max_depth=max_predict_depth)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(model_name))

    def forward(self,
                image,
                sparse_depth,
                crop_mask=None,
                intrinsics=None,
                loss_type='pretrain'):
        '''
        Forwards inputs through network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''
        if loss_type == 'get_time':
            return [self.model.model.module.total_time, self.model.model.module.train_time, self.model.model.module.eval_time]
        if self.max_input_depth is not None:
            if isinstance(sparse_depth, list):
                sparse_depth[0] = torch.clamp(sparse_depth[0], 0, self.max_input_depth)
                sparse_depth[1] = torch.clamp(sparse_depth[1], 0, self.max_input_depth)
            else:
                sparse_depth = torch.clamp(sparse_depth, 0, self.max_input_depth)

        return self.model.forward(image=image,
                                  sparse_depth=sparse_depth,
                                  intrinsics=intrinsics,
                                  crop_mask=crop_mask,
                                  loss_type=loss_type)

    def get_mean_val(self, mean_only=False, adapt=False):
        return self.model.get_mean_val(mean_only=mean_only, adapt=adapt)

    def compute_loss(self,
                     input_rgb,
                     output_depth=None,
                     output_depth_ref=None,
                     ground_truth=None,
                     intrinsincs=None,
                     sparse_depth=None,
                     reference_depth=None,
                     validity_map=None,
                     embedding=None,
                     reference=None,
                     epoch=0,
                     max_input_depth=None,
                     max_predict_depth=100.0,
                     w_loss_sparse_depth=0.0,
                     w_loss_smoothness=0.0,
                     w_loss_robust=1.0,
                     w_loss_cos=1.0,
                     dataset_name=None,
                     loss_type='pretrain'):
        '''
        Call the model's compute loss function

        Currently only supports supervised methods (ENet, PENet, MSGCHN, NLSPN, RGB_guidance_uncertainty)
        Unsupervised methods have various more complex losses that is best trained through
        their repository

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 1 x H x W ground_truth depth with only valid values
        Returns:
            float : loss averaged over the batch
        '''

        if 'prepare' in loss_type:  # and 'meta' not in loss_type:
            return self.prepare_loss(
                embedding=embedding,
                reference=reference)

        if 'cotta' in loss_type:
            loss_adapt, loss_info = self.adapt_loss(
                input_rgb=input_rgb,
                output_depth=output_depth,
                sparse_depth=sparse_depth[0],
                validity_map=validity_map,
                embedding=embedding,
                reference=reference,
                w_loss_sparse_depth=w_loss_sparse_depth,
                w_loss_smoothness=w_loss_smoothness,
                w_loss_cos=w_loss_cos)
            loss_cotta = self.cotta_loss(output_depth, sparse_depth[1])
            loss_info['loss_cotta'] = loss_cotta.data.item()
            loss = loss_adapt + loss_cotta * w_loss_cos
            return loss, loss_info

        if 'init' in loss_type:
            return self.model.compute_loss(
                input_rgb=input_rgb,
                output_depth=output_depth,
                validity_map=validity_map,
                ground_truth=ground_truth,
                loss_type=loss_type)
        if '_bn' in loss_type:
            return self.sparse_depth_loss(
                input_rgb=input_rgb,
                output_depth=output_depth,
                sparse_depth=sparse_depth,
                validity_map=validity_map
            )

        elif 'adapt' in loss_type:
            if self.max_input_depth is not None:
                sparse_depth = torch.clamp(sparse_depth, 0, self.max_input_depth)
            return self.adapt_loss(
                input_rgb=input_rgb,
                output_depth=output_depth,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                embedding=embedding,
                reference=reference,
                w_loss_sparse_depth=w_loss_sparse_depth,
                w_loss_smoothness=w_loss_smoothness,
                w_loss_cos=w_loss_cos)

        elif 'selfsup' in loss_type:
            return self.selfsup_loss(
                input_rgb=input_rgb,
                output_depth=output_depth,
                ground_truth=ground_truth,
                validity_map=validity_map,
                embedding=embedding,
                reference=reference,
                w_loss_dep=1.0,
                w_loss_cos=1.0)

        else:    
            if self.model_name == 'costdcnet':
                loss, loss_info = self.model.compute_loss(
                    input_rgb=input_rgb,
                    output_depth=output_depth,
                    validity_map=validity_map,
                    ground_truth=ground_truth,
                    dataset_name=dataset_name,
                    loss_type=loss_type)

            else:
                loss, loss_info = self.model.compute_loss(
                    input_rgb=input_rgb,
                    output_depth=output_depth,
                    validity_map=validity_map,
                    ground_truth=ground_truth,
                    loss_type=loss_type)
                
            if w_loss_smoothness != 0.0:
                loss += w_loss_smoothness * smoothness_loss_func(output_depth,
                                                input_rgb)
            return loss, loss_info

    def cotta_loss(self,
                          output_depth,
                          ref_depth):
        if 'nlspn' in self.model_name:
            output_depth = torch.clamp(output_depth, min=0, max=self.max_predict_depth)
            ref_depth = torch.clamp(ref_depth, min=0, max=self.max_predict_depth)
        mask = (ref_depth > 0.0001).type_as(output_depth).detach()
        # Obtain valid values
        # Compute individual losses
        # l1 loss
        d = torch.abs(output_depth - ref_depth) * mask
        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])
        loss_l1 = d / (num_valid + 1e-8)
        loss_l1 = loss_l1.sum()
        return loss_l1

    def selfsup_loss(self,
                  input_rgb,
                  output_depth,
                  ground_truth,
                  validity_map,
                  embedding,
                  reference,
                  w_loss_dep=1.0,
                  w_loss_cos=1.0):
        '''
        Adapting model
            with smoothness / sparse depth consistency / self-supervised objective.

        inputs:
            input_rgb : torch.Tensor[float32]
                Input photometric image (B x 3 x H x W)
            output_depth : torch.Tensor[float32]
                Output dense depth prediction (B x 1 x H x W) from model
            sparse_depth : torch.Tensor[float32]
                Input sparse depth map (B x 1 x H x W)
            emb : torch.Tensor[float32]
                Embedding (B x C x (H * W) x D) from cropped image.
            ref : torch.Tensor[float32]
                Reference (B x C x (H * W) x D) from non-cropped image.
            w_loss_sparse_depth : float32
                The weight of sparse depth consistency loss.
            w_loss_smoothness : float32
                The weight of smoothness constraint loss.
            w_loss_cos : float32
                The weight of cosine similarityloss.

        Output :
            loss : The sum of the following adapt losses.
                Sparse depth consistency loss to match the output depth map to according points in sparse depth map.
                Smoothness loss to match
                Cosine similarity loss to maximize the cos similarity between emb and ref
            loss_info : Dictionary which includes the individual value of the each loss.
        '''
        loss_dep, _ = self.model.compute_loss(
            input_rgb=input_rgb,
            output_depth=output_depth,
            validity_map=validity_map,
            ground_truth=ground_truth,
            loss_type='pretrain')

        if embedding is not None and reference is not None:
            embedding = F.normalize(embedding, dim=-1, p=2)
            reference = F.normalize(reference, dim=-1, p=2)
            loss_cos = (2 - 2 * (embedding * reference).sum(dim=-1)).mean()
        else:
            loss_cos = 0

        loss = w_loss_dep * loss_dep + w_loss_cos * loss_cos

        loss_info = {
            'loss': loss,
            'loss_dep': loss_dep,
            'loss_cos': loss_cos
        }

        return loss, loss_info

    def sparse_depth_loss(self,
                  input_rgb,
                  output_depth,
                  sparse_depth,
                  validity_map):
        '''
        Adapting model
            with smoothness / sparse depth consistency / self-supervised objective.

        inputs:
            input_rgb : torch.Tensor[float32]
                Input photometric image (B x 3 x H x W)
            output_depth : torch.Tensor[float32]
                Output dense depth prediction (B x 1 x H x W) from model
            sparse_depth : torch.Tensor[float32]
                Input sparse depth map (B x 1 x H x W)
            emb : torch.Tensor[float32]
                Embedding (B x C x (H * W) x D) from cropped image.
            ref : torch.Tensor[float32]
                Reference (B x C x (H * W) x D) from non-cropped image.
            w_loss_sparse_depth : float32
                The weight of sparse depth consistency loss.
            w_loss_smoothness : float32
                The weight of smoothness constraint loss.
            w_loss_cos : float32
                The weight of cosine similarityloss.

        Output :
            loss : The sum of the following adapt losses.
                Sparse depth consistency loss to match the output depth map to according points in sparse depth map.
                Smoothness loss to match
                Cosine similarity loss to maximize the cos similarity between emb and ref
            loss_info : Dictionary which includes the individual value of the each loss.
        '''

        loss_smooth = smoothness_loss_func(
            predict=output_depth,
            image=input_rgb)

        loss_sparse_consistency = sparse_depth_consistency_loss_func(
            src=output_depth,
            tgt=sparse_depth,
            w=validity_map)

        loss = loss_sparse_consistency + loss_smooth

        loss_info = {
            'loss': loss,
        }
        # , 'sparse': loss_sparse_consistency}

        return loss, loss_info

    def adapt_loss(self,
                  input_rgb,
                  output_depth,
                  sparse_depth,
                  validity_map,
                  embedding,
                  reference,
                  w_loss_sparse_depth=1.0,
                  w_loss_smoothness=1.0,
                  w_loss_cos=1.0):
        '''
        Adapting model
            with smoothness / sparse depth consistency / self-supervised objective.

        inputs:
            input_rgb : torch.Tensor[float32]
                Input photometric image (B x 3 x H x W)
            output_depth : torch.Tensor[float32]
                Output dense depth prediction (B x 1 x H x W) from model
            sparse_depth : torch.Tensor[float32]
                Input sparse depth map (B x 1 x H x W)
            emb : torch.Tensor[float32]
                Embedding (B x C x (H * W) x D) from cropped image.
            ref : torch.Tensor[float32]
                Reference (B x C x (H * W) x D) from non-cropped image.
            w_loss_sparse_depth : float32
                The weight of sparse depth consistency loss.
            w_loss_smoothness : float32
                The weight of smoothness constraint loss.
            w_loss_cos : float32
                The weight of cosine similarityloss.

        Output :
            loss : The sum of the following adapt losses.
                Sparse depth consistency loss to match the output depth map to according points in sparse depth map.
                Smoothness loss to match
                Cosine similarity loss to maximize the cos similarity between emb and ref
            loss_info : Dictionary which includes the individual value of the each loss.
        '''

        loss_smooth = smoothness_loss_func(
            predict=output_depth,
            image=input_rgb)

        loss_sparse_consistency = sparse_depth_consistency_loss_func(
            src=output_depth,
            tgt=sparse_depth,
            w=validity_map)

        if embedding is not None and reference is not None:
            embedding = F.normalize(embedding, dim=-1, p=2)
            reference = F.normalize(reference, dim=-1, p=2)
            loss_cos = (2 - 2 * (embedding * reference).sum(dim=-1)).mean()
            if loss_cos < 0.3:
                w_loss_cos = 0
        else:
            loss_cos = 0

        loss = w_loss_sparse_depth * loss_sparse_consistency + \
            w_loss_smoothness * loss_smooth + \
            w_loss_cos * loss_cos

        loss_info = {
            'loss': loss,
            'loss_smooth': loss_smooth,
            'loss_sparse_depth': loss_sparse_consistency,
            'loss_cos': loss_cos
        }
        # , 'sparse': loss_sparse_consistency}

        return loss, loss_info

    def dense_adapt_loss(self,
                  input_rgb,
                  output_depth,
                  sparse_depth,
                  reference_depth,
                  validity_map,
                  embedding,
                  reference,
                  w_loss_sparse_depth=1.0,
                  w_loss_smoothness=1.0,
                  w_loss_robust=1.0,
                  w_loss_cos=1.0):
        '''
        Adapting model
            with smoothness / sparse depth consistency / self-supervised objective.

        inputs:
            input_rgb : torch.Tensor[float32]
                Input photometric image (B x 3 x H x W)
            output_depth : torch.Tensor[float32]
                Output dense depth prediction (B x 1 x H x W) from model
            sparse_depth : torch.Tensor[float32]
                Input sparse depth map (B x 1 x H x W)
            emb : torch.Tensor[float32]
                Embedding (B x C x (H * W) x D) from cropped image.
            ref : torch.Tensor[float32]
                Reference (B x C x (H * W) x D) from non-cropped image.
            w_loss_sparse_depth : float32
                The weight of sparse depth consistency loss.
            w_loss_smoothness : float32
                The weight of smoothness constraint loss.
            w_loss_robust : float32
                The weight of robustness loss.
            w_loss_cos : float32
                The weight of cosine similarityloss.

        Output :
            loss : The sum of the following adapt losses.
                Sparse depth consistency loss to match the output depth map to according points in sparse depth map.
                Smoothness loss to match
                Cosine similarity loss to maximize the cos similarity between emb and ref
            loss_info : Dictionary which includes the individual value of the each loss.
        '''
        reference_depth = torch.clamp(reference_depth, 0, self.max_predict_depth)
        loss_smooth = smoothness_loss_func(
            predict=output_depth,
            image=input_rgb)

        loss_robust = robustness_loss_func(
            src=output_depth,
            tgt=reference_depth,
            w=1-validity_map)

        loss_sparse_consistency = sparse_depth_consistency_loss_func(
            src=output_depth,
            tgt=sparse_depth,
            w=validity_map)

        if embedding is not None and reference is not None:
            embedding = F.normalize(embedding, dim=-1, p=2)
            reference = F.normalize(reference, dim=-1, p=2)
            loss_cos = (2 - 2 * (embedding * reference).sum(dim=-1)).mean()
        else:
            loss_cos = 0

        loss = w_loss_sparse_depth * loss_sparse_consistency + \
            w_loss_smoothness * loss_smooth + \
            w_loss_cos * loss_cos + \
            w_loss_robust * loss_robust

        loss_info = {
            'loss': loss,
            'loss_smooth': loss_smooth,
            'loss_sparse_depth': loss_sparse_consistency,
            'loss_cos': loss_cos,
            'loss_robust': loss_robust
        }
        # , 'sparse': loss_sparse_consistency}

        return loss, loss_info

    def prepare_loss(self,
                     embedding,
                     reference):
        '''
        Training MLP Layer with self-supervised objective
        emb : torch.Tensor[float32]
            Embedding (C x (H * W) x D) from cropped image.
        ref : torch.Tensor[float32]
            Reference (C x (H * W) x D) from non-cropped image.

        Output : Cosine similarity loss to maximize the cos similarity between emb and ref
        '''
        embedding = F.normalize(embedding, dim=-1, p=2)
        reference = F.normalize(reference, dim=-1, p=2)
        loss = (2 - 2 * (embedding * reference).sum(-1)).mean()
        loss_info = {'loss': loss}
        return loss, loss_info

    def bn_adapt_parameters(self):
        '''
        Returns the list of BatchNorm parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return self.model.bn_adapt_parameters()

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def _prepare_head(self, mode=''):
        self.model._prepare_head(mode=mode)

    def prepare_parameters(self, mode=''):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return self.model.prepare_parameters(mode)

    def adapt_parameters(self, mode=''):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        return self.model.adapt_parameters(mode=mode)

    def train(self, meta=False, prepare=False):
        '''
        Sets model to training mode
        '''
        if meta:
            self.model.train_meta()
        elif prepare:
            self.model.train_prepare()
        else:
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

        self.model.data_parallel()

    def distributed_data_parallel(self, rank):
        '''
        Allows multi-gpu split along batch
        '''

        self.model.distributed_data_parallel(rank)
        if 'costdcnet' in self.model_name:
            self.model.set_device('cuda:{}'.format(rank))

    def restore_model(self, restore_path, optimizer=None, learning_schedule=None, learning_rates=None, n_step_per_epoch=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer or None if no optimizer is passed in
        '''

        return self.model.restore_model(restore_path=restore_path,
                                        optimizer=optimizer)

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

        self.model.save_model(checkpoint_path, step, optimizer, meanvar)

    def convert_syncbn(self, apex=False):
        self.model.convert_syncbn(apex)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image=None,
                    output_image=None,
                    sparse_depth=None,
                    output_depth=None,
                    validity_map=None,
                    ground_truth=None,
                    normalized_image_range=None,
                    scalars={},
                    n_image_per_summary=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image : torch.Tensor[float32]
                N x 3 x H x W image from camera
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse_depth from LiDAR
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth for image
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map from sparse depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth image
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display
        '''
        if normalized_image_range is None:
            do_normalization_standard = False
        else:
            do_normalization_standard = any([
                isinstance(value, tuple) or isinstance(value, list)
                for value in normalized_image_range]) or len(normalized_image_range) > 2
            if (len(normalized_image_range) > 2):
                mean_stddev_split_idx = len(normalized_image_range) // 2
                normalized_image_range = [
                    tuple(normalized_image_range[:mean_stddev_split_idx]),
                    tuple(normalized_image_range[mean_stddev_split_idx:])]
        if do_normalization_standard:
            reverse_mean = []
            reverse_std = []

            for m, std in zip(normalized_image_range[0], normalized_image_range[1]):
                reverse_mean.append(-m/std)
                reverse_std.append(1/std)
            reverse_mean = tuple(reverse_mean)
            reverse_std = tuple(reverse_std)
        else:
            m, std = normalized_image_range[0], normalized_image_range[1]

            reverse_mean = - m / std
            reverse_std = 1 / std

        with torch.no_grad():
            display_summary_image = []
            display_summary_depth = []
            display_summary_output_image = []

            display_summary_image_text = tag
            display_summary_depth_text = tag
            display_summary_output_image_text = tag
            # Log image
            if image is not None:
                # Normalize for display if necessary

                if isinstance(image, list):
                    for img in image:
                        if torch.max(img) > 10.0:
                            img = img / 255.0
                        # Reverse_normalization for visualization
                        # mean = - norm_mean/norm_std, std = 1/norm_std
                        if do_normalization_standard:
                            img = torchvision.transforms.functional.normalize(
                                img,
                                reverse_mean,
                                reverse_std)
                        image_summary = img[0:n_image_per_summary, ...]

                        display_summary_image.append(
                            torch.cat([
                                image_summary.cpu(),
                                torch.zeros_like(image_summary, device=torch.device('cpu'))],
                                dim=-1))
                else:
                    if torch.max(image) > 10.0:
                        image = image / 255.0
                    # Reverse_normalization for visualization
                    # mean = - norm_mean/norm_std, std = 1/norm_std
                    if do_normalization_standard:
                        image = torchvision.transforms.functional.normalize(
                            image,
                            reverse_mean,
                            reverse_std)
                    image_summary = image[0:n_image_per_summary, ...]
                    display_summary_image.append(
                        torch.cat([
                            image_summary.cpu(),
                            torch.zeros_like(image_summary, device=torch.device('cpu'))],
                            dim=-1))

                display_summary_depth.append(display_summary_image[-1])

                display_summary_image_text += '_image'
                display_summary_depth_text += '_image'

                if output_image is not None:
                    display_summary_output_image.append(display_summary_image[-1])
                    output_image = torchvision.transforms.functional.normalize(
                        output_image,
                        reverse_std,
                        reverse_mean)

                    output_image_summary = output_image[0:n_image_per_summary, ...]
                    display_summary_output_image_text += '_output_image'
                    display_summary_output_image.append(
                        torch.cat([
                            output_image_summary.cpu(),
                            torch.zeros_like(output_image_summary, device=torch.device('cpu'))],
                            dim=-1))
                    display_summary_output_image.append(display_summary_output_image[-1])

            if output_depth is not None:

                output_depth_summary = output_depth[0:n_image_per_summary]
                display_summary_depth_text += '_output'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

            # Log output depth vs sparse depth
            if output_depth is not None and sparse_depth is not None and validity_map is not None:
                sparse_depth_summary = sparse_depth[0:n_image_per_summary]
                validity_map_summary = validity_map[0:n_image_per_summary]
                display_summary_depth_text += '_sparse-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth_error_summary = \
                    torch.abs(output_depth_summary - sparse_depth_summary)
                sparse_depth_error_summary = torch.where(
                    validity_map_summary > 0,
                    sparse_depth_error_summary / (sparse_depth_summary + 1e-8),
                    validity_map_summary)

                # Add to list of images to log
                sparse_depth_summary = log_utils.colorize(
                    (sparse_depth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth_error_summary = log_utils.colorize(
                    (sparse_depth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth_summary,
                        sparse_depth_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth_distro', sparse_depth, global_step=step)

            # Log output depth vs ground truth depth
            if output_depth is not None and ground_truth is not None:
                validity_map_ground_truth = torch.where(
                    ground_truth > 0,
                    torch.ones_like(ground_truth),
                    torch.zeros_like(ground_truth))

                validity_map_ground_truth_summary = validity_map_ground_truth[0:n_image_per_summary]
                ground_truth_summary = ground_truth[0:n_image_per_summary]

                display_summary_depth_text += '_groundtruth-error'

                # Compute output error w.r.t. ground truth
                ground_truth_error_summary = \
                    torch.abs(output_depth_summary - ground_truth_summary)

                ground_truth_error_summary = torch.where(
                    validity_map_ground_truth_summary == 1.0,
                    (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                    validity_map_ground_truth_summary)

                # Add to list of images to log
                ground_truth_summary = log_utils.colorize(
                    (ground_truth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth_error_summary = log_utils.colorize(
                    (ground_truth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth_summary,
                        ground_truth_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) >= 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
                    global_step=step)

            if len(display_summary_output_image) >= 1:
                display_summary_output_image = torch.cat(display_summary_output_image, dim=2)

                summary_writer.add_image(
                    display_summary_output_image_text,
                    torchvision.utils.make_grid(display_summary_output_image, nrow=n_image_per_summary),
                    global_step=step)

            if len(display_summary_depth) >= 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
                    global_step=step)
