import torch
import torch.nn as nn
from models.encoder2d import Encoder2D
from models.encoder3d import Encoder3D
from models.unet3d import UNet3D
import MinkowskiEngine as ME
import numpy as np
import torch.nn.functional as F
import glob
import time
import copy

class CostDCNet(nn.Module):
    def __init__(self, opt):
        super(CostDCNet, self).__init__()
        self.opt = opt
        self.enc2d = Encoder2D(in_ch=4, output_dim=16)
        self.enc3d = Encoder3D(1, 16, D= 3, planes=(32, 48, 64))
        self.unet3d = UNet3D(32, self.opt.up_scale**2, f_maps=[32, 48, 64, 80], mode="nearest")
        self.z_step = self.opt.max/(self.opt.res-1)
        self.total_time = 0.0
        self.train_time = 0.0
        self.eval_time = 0.0

    # def parameters(self):
    #     for m in self.model_list:
    #         self.parameters_to_train += list(m.parameters())
    #         params = sum([np.prod(p.size()) for p in m.parameters()])
    #         print("# param of {}: {}".format(m,params))
    #     return
    def set_device(self, rank):
        torch.cuda.set_device(rank)

    def forward(self,
                image,
                sparse_depth,
                crop_mask=None,
                loss_type=None):

        if 'time' in loss_type:
            torch.cuda.synchronize()
            before_op_time = time.time()

        if 'pretrain' in loss_type:
            mode = []
            if 'time' in loss_type:
                mode.append('time')
            output = self._forward(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode)

            if 'bn' in loss_type and self.training:
                output = [output, None, None]

        # Prepare
        elif 'selfsup' in loss_type and 'init_meta' in loss_type:
            mode = []
            if 'loss' in loss_type:
                mode.append('loss')
            if 'adapt' in loss_type:
                mode.append('adapt')
            elif 'seq' in loss_type:
                mode.append('seq')
            else:
                mode.append('parallel')
            if 'reverse' in loss_type:
                mode.append('reverse')
            if 'ema' in loss_type:
                mode.append('ema')
            else:
                raise NotImplementedError
            output = self._rgbd_meta_contrast_init(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )
        elif 'head' in loss_type:

            mode = []
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'seq' in loss_type:
                mode.append('seq')
            else:
                mode.append('parallel')

            if 'reverse' in loss_type:
                mode.append('reverse')

            if 'ema' in loss_type:
                mode.append('ema')
            else:
                raise NotImplementedError

            output = self._rgbd_meta_contrast_prepare(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )
        # init-meta

        elif 'selfsup' in loss_type or 'meta' in loss_type:
            mode = []
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'seq' in loss_type:
                mode.append('seq')
            else:
                mode.append('parallel')

            if 'reverse' in loss_type:
                mode.append('reverse')

            if 'ema' in loss_type:
                mode.append('ema')
            else:
                raise NotImplementedError

            output = self._rgbd_meta_contrast(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )

        elif 'prepare' in loss_type and 'selfsup' in loss_type:
            mode = []

            if 'reverse' in loss_type:
                mode.append('reverse')

            if 'ema' in loss_type:
                mode.append('ema')
            else:
                raise NotImplementedError

            output = self._prepare_selfsup_reverse(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )

        elif 'adapt' in loss_type and 'selfsup' in loss_type:
            mode = []

            if 'reverse' in loss_type:
                mode.append('reverse')

            if 'ema' in loss_type:
                mode.append('ema')
            else:
                raise NotImplementedError

            output = self._adapt_selfsup_reverse(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )

        # Evaluation
        else:
            output = self._forward(
                image=image,
                sparse_depth=sparse_depth)

        if 'time' in loss_type:
            torch.cuda.synchronize()
            op_time = time.time() - before_op_time
            if self.training:
                self.train_time += op_time
            else:
                self.eval_time += op_time
            self.total_time += op_time

        return output

    def _forward(self, image, sparse_depth, mode):

        ###############################################################

        in_2d = torch.cat([image, sparse_depth], 1)
        in_3d = self.depth2MDP(sparse_depth)
        feat2d = self.enc2d(in_2d)
        feat3d = self.enc3d(in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        cost_vol = self.unet3d(rgbd_feat_vol)
        # if we want to train batch_size > 1, we need to modify this part.

        # Test
        # output_list = []
        # for i in range(batch_size):
        #     output = self.upsampling(cost_vol[i:i+1], res = self.opt.res, up_scale=self.opt.up_scale) * self.z_step
        #     output_list.append(output)
        # outputs[("depth", 0, 0)] = torch.vstack(output_list)
        pred = self.upsampling(cost_vol, res=self.opt.res, up_scale=self.opt.up_scale) * self.z_step

        ###############################################################
        # if self.opt.time:
        #     torch.cuda.synchronize()
        #     outputs["time"] = time.time() - before_op_time
        #     outputs["mem"] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        #     print(outputs["time"], outputs["mem"])
        # outputs[("depth", 0, 0)] = pred
        return pred

    def _rgbd_meta_contrast(self, image, sparse_depth, mode='seq'):
        ###############################################################

        in_2d = torch.cat([image, sparse_depth],1)
        in_3d = self.depth2MDP(sparse_depth)
        if 'new' not in mode:
            feat2d = self.conv1_rgb_meta(self.enc2d(in_2d))
        else:
            feat2d = self.enc2d(in_2d)
        feat3d = self.enc3d(in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        cost_vol, feat = self.unet3d(rgbd_feat_vol, return_feature=True)

        # if we want to train batch_size > 1, we need to modify this part..?
        

        pred = self.upsampling(cost_vol, res=self.opt.res, up_scale=self.opt.up_scale) * self.z_step
        # in_2d = torch.cat([image, sparse_depth],1)
        # in_3d = self.depth2MDP(sparse_depth)
        # feat2d = self.enc2d(in_2d)
        # feat3d = self.enc3d(in_3d)
        # rgbd_feat_vol = self.fusion(feat3d, feat2d)

        # cost_vol = self.unet3d(rgbd_feat_vol)

        # # if we want to train batch_size > 1, we need to modify this part..?

        # pred = self.upsampling(cost_vol, res = self.opt.res, up_scale=self.opt.up_scale) * self.z_step
        if self.training:
            with torch.no_grad():
                in_2d_zero = torch.cat([torch.zeros_like(image), sparse_depth], 1)
                feat2d_zero = self.conv1_rgb_meta(self.enc2d(in_2d_zero))
                feat3d_zero = self.enc3d(in_3d)
                rgbd_feat_vol_zero = self.fusion(feat3d_zero, feat2d_zero)

                _, feat_zero = self.unet3d(rgbd_feat_vol_zero, return_feature=True)
            ###############################################################
            b, c, d, h, w = feat.size()
            feat = feat.reshape(b, c*d, h, w)
            feat_zero = feat_zero.reshape(b, c*d, h, w)
            
            feat_dim = feat.size(1)
            proj_rgb = self.pred(self.proj(feat_zero.permute(0, 2, 3, 1).reshape(-1, feat_dim))).detach()
            proj = self.proj_t(feat.permute(0, 2, 3, 1).reshape(-1, feat_dim))

        if self.training:
            return [pred, proj_rgb, proj]
        else:
            return pred

    def _rgbd_meta_contrast_prepare(self, image, sparse_depth, mode='seq'):

        if self.opt.time:
            torch.cuda.synchronize()
            before_op_time = time.time()

        batch_size = image.size(0)

        outputs = {}
        ###############################################################
        with torch.no_grad():
            in_2d = torch.cat([image, sparse_depth],1)
            in_3d = self.depth2MDP(sparse_depth)
            feat2d = self.conv1_rgb_meta(self.enc2d(in_2d))
            feat3d = self.enc3d(in_3d)
            rgbd_feat_vol = self.fusion(feat3d, feat2d)

            cost_vol, feat = self.unet3d(rgbd_feat_vol, return_feature = True)

            # if we want to train batch_size > 1, we need to modify this part..?

            in_2d_zero = torch.cat([torch.zeros_like(image).cuda(), sparse_depth], 1)
            if 'new' not in mode:
                feat2d_zero = self.conv1_rgb_meta(self.enc2d(in_2d_zero))
            else:
                feat2d_zero = self.enc2d(in_2d_zero)
            feat3d_zero = self.enc3d(in_3d)
            rgbd_feat_vol_zero = self.fusion(feat3d_zero, feat2d_zero)

            _, feat_zero = self.unet3d(rgbd_feat_vol_zero, return_feature=True)

            # pred = self.upsampling(cost_vol, res=self.opt.res, up_scale=self.opt.up_scale) * self.z_step
            # if we want to train batch_size > 1, we need to modify this part..?
            ###############################################################

        if 'ema' in mode:
            b, c, d, h, w = feat.size()
            feat = feat.reshape(b, c*d, h, w)
            feat_zero = feat_zero.reshape(b, c*d, h, w)
            if 'reverse' not in mode:
                if 'adapt' not in mode:
                    self._update_head()
                    feat_dim = feat.size(1)
                    proj_rgb = self.pred(self.proj(feat.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                    proj = self.proj_t(feat_zero.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                else:
                    feat_dim = feat.size(1)
                    proj_rgb = self.pred(self.proj(feat.permute(0, 2, 3, 1).reshape(-1, feat_dim)))
                    proj = self.proj_t(feat_zero.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()

            else:
                if 'adapt' not in mode:
                    self._update_head()
                    feat_dim = feat.size(1)
                    proj_rgb = self.pred(self.proj(feat_zero.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                    proj = self.proj_t(feat.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                else:
                    feat_dim = feat.size(1)
                    proj_rgb = self.pred(self.proj(feat_zero.permute(0, 2, 3, 1).reshape(-1, feat_dim))).detach()
                    proj = self.proj_t(feat.permute(0, 2, 3, 1).reshape(-1, feat_dim))

        return None, proj_rgb, proj

    def _rgbd_meta_contrast_init(self, image, sparse_depth, mode='seq'):

        if self.opt.time:
            torch.cuda.synchronize()
            before_op_time = time.time()
        outputs = {}
        ###############################################################

        in_2d = torch.cat([image, sparse_depth], 1)
        in_3d = self.depth2MDP(sparse_depth)
        if 'new' not in mode:
            feat2d = self.conv1_rgb_meta(self.enc2d(in_2d))
        else:
            feat2d = self.enc2d(in_2d)
        feat3d = self.enc3d(in_3d)
        rgbd_feat_vol = self.fusion(feat3d, feat2d)

        cost_vol = self.unet3d(rgbd_feat_vol)

        # if we want to train batch_size > 1, we need to modify this part..?

        pred = self.upsampling(cost_vol, res=self.opt.res, up_scale=self.opt.up_scale) * self.z_step

        ###############################################################
        if self.opt.time:
            torch.cuda.synchronize()
            outputs["time"] = time.time() - before_op_time
            outputs["mem"] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            print(outputs["time"], outputs["mem"])
        outputs[("depth", 0, 0)] = pred
        if 'loss' in mode:
            return pred, l1_loss
        else:
            return pred

    def depth2MDP(self, dep):
        # Depth to sparse tensor in MDP (multiple-depth-plane)
        idx = torch.round(dep / self.z_step).type(torch.int64)
        idx[idx>(self.opt.res-1)] = self.opt.res - 1
        idx[idx<0] = 0
        inv_dep = (idx * self.z_step)
        res_map = (dep - inv_dep) /self.z_step

        B, C, H, W = dep.size()
        ones = (idx !=0).float()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        grid_ = torch.stack((grid_y, grid_x), 2).to(dep.device)
        # grid_ = self.grid.clone().detach()
        grid_ = grid_.unsqueeze(0).repeat((B,1,1,1))
        points_yx = grid_.reshape(-1,2)
        point_z = idx.reshape(-1, 1)
        m = (idx != 0).reshape(-1)
        points3d = torch.cat([point_z, points_yx], dim=1)[m]
        split_list = torch.sum(ones, dim=[1,2,3], dtype=torch.int).tolist()
        coords = points3d.split(split_list)
        # feat = torch.ones_like(points3d)[:,0].reshape(-1,1)       ## if occ to feat
        feat = res_map
        feat = feat.permute(0,2,3,1).reshape(-1, feat.size(1))[m]   ## if res to feat

        # Convert to a sparse tensor
        in_field = ME.TensorField(
            features = feat,
            coordinates=ME.utils.batched_coordinates(coords, dtype=torch.float32),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=dep.device,
        )
        return in_field.sparse()

    def fusion(self, sout, feat2d):
        # sparse tensor to dense tensor
        B0,C0,H0,W0 = feat2d.size()
        # sout = sout.to(torch.device('cpu'))
        dense_output_, min_coord, tensor_stride = sout.dense()
        dense_output_.cuda()

        dense_output = dense_output_[:, :, :self.opt.res, :H0, :W0]
        B,C,D,H,W = dense_output.size()
        feat3d_ = torch.zeros((B0, C0, self.opt.res, H0, W0), device = feat2d.device)
        feat3d_[:B,:,:D,:H,:W] += dense_output

        # construct type C feat vol
        mask = (torch.sum((feat3d_ != 0), dim=1, keepdim=True)!= 0).float()
        mask_ = mask + (1 - torch.sum(mask, dim=2,keepdim=True).repeat(1,1,mask.size(2),1,1))
        feat2d_ = feat2d.unsqueeze(2).repeat(1,1,self.opt.res,1,1) * mask_
        return torch.cat([feat2d_, feat3d_],dim = 1)

    def upsampling(self, cost, res=64, up_scale=None):
        # if up_scale is None not apply per-plane pixel shuffle
        if not up_scale == None:
            b, c, d, h, w = cost.size()
            cost = cost.transpose(1,2).reshape(b, -1, h, w)  # size: (b, c*d, h, w)
            cost = F.pixel_shuffle(cost, up_scale)  # size: (b, , h, w)
        else:
            cost = cost.squeeze(1)
        prop = F.softmax(cost, dim = 1)
        pred =  self.disparity_regression(prop, res)
        return pred

    def disparity_regression(self, x, maxdisp):
        assert len(x.shape) == 4
        disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
        disp_values = disp_values.contiguous().view(1, maxdisp, 1, 1)
        return torch.sum(x * disp_values, 1, keepdim=True)

    def _update_head(self, tau=0.999):
        for t, s in zip(self.proj_t.parameters(), self.proj.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def _prepare_head(self, mode=''):
        '''
        Initialize the projection and prediction heads
        '''
        if 'roi' in mode:
            in_channel_rgb = 48
            in_channel_dep = 16
        else:
            in_channel_rgb = 48 * 16
            in_channel_dep = 16 * 16

        if 'mask' in mode:
            self.proj_rgb = self.MLP(in_channel_rgb, in_channel_dep, 512).to(torch.device('cuda'))
            self.proj = self.MLP(in_channel_dep, 512, 512).to(torch.device('cuda'))
            self.pred = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024)
            ).cuda()
        elif 'rgbd' in mode:
            self.rgb_matching_layer = nn.Sequential(conv_bn_relu(48, 256, 3, 1, 1).cuda(),
                                                    conv_bn_relu(256, 16, 3, 1, 1).cuda())
            self.proj = self.MLP(in_channel_dep, 1024, 1024).cuda()
            if 'ema' in mode:
                self.proj_t = copy.deepcopy(self.proj)

        elif 'selfsup' in mode:
            if 'ema' in mode:
                self.proj = self.MLP(160, 512, 512).to(torch.device('cuda'))
                self.proj_t = copy.deepcopy(self.proj)
                self.pred = self.MLP(512, 512, 512).to(torch.device('cuda'))
                # self.proj = self.MLP(512, 1024, 1024).to(torch.device('cuda'))
                # self.proj_t = copy.deepcopy(self.proj)
                # self.pred = self.MLP(1024, 1024, 1024).to(torch.device('cuda'))
            elif 'ena' in mode:
                self.proj = self.MLP(160, 512, 512).to(torch.device('cuda'))
                self.proj_t = copy.deepcopy(self.proj)
                self.pred = self.MLP(512, 512, 512).to(torch.device('cuda'))
            else:
                self.proj = self.MLP(512, 1024, 1024).to(torch.device('cuda'))
                self.pred = self.pred = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 1024),
                    nn.BatchNorm1d(1024)).to(torch.device('cuda'))

        if mode is not None and 'meta' in mode:
            if 'seq' in mode:
                if 'new' in mode:
                    self.conv1_rgb_meta = nn.Identity()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                    self.enc2d.conv1_rgb_meta = conv_bn_relu(128, 16, 3, 1, 1).cuda()
                    self.enc2d.meta_bn_rgb = nn.BatchNorm2d(16).cuda()

                elif '1layer' in mode:
                    self.conv1_rgb_meta = nn.Conv2d(16, 16, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()

                elif '2layers' in mode:
                    self.conv1_rgb_meta = Res_Conv(16, 64, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()

                elif '1conv' in mode:
                    self.conv1_rgb_meta = nn.Conv2d(48, 48, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.BatchNorm2d(48).cuda()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                else:
                    self.conv1_rgb_meta = nn.Sequential(conv_bn_relu(48, 48, 3, 1, 1).cuda(),
                                                    conv_bn_relu(48, 48, 3, 1, 1).cuda())
                    self.meta_bn_rgb = nn.BatchNorm2d(48).cuda()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()

            elif 'parallel' in mode:
                self.conv1_rgb_meta = conv_bn_relu(3, 48, 3, 1, 1).cuda()
                self.meta_bn_rgb = nn.BatchNorm2d(48).cuda()
                self.conv1_dep_meta = conv_bn_relu(1, 16, 3, 1, 1).cuda()
                self.meta_bn_dep = nn.BatchNorm2d(16).cuda()

            elif 'adjust' in mode:
                self.conv1_rgb_meta = nn.Conv2d(3, 3, 3, 1, 1).cuda()
                self.conv1_dep_meta = nn.Identity()


    def MLP(self, dim, projection_size, hidden_size=4096):
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers

class Res_Conv(nn.Module):
    def __init__(self, ch_in, ch_hidden, kernel_size, stride, padding):
        super(Res_Conv, self).__init__()
        self.conv1_meta = nn.Sequential(conv_bn_relu(ch_in, ch_hidden, kernel_size, stride, padding).cuda(),
            nn.Conv2d(ch_hidden, ch_in, kernel_size, stride, padding).cuda(),
            nn.BatchNorm2d(ch_in).cuda())

    def forward(self, x):
        return self.conv1_meta(x) + x