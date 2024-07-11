"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon

    European Conference on Computer Vision (ECCV), Aug 2020

    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)

    ======================================================================

    NLSPN implementation
"""


from common import conv_bn_relu, convt_bn_relu, get_resnet18, get_resnet34
from modulated_deform_conv_func import ModulatedDeformConvFunction
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy
import time
from typing import Type, Any, Callable, Union, List, Optional
# import math

# class ScaledDotProductAttention(nn.Module):

#     def forward(self, query, key, value, mask=None):
#         dk = query.size()[-1]
#         scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attention = F.softmax(scores, dim=-1)
#         return attention.matmul(value)

class Res_Conv(nn.Module):
    def __init__(self, ch_in, ch_hidden, kernel_size, stride, padding):
        super(Res_Conv, self).__init__()
        self.conv1_meta = nn.Sequential(conv_bn_relu(ch_in, ch_hidden, kernel_size, stride, padding).cuda(),
            nn.Conv2d(ch_hidden, ch_in, kernel_size, stride, padding).cuda(),
            nn.BatchNorm2d(ch_in).cuda())

    def forward(self, x):
        return self.conv1_meta(x) + x


class Res_Conv2(nn.Module):
    def __init__(self, ch_in, ch_hidden, kernel_size, stride, padding):
        super(Res_Conv2, self).__init__()
        self.conv1_meta = nn.Sequential(conv_bn_relu(ch_in, ch_hidden, kernel_size, stride, padding).cuda(),
            conv_bn_relu(ch_hidden, ch_in, kernel_size, stride, padding).cuda())
        self.bn1_meta = nn.BatchNorm2d(ch_in).cuda()

    def forward(self, x):
        return self.conv1_meta(x) + self.bn1_meta(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, in_dim, key_dim, out_dim, num_head):
        super(MultiHeadAttention).__init__()
        self.q = nn.Linear(in_features=in_dim, out_features=in_dim, bias=True)
        self.k = nn.Linear(in_features=key_dim, out_features=in_dim, bias=True)
        self.v = nn.Linear(in_features=in_dim, out_features=in_dim, bias=True)
        self.activation = F.relu
        self.linear = nn.Linear()

    def forward(self):
        return 0
#     return

class NLSPN(nn.Module):
    def __init__(self, args, ch_g, ch_f, k_g, k_f):
        super(NLSPN, self).__init__()

        # Guidance : [B x ch_g x H x W]
        # Feature : [B x ch_f x H x W]

        # Currently only support ch_f == 1
        assert ch_f == 1, 'only tested with ch_f == 1 but {}'.format(ch_f)

        assert (k_g % 2) == 1, \
            'only odd kernel is supported but k_g = {}'.format(k_g)
        pad_g = int((k_g - 1) / 2)
        assert (k_f % 2) == 1, \
            'only odd kernel is supported but k_f = {}'.format(k_f)
        pad_f = int((k_f - 1) / 2)

        self.args = args
        self.prop_time = self.args.prop_time
        self.affinity = self.args.affinity

        self.ch_g = ch_g
        self.ch_f = ch_f
        self.k_g = k_g
        self.k_f = k_f
        # Assume zero offset for center pixels
        self.num = self.k_f * self.k_f - 1
        self.idx_ref = self.num // 2

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            self.conv_offset_aff = nn.Conv2d(
                self.ch_g, 3 * self.num, kernel_size=self.k_g, stride=1,
                padding=pad_g, bias=True
            )
            self.conv_offset_aff.weight.data.zero_()
            self.conv_offset_aff.bias.data.zero_()

            if self.affinity == 'TC':
                self.aff_scale_const = nn.Parameter(self.num * torch.ones(1))
                self.aff_scale_const.requires_grad = False
            elif self.affinity == 'TGASS':
                self.aff_scale_const = nn.Parameter(
                    self.args.affinity_gamma * self.num * torch.ones(1))
            else:
                self.aff_scale_const = nn.Parameter(torch.ones(1))
                self.aff_scale_const.requires_grad = False
        else:
            raise NotImplementedError

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((self.ch_f, 1, self.k_f, self.k_f)))
        self.b = nn.Parameter(torch.zeros(self.ch_f))

        self.w.requires_grad = False
        self.b.requires_grad = False

        self.w_conf = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.w_conf.requires_grad = False

        self.stride = 1
        self.padding = pad_f
        self.dilation = 1
        self.groups = self.ch_f
        self.deformable_groups = 1
        self.im2col_step = 64

    def _get_offset_affinity(self, guidance, confidence=None, rgb=None):
        B, _, H, W = guidance.shape

        if self.affinity in ['AS', 'ASS', 'TC', 'TGASS']:
            offset_aff = self.conv_offset_aff(guidance)
            o1, o2, aff = torch.chunk(offset_aff, 3, dim=1)

            # Add zero reference offset
            offset = torch.cat((o1, o2), dim=1).view(B, self.num, 2, H, W)
            list_offset = list(torch.chunk(offset, self.num, dim=1))
            list_offset.insert(self.idx_ref,
                               torch.zeros((B, 1, 2, H, W)).type_as(offset))
            offset = torch.cat(list_offset, dim=1).view(B, -1, H, W)

            if self.affinity in ['AS', 'ASS']:
                pass
            elif self.affinity == 'TC':
                aff = torch.tanh(aff) / self.aff_scale_const
            elif self.affinity == 'TGASS':
                aff = torch.tanh(aff) / (self.aff_scale_const + 1e-8)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Apply confidence
        # TODO : Need more efficient way
        if self.args.conf_prop:
            list_conf = []
            offset_each = torch.chunk(offset, self.num + 1, dim=1)

            modulation_dummy = torch.ones((B, 1, H, W)).type_as(offset).detach()

            for idx_off in range(0, self.num + 1):
                ww = idx_off % self.k_f
                hh = idx_off // self.k_f

                if ww == (self.k_f - 1) / 2 and hh == (self.k_f - 1) / 2:
                    continue

                offset_tmp = offset_each[idx_off].clone().detach()
                # NOTE : Use --legacy option ONLY for the pre-trained models
                # for ECCV20 results.
                if self.args.legacy:
                    offset_tmp[:, 0, :, :] = \
                        offset_tmp[:, 0, :, :] + hh - (self.k_f - 1) / 2
                    offset_tmp[:, 1, :, :] = \
                        offset_tmp[:, 1, :, :] + ww - (self.k_f - 1) / 2

                conf_tmp = ModulatedDeformConvFunction.apply(
                    confidence, offset_tmp, modulation_dummy, self.w_conf,
                    self.b, self.stride, 0, self.dilation, self.groups,
                    self.deformable_groups, self.im2col_step)
                list_conf.append(conf_tmp)

            conf_aff = torch.cat(list_conf, dim=1)
            aff = aff * conf_aff.contiguous()

        # Affinity normalization
        aff_abs = torch.abs(aff)
        aff_abs_sum = torch.sum(aff_abs, dim=1, keepdim=True) + 1e-4

        if self.affinity in ['ASS', 'TGASS']:
            aff_abs_sum[aff_abs_sum < 1.0] = 1.0

        if self.affinity in ['AS', 'ASS', 'TGASS']:
            aff = aff / aff_abs_sum

        aff_sum = torch.sum(aff, dim=1, keepdim=True)
        aff_ref = 1.0 - aff_sum

        list_aff = list(torch.chunk(aff, self.num, dim=1))
        list_aff.insert(self.idx_ref, aff_ref)
        aff = torch.cat(list_aff, dim=1)

        return offset, aff

    def _propagate_once(self, feat, offset, aff):
        feat = ModulatedDeformConvFunction.apply(
            feat, offset, aff, self.w, self.b, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups, self.im2col_step
        )

        return feat

    def forward(self, feat_init, guidance, confidence=None, feat_fix=None,
                rgb=None):
        assert self.ch_g == guidance.shape[1]
        assert self.ch_f == feat_init.shape[1]

        if self.args.conf_prop:
            assert confidence is not None

        if self.args.conf_prop:
            offset, aff = self._get_offset_affinity(guidance, confidence, rgb)
        else:
            offset, aff = self._get_offset_affinity(guidance, None, rgb)

        # Propagation
        if self.args.preserve_input:
            assert feat_init.shape == feat_fix.shape
            mask_fix = torch.sum(feat_fix > 0.0, dim=1, keepdim=True).detach()
            mask_fix = (mask_fix > 0.0).type_as(feat_fix)

        feat_result = feat_init

        list_feat = []

        for k in range(1, self.prop_time + 1):
            # Input preservation for each iteration
            if self.args.preserve_input:
                feat_result = (1.0 - mask_fix) * feat_result \
                    + mask_fix * feat_fix

            feat_result = self._propagate_once(feat_result, offset, aff)

            list_feat.append(feat_result)

        return feat_result, list_feat, offset, aff, self.aff_scale_const.data


class NLSPNModel_Adapt(nn.Module):
    def __init__(self, args):
        super(NLSPNModel_Adapt, self).__init__()

        self.args = args

        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        # Encoder
        self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
                                      bn=False)
        self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                      bn=False)
        # self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, dilation=2, padding=2,
        #                               bn=False)

        if self.args.network == 'resnet18':
            net = get_resnet18(self.args.from_scratch)
        elif self.args.network == 'resnet34':
            net = get_resnet34(self.args.from_scratch)
        else:
            raise NotImplementedError

        # 1/1
        self.conv2 = net.layer1
        # 1/2
        self.conv3 = net.layer2
        # 1/4
        self.conv4 = net.layer3
        # 1/8
        self.conv5 = net.layer4

        del net

        # 1/16
        self.conv6 = conv_bn_relu(512, 512, kernel=3, stride=2, padding=1)

        # Shared Decoder
        # 1/8
        self.dec5 = convt_bn_relu(512, 256, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/4
        self.dec4 = convt_bn_relu(256+512, 128, kernel=3, stride=2,
                                  padding=1, output_padding=1)
        # 1/2
        self.dec3 = convt_bn_relu(128+256, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # 1/1
        self.dec2 = convt_bn_relu(64+128, 64, kernel=3, stride=2,
                                  padding=1, output_padding=1)

        # Init Depth Branch
        # 1/1
        self.id_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.id_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                    padding=1, bn=False, relu=True)

        # Guidance Branch
        # 1/1

        self.gd_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+64, 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid()
            )

        self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                self.args.prop_kernel)

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

        self.glob_mean = None
        self.total_time = 0.0
        self.train_time = 0.0
        self.eval_time = 0.0

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        # Remove additional padding
        if Hd > He:
            h = Hd - He
            fd = fd[:, :, :-h, :]

        if Wd > We:
            w = Wd - We
            fd = fd[:, :, :, :-w]

        f = torch.cat((fd, fe), dim=dim)

        return f

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
            if 'bn' in loss_type:
                mode.append('bn')
            output = self._forward(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode)
        # Prepare
        elif 'head' in loss_type:

            mode = []
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'adjust' in loss_type:
                mode.append('adjust')
            elif 'seq' in loss_type:
                mode.append('seq')
            else:
                mode.append('parallel')

            if 'dense' in loss_type:
                mode.append('dense')
            if 'reverse' in loss_type:
                mode.append('reverse')

            if 'ena' in loss_type:
                mode.append('ena')
            if 'enb' in loss_type:
                mode.append('enb')
            if 'enc' in loss_type:
                mode.append('enc')

            if 'ema' in loss_type:
                mode.append('ema')
            elif 'feature' in loss_type:
                mode.append('feature')
            else:
                raise NotImplementedError

            output = self._rgbd_meta_contrast_prepare(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )
        elif 'selfsup' in loss_type and 'init_meta' in loss_type:
            mode = []
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'adjust' in loss_type:
                mode.append('adjust')
            elif 'seq' in loss_type:
                mode.append('seq')
            else:
                mode.append('parallel')

            if 'new' in loss_type:
                mode.append('new')

            if 'loss' in loss_type:
                mode.append('loss')

            if 'reverse' in loss_type:
                mode.append('reverse')

            if 'ena' in loss_type:
                mode.append('ena')
            if 'enb' in loss_type:
                mode.append('enb')
            if 'enc' in loss_type:
                mode.append('enc')

            if 'ema' in loss_type:
                mode.append('ema')
            elif 'feature' in loss_type:
                mode.append('feature')
            else:
                raise NotImplementedError

            output = self._rgbd_meta_contrast_init(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )

        elif 'adapt' in loss_type or ('selfsup' in loss_type or 'meta' in loss_type):
            mode = []

            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'adjust' in loss_type:
                mode.append('adjust')
            elif 'seq' in loss_type:
                mode.append('seq')
            else:
                mode.append('parallel')


            mode.append('reverse')

            mode.append('ema')

            output = self._rgbd_meta_contrast(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode
            )
        else:
            mode = []
            if 'bn' in loss_type:
                mode.append('bn')
            output = self._forward(
                image=image,
                sparse_depth=sparse_depth,
                mode=mode)

        # elif 'prepare' in loss_type and 'selfsup' in loss_type:
        #     mode = []

        #     if 'reverse' in loss_type:
        #         mode.append('reverse')

        #     if 'ema' in loss_type:
        #         mode.append('ema')
        #     else:
        #         raise NotImplementedError

        #     output = self._prepare_selfsup_reverse(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # elif 'adapt' in loss_type and 'selfsup' in loss_type:
        #     mode = []

        #     if 'reverse' in loss_type:
        #         mode.append('reverse')

        #     if 'ema' in loss_type:
        #         mode.append('ema')
        #     elif 'feature' in loss_type:
        #         mode.append('feature')
        #     else:
        #         raise NotImplementedError

        #     output = self._adapt_selfsup_reverse(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # elif 'train' in loss_type and 'meta' in loss_type:
        #     mode = []

        #     if 'adjust' in loss_type:
        #         mode.append('adjust')
        #     elif 'seq' in loss_type:
        #         mode.append('seq')
        #     else:
        #         mode.append('parallel')

        #     if 'enc' in loss_type:
        #         mode.append('enc')

        #     output = self._rgbd_meta_train(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # # Meta prepare - prepare MLP parameters
        # elif 'prepare' in loss_type and 'meta' in loss_type:
        #     mode = []

        #     # Define Encoder forward mode
        #     if 'adjust' in loss_type:
        #         mode.append('adjust')
        #     elif 'seq' in loss_type:
        #         mode.append('seq')
        #     else:
        #         mode.append('parallel')

        #     # Define Head forward mode

        #     if 'ema' in loss_type:
        #         mode.append('ema')
        #     output = self._rgbd_meta_prepare(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # # Meta prepare - prepare MLP parameters
        # elif 'adapt' in loss_type and 'meta' in loss_type:
        #     mode = []

        #     # Define Encoder forward mode
        #     if 'adjust' in loss_type:
        #         mode.append('adjust')
        #     elif 'seq' in loss_type:
        #         mode.append('seq')
        #     else:
        #         mode.append('parallel')

        #     # Define Head forward mode
        #     if 'roi' in loss_type:
        #         mode.append('roi')
        #     else:
        #         mode.append('spatial')

        #     output = self._rgbd_meta_adapt(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # # Adapt
        # elif loss_type == 'seq_meta_adapt_meancov_0':
        #     # Define Encoder forward mode
        #     if 'adjust' in loss_type:
        #         mode.append('adjust')
        #     elif 'seq' in loss_type:
        #         mode.append('seq')
        #     else:
        #         mode.append('parallel')

        #     # Define Head forward mode
        #     if 'roi' in loss_type:
        #         mode.append('roi')
        #     else:
        #         mode.append('spatial')
        #     output = self._adapt_meancov_dep(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # elif loss_type == 'seq_meta_adapt_meancov_1':
        #     # Define Encoder forward mode
        #     if 'adjust' in loss_type:
        #         mode.append('adjust')
        #     elif 'seq' in loss_type:
        #         mode.append('seq')
        #     else:
        #         mode.append('parallel')

        #     # Define Head forward mode
        #     if 'roi' in loss_type:
        #         mode.append('roi')
        #     else:
        #         mode.append('spatial')

        #     output = self._adapt_meancov_rgb(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         mode=mode
        #     )

        # elif loss_type == 'adapt':
        #     output = self._adapt(
        #         image=image,
        #         sparse_depth=sparse_depth,
        #         crop_mask=crop_mask)
        # # Calibration
        # elif loss_type == 'get_meanvar':
        #     output = self._rgbd_meanvar_get(
        #         image=image,
        #         sparse_depth=sparse_depth
        #     )

        # elif loss_type == 'get_meanvar_meta':
        #     output = self._rgbd_meanvar_get_meta(
        #         image=image,
        #         sparse_depth=sparse_depth
        #     )

        # Evaluation

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
        '''
        Forwards inputs through the network
        Arg(s):
            image : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            sparse_depth : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        # Encoding
        fe1_rgb = self.conv1_rgb(image)
        fe1_dep = self.conv1_dep(sparse_depth).cuda()

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)

        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)

        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, sparse_depth, image)
        # Remove negative depth
        y = torch.clamp(y, min=0)
        output = y

        if 'bn' in mode and self.training:
            return output, None, None
        else:
            return output

    def _rgbd_meta_contrast(self, image, sparse_depth, mode=[]):
        '''
        Forwards inputs through the network
        Arg(s):
            image : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            sparse_depth : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            mode : list[str]
                Rules forward function structures
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        # Encoding rgb images
        # Forward for different modes:

        fe1_rgb = self.conv1_rgb_meta(self.conv1_rgb(image))
        fe1_dep = self.conv1_dep_meta(self.conv1_dep(sparse_depth))

        # with torch.no_grad():
        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)
        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, sparse_depth, image)
        # Remove negative depth
        y = torch.clamp(y, min=0)

        output_depth = y

        if self.training:
            with torch.no_grad():
                fe1_rgb_nrgb = self.conv1_rgb_meta(self.conv1_rgb(torch.zeros_like(image).cuda()))
                fe1_dep_nrgb = self.conv1_dep_meta(self.conv1_dep(sparse_depth))
                fe1_nrgb = torch.cat((fe1_rgb_nrgb, fe1_dep_nrgb), dim=1)
                fe2_nrgb = self.conv2(fe1_nrgb)
                fe3_nrgb = self.conv3(fe2_nrgb)
                fe4_nrgb = self.conv4(fe3_nrgb)
                fe5_nrgb = self.conv5(fe4_nrgb)
                fe6_nrgb = self.conv6(fe5_nrgb)

            if 'ema' in mode:
                if 'reverse' not in mode:
                    if 'adapt' not in mode:
                        self._update_head()
                        feat_dim = fe6.size(1)
                        proj_rgb = self.pred(self.proj(fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                        proj = self.proj_t(fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                    else:
                        feat_dim = fe6.size(1)
                        proj_rgb = self.pred(self.proj(fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim)))
                        proj = self.proj_t(fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                else:
                    if 'adapt' not in mode:
                        self._update_head()
                        feat_dim = fe6.size(1)
                        proj_rgb = self.pred(self.proj(fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                        proj = self.proj_t(fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                    else:
                        feat_dim = fe6.size(1)
                        proj_rgb = self.pred(self.proj(fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                        proj = self.proj_t(fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim))
            elif 'feature' in mode:
                feat_dim = fe6.size(1)
                proj_rgb = fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim)
                proj = fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim)

            return [output_depth, proj_rgb, proj]
        else:
            return output_depth

    def _rgbd_meta_contrast_init(self, image, sparse_depth, mode='seq'):
        '''
        Forwards inputs through the network
        Arg(s):
            image : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            sparse_depth : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            mode : list[str]
                Rules forward function structures
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        # Encoding rgb images
        # Forward for different modes:
        fe1_rgb_orig = self.conv1_rgb(image)

        if 'new' in mode:
            fe1_rgb = self.meta_bn_rgb(fe1_rgb_orig)
            fe1_rgb_meta = self.conv1_rgb_meta(image)
            fe1_rgb += fe1_rgb_meta
        else:
            fe1_rgb = self.conv1_rgb_meta(fe1_rgb_orig)

        fe1_dep = self.conv1_dep_meta(self.conv1_dep(sparse_depth))

        if 'loss' in mode:
            loss = torch.abs(fe1_rgb - fe1_rgb_orig).mean()

        fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
        fe2 = self.conv2(fe1)
        fe3 = self.conv3(fe2)
        fe4 = self.conv4(fe3)
        fe5 = self.conv5(fe4)
        fe6 = self.conv6(fe5)
        # Shared Decoding
        fd5 = self.dec5(fe6)
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        id_fd1 = self.id_dec1(self._concat(fd2, fe2))
        pred_init = self.id_dec0(self._concat(id_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        # Diffusion
        y, y_inter, offset, aff, aff_const = \
            self.prop_layer(pred_init, guide, confidence, sparse_depth, image)
        # Remove negative depth
        y = torch.clamp(y, min=0)

        output_depth = y
        if 'loss' in mode and self.training:
            return output_depth, loss
        else:
            return output_depth

    def _rgbd_meta_contrast_prepare(self, image, sparse_depth, mode='seq'):
        '''
        Forwards inputs through the network
        Arg(s):
            image : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            sparse_depth : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            mode : list[str]
                Rules forward function structures
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        # Encoding rgb images
        # Forward for different modes:
        with torch.no_grad():
            fe1_rgb = self.conv1_rgb_meta(self.conv1_rgb(image))
            fe1_dep = self.conv1_dep_meta(self.conv1_dep(sparse_depth))
            fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
            fe2 = self.conv2(fe1)
            fe3 = self.conv3(fe2)
            fe4 = self.conv4(fe3)
            fe5 = self.conv5(fe4)
            fe6 = self.conv6(fe5)

            fe1_rgb_nrgb = self.conv1_rgb_meta(self.conv1_rgb(torch.zeros_like(image).cuda()))
            fe1_dep_nrgb = self.conv1_dep_meta(self.conv1_dep(sparse_depth))
            fe1_nrgb = torch.cat((fe1_rgb_nrgb, fe1_dep_nrgb), dim=1)
            fe2_nrgb = self.conv2(fe1_nrgb)
            fe3_nrgb = self.conv3(fe2_nrgb)
            fe4_nrgb = self.conv4(fe3_nrgb)
            fe5_nrgb = self.conv5(fe4_nrgb)
            fe6_nrgb = self.conv6(fe5_nrgb)

        if 'ema' in mode:
            if 'reverse' not in mode:
                self._update_head()
                feat_dim = fe6.size(1)
                proj_rgb = self.pred(self.proj(fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                proj = self.proj_t(fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
            else:
                self._update_head()
                feat_dim = fe6.size(1)
                proj_rgb = self.pred(self.proj(fe6_nrgb.permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                proj = self.proj_t(fe6.permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()

        return None, proj_rgb, proj

    # def _prepare(self, image, sparse_depth, crop_mask):
    #     '''
    #     Forwards inputs through the network
    #     Arg(s):
    #         image : a list of torch.Tensor[float32]
    #             [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
    #         sparse_depth : torch.Tensor[float32]
    #             [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
    #         intrinsics : torch.Tensor[float32]
    #             N x 3 x 3 intrinsic camera calibration matrix
    #     Returns:
    #         torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
    #     '''
    #     cropped = image[1].clone()
    #     # cropped.narrow(1, 0, 1).masked_fill_(crop_mask.narrow(1, 0, 1).bool(), 0)
    #     # cropped.narrow(1, 1, 1).masked_fill_(crop_mask.narrow(1, 0, 1).bool(), 0)
    #     # cropped.narrow(1, 2, 1).masked_fill_(crop_mask.narrow(1, 0, 1).bool(), 0)
    #     # Encoding rgb images

    #     with torch.no_grad():
    #         fe1_rgb = self.conv1_rgb(image[0])
    #         fe1_dep = self.conv1_dep(sparse_depth[0])
    #         fe1_dep_drop = self.conv1_dep(sparse_depth[1])
    #         fe1_rgb_crop = self.conv1_rgb(cropped)

    #         fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
    #         fe2 = self.conv2(fe1)
    #         fe3 = self.conv3(fe2)
    #         fe4 = self.conv4(fe3)
    #         fe5 = self.conv5(fe4)
    #         fe6 = self.conv6(fe5)

    #         B, dim = fe6.size(0), fe6.size(1)

    #         fe6 = fe6.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
    #         proj = self.proj(fe6)

    #         fe1_crop = torch.cat((fe1_rgb_crop, fe1_dep_drop), dim=1)
    #         fe2_crop = self.conv2(fe1_crop)
    #         fe3_crop = self.conv3(fe2_crop)
    #         fe4_crop = self.conv4(fe3_crop)
    #         fe5_crop = self.conv5(fe4_crop)
    #         fe6_crop = self.conv6(fe5_crop)

    #     fe6_crop = fe6_crop.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
    #     proj_crop = self.proj(fe6_crop)
    #     pred = self.pred(proj_crop)
    #     return pred, proj

    # def _rgbd_prepare(self, image, sparse_depth, crop_mask):
    #     '''
    #     Forwards inputs through the network
    #     Arg(s):
    #         image : a list of torch.Tensor[float32]
    #             [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
    #         sparse_depth : torch.Tensor[float32]
    #             [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
    #         intrinsics : torch.Tensor[float32]
    #             N x 3 x 3 intrinsic camera calibration matrix
    #     Returns:
    #         torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
    #     '''
    #     # Encoding rgb images
    #     B = image.size(0)

    #     with torch.no_grad():
    #         # validity_map = torch.where(sparse_depth > 0, 1., 0.)
    #         # validity_map = F.conv2d(validity_map, weight=torch.ones([1, 1, 3, 3]).cuda().detach(), padding=1).bool().view(B, -1).view(-1)
    #         # fe1_rgb = self.conv1_rgb(image)
    #         # fe1_dep = self.conv1_dep(sparse_depth)
    #         # B, dim_rgb = fe1_rgb.size(0), fe1_rgb.size(1)
    #         # dim_dep = fe1_dep.size(1)
    #         # fe1_rgb = fe1_rgb.view(B, dim_rgb, -1)
    #         # fe1_dep = fe1_dep.view(B, dim_dep, -1)
    #         # fe1_rgb = fe1_rgb.transpose(1, 2).reshape(-1, dim_rgb)
    #         # fe1_dep = fe1_dep.transpose(1, 2).reshape(-1, dim_dep)
    #         # fe1_rgb = fe1_rgb[validity_map != 0, :]
    #         # fe1_dep = fe1_dep[validity_map != 0, :]
    #         fe1_rgb = self.conv1_rgb(image)
    #         fe1_dep = self.conv1_dep(sparse_depth)
    #         B, dim_rgb = fe1_rgb.size(0), fe1_rgb.size(1)
    #         dim_dep = fe1_dep.size(1)
    #         fe1_rgb = fe1_rgb.view(B, dim_rgb, -1)
    #         fe1_dep = fe1_dep.view(B, dim_dep, -1)
    #         fe1_rgb = fe1_rgb.transpose(1, 2).reshape(-1, dim_rgb*16)
    #         fe1_dep = fe1_dep.transpose(1, 2).reshape(-1, dim_dep*16)

    #     proj_rgb = self.proj_rgb(fe1_rgb)
    #     proj_rgb = self.proj(proj_rgb)
    #     proj = self.proj(fe1_dep).detach()
    #     return proj_rgb, proj

    # def _rgbd_adapt(self, image, sparse_depth, masked=None):
    #     '''
    #     Forwards inputs through the network
    #     Arg(s):
    #         image : a list of torch.Tensor[float32]
    #             [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
    #         sparse_depth : torch.Tensor[float32]
    #             [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
    #         intrinsics : torch.Tensor[float32]
    #             N x 3 x 3 intrinsic camera calibration matrix
    #     Returns:
    #         torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
    #     '''
    #     # Encoding rgb images

    #     if masked is None:
    #         input_depth = sparse_depth
    #     else:
    #         input_depth = masked

    #     fe1_rgb = self.conv1_rgb(image)
    #     fe1_dep = self.conv1_dep(input_depth)
    #     fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
    #     fe2 = self.conv2(fe1)
    #     fe3 = self.conv3(fe2)
    #     fe4 = self.conv4(fe3)
    #     fe5 = self.conv5(fe4)
    #     fe6 = self.conv6(fe5)

    #     # Shared Decoding
    #     fd5 = self.dec5(fe6)
    #     fd4 = self.dec4(self._concat(fd5, fe5))
    #     fd3 = self.dec3(self._concat(fd4, fe4))
    #     fd2 = self.dec2(self._concat(fd3, fe3))

    #     # Init Depth Decoding
    #     id_fd1 = self.id_dec1(self._concat(fd2, fe2))
    #     pred_init = self.id_dec0(self._concat(id_fd1, fe1))

    #     # Guidance Decoding
    #     gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
    #     guide = self.gd_dec0(self._concat(gd_fd1, fe1))

    #     if self.args.conf_prop:
    #         # Confidence Decoding
    #         cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
    #         confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
    #     else:
    #         confidence = None

    #     # Diffusion
    #     y, y_inter, offset, aff, aff_const = \
    #         self.prop_layer(pred_init, guide, confidence, sparse_depth, image)
    #     # Remove negative depth
    #     y = torch.clamp(y, min=0)

    #     output = y

    #     fe1_dep_rep = self.conv1_dep(sparse_depth).detach()
    #     # with torch.no_grad():
    #     #     B, dim_rgb = fe1_rgb.size(0), fe1_rgb.size(1)
    #     #     # validity_map = torch.where(sparse_depth > 0, 1, 0).bool().view(B, -1).view(-1)
    #     #     validity_map = torch.where(sparse_depth > 0, 1., 0.)
    #     #     validity_map = F.conv2d(validity_map, weight=torch.ones([1, 1, 3, 3]).cuda().detach(), padding=1).bool().view(B, -1).view(-1)
    #     #     dim_dep = fe1_dep.size(1)
    #     #     fe1_rgb = fe1_rgb.view(B, dim_rgb, -1)
    #     #     fe1_dep_rep = fe1_dep_rep.view(B, dim_dep, -1)
    #     #     fe1_rgb = fe1_rgb.transpose(1, 2).reshape(-1, dim_rgb)
    #     #     fe1_dep_rep = fe1_dep_rep.transpose(1, 2).reshape(-1, dim_dep)
    #     #     fe1_rgb = fe1_rgb[validity_map != 0, :]
    #     #     fe1_dep_rep = fe1_dep_rep[validity_map != 0, :].detach()
    #     #     proj_rgb = self.proj_rgb(fe1_rgb)
    #     #     proj_rgb = self.proj(proj_rgb)
    #     #     proj = self.proj(fe1_dep_rep)
    #     with torch.no_grad():
    #         B, dim_rgb = fe1_rgb.size(0), fe1_rgb.size(1)
    #         dim_dep = fe1_dep_rep.size(1)
    #         fe1_rgb = fe1_rgb.reshape(B, dim_rgb, -1).transpose(1, 2).reshape(-1, dim_rgb*16)
    #         fe1_dep_rep = fe1_dep_rep.reshape(B, dim_dep, -1).transpose(1, 2).reshape(-1, dim_dep*16)
    #         proj_rgb = self.proj_rgb(fe1_rgb)
    #         proj_rgb = self.proj(proj_rgb)
    #         proj = self.proj(fe1_dep_rep.detach())

    #     return proj_rgb, proj, output

    # def _adapt(self, image, sparse_depth, crop_mask):
    #     '''
    #     Forwards inputs through the network
    #     Arg(s):
    #         image : a list of torch.Tensor[float32]
    #             [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
    #         sparse_depth : a list of torch.Tensor[float32]
    #             [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
    #         intrinsics : torch.Tensor[float32]
    #             N x 3 x 3 intrinsic camera calibration matrix
    #     Returns:
    #         proj : (NHW) x D projection layer embeddings; torch.Tensor[float32]
    #         pred : (NHW) x D feature embeddings from pred; torch.Tensor[float32]
    #         output : N x 1 x H x W output dense depth map; torch.Tensor[float32]
    #     '''

    #     fe1_rgb = self.conv1_rgb(image[0])
    #     fe1_dep = self.conv1_dep(sparse_depth)
    #     fe1_rgb_crop = self.conv1_rgb(image[1])

    #     fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
    #     fe2 = self.conv2(fe1)
    #     fe3 = self.conv3(fe2)
    #     fe4 = self.conv4(fe3)
    #     fe5 = self.conv5(fe4)
    #     fe6 = self.conv6(fe5)

    #     fe1_crop = torch.cat((fe1_rgb_crop, fe1_dep), dim=1)

    #     fe2_crop = self.conv2(fe1_crop)
    #     fe3_crop = self.conv3(fe2_crop)
    #     fe4_crop = self.conv4(fe3_crop)
    #     fe5_crop = self.conv5(fe4_crop)
    #     fe6_crop = self.conv6(fe5_crop)

    #     # Shared Decoding
    #     fd5 = self.dec5(fe6)
    #     fd4 = self.dec4(self._concat(fd5, fe5))
    #     fd3 = self.dec3(self._concat(fd4, fe4))
    #     fd2 = self.dec2(self._concat(fd3, fe3))

    #     # Init Depth Decoding
    #     id_fd1 = self.id_dec1(self._concat(fd2, fe2))
    #     pred_init = self.id_dec0(self._concat(id_fd1, fe1))

    #     # Guidance Decoding
    #     gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
    #     guide = self.gd_dec0(self._concat(gd_fd1, fe1))

    #     if self.args.conf_prop:
    #         # Confidence Decoding
    #         cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
    #         confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
    #     else:
    #         confidence = None

    #     B, dim = fe6.size(0), fe6.size(1)
    #     with torch.no_grad():
    #         fe6 = fe6.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim).detach()
    #         proj = self.proj(fe6)

    #         fe6_crop = fe6_crop.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim).detach()
    #         proj_crop = self.proj(fe6_crop)
    #         pred = self.pred(proj_crop)

    #     # Diffusion
    #     y, y_inter, offset, aff, aff_const = \
    #         self.prop_layer(pred_init, guide, confidence, sparse_depth, image)
    #     # Remove negative depth
    #     y = torch.clamp(y, min=0)

    #     output = y

    #     return pred, proj.detach(), output

    def _update_head(self, tau=0.999):
        for t, s in zip(self.proj_t.parameters(), self.proj.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def _head_param(self):
        return self.proj.parameters(), self.proj_t.parameters()

    def _head_target_param(self):
        return self.proj.parameters(), self.proj_t.parameters()

    def _load_head_params(self, proj_params, proj_t_params):
        self.proj.parameters(), self.proj_t.parameters()

        for t, s in zip(self.proj.parameters(), proj_params):
            t.data.copy_(s.data)

        for t, s in zip(self.proj_t.parameters(), proj_t_params):
            t.data.copy_(s.data)

    def _prepare_head(self, mode=''):
        '''
        Initialize the projection and prediction heads
        '''

        if 'selfsup' in mode:
            if 'ema' in mode:
                self.proj = self.MLP(512, 1024, 1024).to(torch.device('cuda'))
                self.proj_t = copy.deepcopy(self.proj)
                self.pred = self.MLP(1024, 1024, 1024).to(torch.device('cuda'))
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

        if mode is not None and 'baseline' in mode:
            self.conv1_rgb_baseline = copy.deepcopy(self.conv1_rgb).cuda()
            self.conv1_dep_baseline = copy.deepcopy(self.conv1_dep).cuda()
            self.bn_baseline_rgb = nn.BatchNorm2d(48).cuda()
            self.bn_baseline_dep = nn.BatchNorm2d(16).cuda()

            self.bn_baseline_fe1 = nn.BatchNorm2d(64).cuda()
            self.conv_baseline = conv_bn_relu(64, 64, 3, 1, 1).cuda()
        if mode is not None and 'meta' in mode:
            if 'seq' in mode:
                if 'new' in mode:
                    self.conv1_rgb_meta = conv_bn_relu(3, 48, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.BatchNorm2d(48).cuda()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                elif '1layer' in mode:
                    self.conv1_rgb_meta = nn.Conv2d(48, 48, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()

                elif '2layers' in mode:
                    self.conv1_rgb_meta = Res_Conv(48, 256, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                elif 'resblock' in mode:
                    self.conv1_rgb_meta = BasicBlock(48, 48, 1).cuda()
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

    def MLP(self, dim, projection_size, hidden_size=4096):
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def kl_divergence(self, mu1, mu2, sigma_1, sigma_2):
        sigma_diag_1 = torch.eye(sigma_1.shape[0]).cuda() * sigma_1
        sigma_diag_2 = torch.eye(sigma_2.shape[0]).cuda() * sigma_2

        sigma_diag_2_inv = torch.linalg.inv(sigma_diag_2)

        kl = 0.5 * (torch.log(torch.linalg.det(sigma_diag_2) / torch.linalg.det(sigma_diag_2)) \
            - mu1.shape[0] + torch.trace(torch.matmul(sigma_diag_2_inv, sigma_diag_1)) \
            + torch.matmul(torch.matmul((mu2 - mu1), sigma_diag_2_inv), torch.transpose(mu2 - mu1, 0, 1)))
        return kl

    # def _rgbd_enc_meta_train(self, image, sparse_depth):
    #     '''
    #     Forwards inputs through the network
    #     Arg(s):
    #         image : a list of torch.Tensor[float32]
    #             [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
    #         sparse_depth : torch.Tensor[float32]
    #             [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
    #         intrinsics : torch.Tensor[float32]
    #             N x 3 x 3 intrinsic camera calibration matrix
    #     Returns:
    #         torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
    #     '''
    #     # Encoding rgb images
    #     B = image.size(0)

    #     fe1_rgb = self.conv1_rgb(image)
    #     fe1_rgb_new = self.conv1_rgb_meta(fe1_rgb)

    #     fe1_dep = self.conv1_dep(sparse_depth)
    #     fe1_dep_new = self.conv1_dep_meta(fe1_dep)

    #     fe1 = torch.cat((fe1_rgb_new, fe1_dep_new), dim=1)
    #     fe2 = self.conv2(fe1)
    #     fe3_add = self.conva_meta(fe1)
    #     fe3 = self.conv3(fe2) + fe3_add
    #     fe5_add = self.convb_meta(fe3)
    #     fe4 = self.conv4(fe3)
    #     fe5 = self.conv5(fe4) + fe5_add
    #     fe6 = self.conv6(fe5)

    #     # Shared Decoding
    #     fd4_add = self.deca_meta(fe6)
    #     fd5 = self.dec5(fe6)
    #     fd4 = self.dec4(self._concat(fd5, fe5)) + fd4_add
    #     fd2_add = self.decb_meta(self._concat(fd4, fe4))
    #     fd3 = self.dec3(self._concat(fd4, fe4))
    #     fd2 = self.dec2(self._concat(fd3, fe3)) + fd2_add

    #     # Init Depth Decoding
    #     id_fd1 = self.id_dec1(self._concat(fd2, fe2))
    #     pred_init = self.id_dec0(self._concat(id_fd1, fe1))

    #     # Guidance Decoding
    #     gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
    #     guide = self.gd_dec0(self._concat(gd_fd1, fe1))

    #     if self.args.conf_prop:
    #         # Confidence Decoding
    #         cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
    #         confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
    #     else:
    #         confidence = None

    #     # Diffusion
    #     y, y_inter, offset, aff, aff_const = \
    #         self.prop_layer(pred_init, guide, confidence, sparse_depth, image)
    #     # Remove negative depth
    #     y = torch.clamp(y, min=0)

    #     output = y

    #     l1_consistency_loss = torch.abs(fe1_rgb_new - fe1_rgb).sum() / B

    #     return output, l1_consistency_loss

    # def _rgbd_enc_meta_prepare(self, image, sparse_depth):
    #     '''
    #     Forwards inputs through the network
    #     Arg(s):
    #         image : a list of torch.Tensor[float32]
    #             [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
    #         sparse_depth : torch.Tensor[float32]
    #             [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
    #         intrinsics : torch.Tensor[float32]
    #             N x 3 x 3 intrinsic camera calibration matrix
    #     Returns:
    #         torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
    #     '''
    #     # Encoding rgb images
    #     self._update_head()
    #     B = image[0].size(0)
    #     with torch.no_grad():
    #         fe1_rgb = self.conv1_rgb(image[0])
    #         fe1_rgb_new = self.conv1_rgb_meta(fe1_rgb)

    #         fe1_dep = self.conv1_dep(sparse_depth)
    #         fe1_dep_new = self.conv1_dep_meta(fe1_dep)

    #         fe1 = torch.cat((fe1_rgb_new, fe1_dep_new), dim=1)
    #         fe3_add = self.conva_meta(fe1)
    #         fe2 = self.conv2(fe1)
    #         fe3 = self.conv3(fe2) + fe3_add
    #         fe5_add = self.convb_meta(fe3)
    #         fe4 = self.conv4(fe3)
    #         fe5 = self.conv5(fe4) + fe5_add
    #         fe6 = self.conv6(fe5)

    #         fe1_rgb_crop = self.conv1_rgb(image[1])
    #         fe1_rgb_new_crop = self.conv1_rgb_meta(fe1_rgb_crop)

    #         fe1_dep_crop = self.conv1_dep(sparse_depth)
    #         fe1_dep_new_crop = self.conv1_dep_meta(fe1_dep_crop)

    #         fe1_crop = torch.cat((fe1_rgb_new_crop, fe1_dep_new_crop), dim=1)
    #         fe3_add_crop = self.conva_meta(fe1_crop)
    #         fe2_crop = self.conv2(fe1_crop)
    #         fe3_crop = self.conv3(fe2_crop) + fe3_add_crop
    #         fe5_add_crop = self.convb_meta(fe3_crop)
    #         fe4_crop = self.conv4(fe3_crop)
    #         fe5_crop = self.conv5(fe4_crop) + fe5_add_crop
    #         fe6_crop = self.conv6(fe5_crop)

    #         print(fe6.size())
    #         fe6 = fe6.view(B, 512, -1).transpose(1, 2).view(-1, 512)
    #         fe6_crop = fe6_crop.view(B, 512, -1).transpose(1, 2).view(-1, 512)

    #     emb = self.pred(self.proj(fe6_crop))
    #     ref = self.proj_t(fe6).detach()

    #     return emb, ref
