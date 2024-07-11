"""
Author: Ang Li
Date: 2020-6-14
licensed under the Apache License 2.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
import time

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
        layers.append(nn.LeakyReLU(0.2, inplace=True))

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

class DepthEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(DepthEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)

        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding),
                                  )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x2=None, pre_x3=None, pre_x4=None):
        # input

        x0 = self.init(input)
        if pre_x4 is not None:
            x0 = x0 + F.interpolate(pre_x4, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        if pre_x3 is not None:  # newly added skip connection
            x1 = x1 + F.interpolate(pre_x3, scale_factor=scale, mode='bilinear', align_corners=True)

        x2 = self.enc2(x1)  # 1/4 input size
        if pre_x2 is not None:  # newly added skip connection
            x2 = x2 + F.interpolate(pre_x2, scale_factor=scale, mode='bilinear', align_corners=True)

        return x0, x1, x2


class RGBEncoder(nn.Module):
    def __init__(self, in_layers, layers, filter_size):
        super(RGBEncoder, self).__init__()

        padding = int((filter_size - 1) / 2)
        self.layers = layers
        self.init = nn.Sequential(nn.Conv2d(in_layers, layers, filter_size, stride=1, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding))

        self.enc1 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc2 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc3 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        self.enc4 = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=2, padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers, layers, filter_size, stride=1, padding=padding), )

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, input, scale=2, pre_x=None):
        # input

        x0 = self.init(input)
        if pre_x is not None:
            x0 = x0 + F.interpolate(pre_x, scale_factor=scale, mode='bilinear', align_corners=True)

        x1 = self.enc1(x0)  # 1/2 input size
        x2 = self.enc2(x1)  # 1/4 input size
        x3 = self.enc3(x2)  # 1/8 input size
        x4 = self.enc4(x3)  # 1/16 input size

        return [x0, x1, x2, x3, x4]


class DepthDecoder(nn.Module):
    def __init__(self, layers, filter_size):
        super(DepthDecoder, self).__init__()
        padding = int((filter_size - 1) / 2)

        self.dec2 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.dec1 = nn.Sequential(nn.ReLU(),
                                  nn.ConvTranspose2d(layers // 2, layers // 2, filter_size, stride=2, padding=padding,
                                                     output_padding=padding),
                                  nn.ReLU(),
                                  nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                  )

        self.prdct = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(layers // 2, layers // 2, filter_size, stride=1, padding=padding),
                                   nn.ReLU(),
                                   nn.Conv2d(layers // 2, 1, filter_size, stride=1, padding=padding))

        # Init Weights
        for m in self.modules():
            if isinstance(m, nn.Sequential):
                for p in m:
                    if isinstance(p, nn.Conv2d) or isinstance(p, nn.ConvTranspose2d):
                        nn.init.xavier_normal_(p.weight)
                        nn.init.constant_(p.bias, 0.01)

    def forward(self, pre_dx, pre_cx):

        x2 = pre_dx[2] + pre_cx[2]  # torch.cat((pre_dx[2], pre_cx[2]), 1)
        x1 = pre_dx[1] + pre_cx[1]  # torch.cat((pre_dx[1], pre_cx[1]), 1) #
        x0 = pre_dx[0] + pre_cx[0]

        x3 = self.dec2(x2)  # 1/2 input size
        x4 = self.dec1(x1 + x3)  # 1/1 input size

        # prediction
        output_d = self.prdct(x4 + x0)

        return x2, x3, x4, output_d


class network_adapt(nn.Module):
    def __init__(self, inpainting):
        super(network_adapt, self).__init__()

        denc_layers = 32
        cenc_layers = 32
        ddcd_layers = denc_layers + cenc_layers
        self.inpainting = inpainting
        self.rgb_encoder = RGBEncoder(3, cenc_layers, 3)

        self.depth_encoder1 = DepthEncoder(1, denc_layers, 3)
        self.depth_decoder1 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder2 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder2 = DepthDecoder(ddcd_layers, 3)

        self.depth_encoder3 = DepthEncoder(2, denc_layers, 3)
        self.depth_decoder3 = DepthDecoder(ddcd_layers, 3)

        self.total_time = 0.0
        self.train_time = 0.0
        self.eval_time = 0.0

    def forward(self, image, sparse_depth, crop_mask=None, loss_type=None):
        if 'time' in loss_type:
            torch.cuda.synchronize()
            before_op_time = time.time()

        if loss_type == 'prepare':
            output = self._prepare(image, sparse_depth, crop_mask)
        elif 'init_meta' in loss_type:
            mode = []
            if 'loss' in loss_type:
                mode.append('loss')
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'reverse' in loss_type:
                mode.append('reverse')
            if 'seq' in loss_type:
                mode.append('seq')
            if 'ema' in loss_type:
                mode.append('ema')
            elif 'feature' in loss_type:
                mode.append('feature')
            else:
                raise NotImplementedError
            output = self._rgbd_meta_contrast_init(image, sparse_depth, mode)

        elif 'head' in loss_type:
            mode = []
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'reverse' in loss_type:
                mode.append('reverse')
            if 'seq' in loss_type:
                mode.append('seq')
            if 'ema' in loss_type:
                mode.append('ema')
            elif 'feature' in loss_type:
                mode.append('feature')
            else:
                raise NotImplementedError
            output = self._rgbd_meta_contrast_head(image, sparse_depth, mode)

        elif 'meta' in loss_type and 'selfsup' in loss_type:
            mode = []
            if 'adapt' in loss_type:
                mode.append('adapt')
            if 'reverse' in loss_type:
                mode.append('reverse')
            if 'seq' in loss_type:
                mode.append('seq')
            if 'ema' in loss_type:
                mode.append('ema')
            elif 'feature' in loss_type:
                mode.append('feature')
            else:
                raise NotImplementedError
            output = self._rgbd_meta_contrast(image, sparse_depth, mode)

        # elif loss_type == 'prepare_rgbd':
        #     output = self._rgb_to_d_prepare(image, sparse_depth, crop_mask)
        # elif loss_type == 'prepare_meta':
        #     output = self._meta_prepare(image, sparse_depth)
        # elif loss_type == 'adapt_rgbd':
        #     output = self._rgb_to_d_adapt(image, sparse_depth, crop_mask)
        # elif loss_type == 'adapt':
        #     output = self._adapt(image, sparse_depth)
        elif 'pretrain' in loss_type:
            output = self._forward(image, sparse_depth)
        # else:
        #     raise NotImplementedError

        if 'time' in loss_type:
            torch.cuda.synchronize()
            op_time = time.time() - before_op_time
            if self.training:
                self.train_time += op_time
            else:
                self.eval_time += op_time
            self.total_time += op_time

        return output

    def _forward(self,
                 input_rgb,
                 input_d):
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
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        # for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return [output_d11, output_d12, output_d14]

    def _rgbd_meta_contrast(self,
                               input_rgb,
                               input_d,
                               mode):
        '''
        Forwards inputs through the network
        Arg(s):
            input_rgb : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            input_d : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            mode : list[str]
                forward mode arguments
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        C = (input_d > 0).float()
        if 'seq' in mode:
            enc_c = self.rgb_encoder(input_rgb)
            enc_c[2] = self.conv1_rgb_meta(enc_c[2])
        else:
            enc_c = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        # for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11

        if self.training:
            with torch.no_grad():
                if 'seq' in mode:
                    enc_c_zero = self.rgb_encoder(torch.zeros_like(input_rgb).cuda())
                    enc_c_zero[2] = self.conv1_rgb_meta(enc_c_zero[2])
                else:
                    enc_c_zero = self.rgb_encoder(torch.zeros_like(input_rgb).cuda())
                # for the 1/4 res
                input_d14_zero = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
                enc_d14_zero = self.depth_encoder1(input_d14_zero)
                dcd_d14_zero = self.depth_decoder1(enc_d14_zero, enc_c_zero[2:5])

                # for the 1/2 res
                input_d12_zero = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
                predict_d12_zero = F.interpolate(dcd_d14_zero[3], scale_factor=2, mode='bilinear', align_corners=True)
                input_12_zero = torch.cat((input_d12_zero, predict_d12_zero), 1)

                enc_d12_zero = self.depth_encoder2(input_12_zero, 2, dcd_d14_zero[0], dcd_d14_zero[1], dcd_d14_zero[2])
                dcd_d12_zero = self.depth_decoder2(enc_d12_zero, enc_c_zero[1:4])

                # for the 1/1 res
                predict_d11_zero = F.interpolate(dcd_d12_zero[3] + predict_d12_zero, scale_factor=2, mode='bilinear', align_corners=True)
                input_11_zero = torch.cat((input_d, predict_d11_zero), 1)

                enc_d11_zero = self.depth_encoder3(input_11_zero, 2, dcd_d12_zero[0], dcd_d12_zero[1], dcd_d12_zero[2])

            if 'ema' in mode:
                if 'reverse' not in mode:
                    if 'adapt' not in mode:
                        feat_dim = enc_d11[-1].size(1)
                        self._update_head()
                        proj_rgb = self.pred(self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)))
                        proj = self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                    else:
                        feat_dim = enc_d11[-1].size(1)
                        proj_rgb = self.pred(self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)))
                        proj = self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                else:
                    if 'adapt' not in mode:
                        self._update_head()
                        feat_dim = enc_d11[-1].size(1)
                        proj_rgb = self.pred(self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                        proj = self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                    else:
                        feat_dim = enc_d11[-1].size(1)
                        proj_rgb = self.pred(self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                        proj = self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim))
            return output_d11, proj_rgb, proj
        else:
            return output_d11

    def _rgbd_meta_contrast_init(self,
                               input_rgb,
                               input_d,
                               mode):

        '''
        Forwards inputs through the network
        Arg(s):
            input_rgb : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            input_d : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            mode : list[str]
                forward mode arguments
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        C = (input_d > 0).float()

        with torch.no_grad():
            enc_c = self.rgb_encoder(input_rgb)
        enc_c_origin = enc_c[2]
        enc_c[2] = self.conv1_rgb_meta(enc_c_origin)

        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        # for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11

        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return [output_d11, output_d12, output_d14]

    def _rgbd_meta_contrast_head(self,
                               input_rgb,
                               input_d,
                               mode):
        '''
        Forwards inputs through the network
        Arg(s):
            input_rgb : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            input_d : torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            mode : list[str]
                forward mode arguments
        Returns:
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj and pred
        '''
        with torch.no_grad():

            C = (input_d > 0).float()
            if 'seq' in mode:
                enc_c = self.rgb_encoder(input_rgb)
                enc_c[2] = self.conv1_rgb_meta(enc_c[2])
            else:
                enc_c = self.rgb_encoder(input_rgb)

            # for the 1/4 res
            input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
            enc_d14 = self.depth_encoder1(input_d14)
            dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

            # for the 1/2 res
            input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
            predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
            input_12 = torch.cat((input_d12, predict_d12), 1)

            enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
            dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

            # for the 1/1 res
            predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
            input_11 = torch.cat((input_d, predict_d11), 1)

            enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])

            if 'seq' in mode:
                enc_c_zero = self.rgb_encoder(torch.zeros_like(input_rgb).cuda())
                enc_c_zero[2] = self.conv1_rgb_meta(enc_c_zero[2])
            else:
                enc_c_zero = self.rgb_encoder(torch.zeros_like(input_rgb).cuda())
            # for the 1/4 res
            input_d14_zero = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
            enc_d14_zero = self.depth_encoder1(input_d14_zero)
            dcd_d14_zero = self.depth_decoder1(enc_d14_zero, enc_c_zero[2:5])

            # for the 1/2 res
            input_d12_zero = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
            predict_d12_zero = F.interpolate(dcd_d14_zero[3], scale_factor=2, mode='bilinear', align_corners=True)
            input_12_zero = torch.cat((input_d12_zero, predict_d12_zero), 1)

            enc_d12_zero = self.depth_encoder2(input_12_zero, 2, dcd_d14_zero[0], dcd_d14_zero[1], dcd_d14_zero[2])
            dcd_d12_zero = self.depth_decoder2(enc_d12_zero, enc_c_zero[1:4])

            # for the 1/1 res
            predict_d11_zero = F.interpolate(dcd_d12_zero[3] + predict_d12_zero, scale_factor=2, mode='bilinear', align_corners=True)
            input_11_zero = torch.cat((input_d, predict_d11_zero), 1)

            enc_d11_zero = self.depth_encoder3(input_11_zero, 2, dcd_d12_zero[0], dcd_d12_zero[1], dcd_d12_zero[2])

        if 'ema' in mode:
            if 'reverse' not in mode:
                if 'adapt' not in mode:
                    feat_dim = enc_d11[-1].size(1)
                    self._update_head()
                    proj_rgb = self.pred(self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)))
                    proj = self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                else:
                    feat_dim = enc_d11[-1].size(1)
                    proj_rgb = self.pred(self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)))
                    proj = self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
            else:
                if 'adapt' not in mode:
                    self._update_head()
                    feat_dim = enc_d11[-1].size(1)
                    proj_rgb = self.pred(self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach())
                    proj = self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim)).detach()
                else:
                    feat_dim = enc_d11[-1].size(1)
                    proj_rgb = self.pred(self.proj(enc_d11_zero[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim).detach()))
                    proj = self.proj(enc_d11[-1].permute(0, 2, 3, 1).reshape(-1, feat_dim))
        return None, proj_rgb, proj

    def _update_head(self, tau=0.999):
        for t, s in zip(self.proj_t.parameters(), self.proj.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def _meta_prepare(self,
                 input_rgb,
                 input_d):
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
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)

        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)

        # 0 - 4
        enc_c_meta = enc_c.copy()
        enc_c_meta[1] = self.meta_layer1(enc_c[0]) + self.meta_bn1(enc_c[1])
        enc_c_meta[2] = self.meta_layer1(enc_c[1]) + self.meta_bn1(enc_c[2])
        enc_c_meta[3] = self.meta_layer1(enc_c[2]) + self.meta_bn1(enc_c[3])
        enc_c_meta[4] = self.meta_layer1(enc_c[3]) + self.meta_bn1(enc_c[4])

        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        # for the 1/2 res
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        output_d12 = predict_d11
        output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)

        return [output_d11, output_d12, output_d14]

    def _prepare(self,
                 input_rgb,
                 input_d,
                 crop_mask):
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
        C = (input_d[0] > 0).float()

        with torch.no_grad():
            enc_c = self.rgb_encoder(input_rgb[0])

            # for the 1/4 res
            input_d14 = F.avg_pool2d(input_d[0], 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
            enc_d14 = self.depth_encoder1(input_d14)
            dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

            # for the 1/2 res
            input_d12 = F.avg_pool2d(input_d[0], 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
            predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
            input_12 = torch.cat((input_d12, predict_d12), 1)

            enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
            dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

            # for the 1/1 res
            predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
            input_11 = torch.cat((input_d[0], predict_d11), 1)

            enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])[-1]
            enc_d11 = F.avg_pool2d(enc_d11, 4, 4)
            B, dim = enc_d11.size(0), enc_d11.size(1)
            enc_d11 = enc_d11.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
            ref = self.proj(enc_d11)

        with torch.no_grad():
            enc_c_crop = self.rgb_encoder(input_rgb[1])

            # for the 1/4 res
            input_d14_crop = F.avg_pool2d(input_d[1], 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
            enc_d14_crop = self.depth_encoder1(input_d14_crop)
            dcd_d14_crop = self.depth_decoder1(enc_d14_crop, enc_c_crop[2:5])

            # for the 1/2 res
            input_d12_crop = F.avg_pool2d(input_d[1], 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
            predict_d12_crop = F.interpolate(dcd_d14_crop[3], scale_factor=2, mode='bilinear', align_corners=True)
            input_12_crop = torch.cat((input_d12_crop, predict_d12_crop), 1)

            enc_d12_crop = self.depth_encoder2(input_12_crop, 2, dcd_d14_crop[0], dcd_d14_crop[1], dcd_d14_crop[2])
            dcd_d12_crop = self.depth_decoder2(enc_d12_crop, enc_c_crop[1:4])

            # for the 1/1 res
            predict_d11_crop = F.interpolate(dcd_d12_crop[3] + predict_d12_crop, scale_factor=2, mode='bilinear', align_corners=True)
            input_11_crop = torch.cat((input_d[1], predict_d11_crop), 1)

            enc_d11_crop = self.depth_encoder3(input_11_crop, 2, dcd_d12_crop[0], dcd_d12_crop[1], dcd_d12_crop[2])[-1]
            enc_d11_crop = F.avg_pool2d(enc_d11_crop, 4, 4)

            enc_d11_crop = enc_d11_crop.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)

        enc_d11_crop = self.proj(enc_d11_crop)
        emb = self.pred(enc_d11_crop)

        return emb, ref.detach()

    def _rgb_to_d_prepare(self,
                 input_rgb,
                 input_d,
                 crop_mask):
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
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj (depth reference)
            torch.Tensor[float32] : (NHW) x D feature embeddings from pred (rgb reference)
        '''
        C = (input_d > 0).float()

        with torch.no_grad():
            enc_c = self.rgb_encoder(input_rgb)[-1]
            # enc_c_crop = self.rgb_encoder(input_rgb[1])
            # for the 1/4 res
            input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
            enc_d14 = self.depth_encoder1(input_d14)[-1]

            # dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])
            # for the 1/2 ress
            # input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
            # predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
            # input_12 = torch.cat((input_d12, predict_d12), 1)

            # enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
            # dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

            # # for the 1/1 res
            # predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
            # input_11 = torch.cat((input_d, predict_d11), 1)

            # enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
            # enc_d11 = F.avg_pool2d(enc_d11, 4, 4)
            # B, dim = enc_d11.size(0), enc_d11.size(1)
            # enc_d11 = enc_d11.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
            # dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])
            # output_d11 = dcd_d11[3] + predict_d11
            # output_d12 = predict_d11
            # output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)
            B, dim = enc_c.size(0), enc_c.size(1)
            enc_c = enc_c.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim*16)
            enc_d14 = enc_d14.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim*16)
            depth_feat = self.pred(enc_d14)

        rgb_feat = self.pred(self.proj(enc_c))

        return rgb_feat, depth_feat.detach()

    def _rgb_to_d_adapt(self,
                 input_rgb,
                 input_d,
                 crop_mask):
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
            torch.Tensor[float32] : (NHW) x D feature embeddings from proj (depth reference)
            torch.Tensor[float32] : (NHW) x D feature embeddings from pred (rgb reference)
        '''
        C = (input_d > 0).float()

        enc_c = self.rgb_encoder(input_rgb)
        # enc_c_crop = self.rgb_encoder(input_rgb[1])
        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d, 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)[-1]
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])
        # for the 1/2 ress
        input_d12 = F.avg_pool2d(input_d, 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d, predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11
        # output_d12 = predict_d11
        # output_d14 = F.interpolate(dcd_d14[3], scale_factor=4, mode='bilinear', align_corners=True)
        with torch.no_grad():
            B, dim = enc_c.size(0), enc_c.size(1)
            enc_c_feat = enc_c[-1].reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
            enc_d14_feat = enc_d14[-1].reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
            depth_feat = self.proj(enc_d14_feat)
            rgb_feat = self.pred(self.proj(enc_c_feat))

        return rgb_feat, depth_feat.detach(), output_d11

    def _adapt(self,
                 input_rgb,
                 input_d):
        '''
        Forwards inputs through the network
        Arg(s):
            image : a list of torch.Tensor[float32]
                [ N x 3 x H x W, N x 3 x H x W ] with original/masked images
            sparse_depth : a list of torch.Tensor[float32]
                [ N x 1 x H x W, N x 1 x H x W ] projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32]
                (NHW) x D projection layer embeddings
            torch.Tensor[float32]
                (NHW) x D prediction layer embeddings
            torch.Tensor[float32]
                N x 1 x H x W output dense depth map
        '''
        C = (input_d[0] > 0).float()

        enc_c = self.rgb_encoder(input_rgb[0])

        # for the 1/4 res
        input_d14 = F.avg_pool2d(input_d[0], 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14 = self.depth_encoder1(input_d14)
        dcd_d14 = self.depth_decoder1(enc_d14, enc_c[2:5])

        # for the 1/2 res
        input_d12 = F.avg_pool2d(input_d[0], 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12 = F.interpolate(dcd_d14[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12 = torch.cat((input_d12, predict_d12), 1)

        enc_d12 = self.depth_encoder2(input_12, 2, dcd_d14[0], dcd_d14[1], dcd_d14[2])
        dcd_d12 = self.depth_decoder2(enc_d12, enc_c[1:4])

        # for the 1/1 res
        predict_d11 = F.interpolate(dcd_d12[3] + predict_d12, scale_factor=2, mode='bilinear', align_corners=True)
        input_11 = torch.cat((input_d[0], predict_d11), 1)

        enc_d11 = self.depth_encoder3(input_11, 2, dcd_d12[0], dcd_d12[1], dcd_d12[2])
        dcd_d11 = self.depth_decoder3(enc_d11, enc_c[0:3])

        output_d11 = dcd_d11[3] + predict_d11

        with torch.no_grad():
            enc_d11 = F.avg_pool2d(enc_d11[-1].clone().detach(), 4, 4)
            B, dim = enc_d11.size(0), enc_d11.size(1)
            enc_d11 = enc_d11.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)
            ref = self.proj(enc_d11)

        enc_c_crop = self.rgb_encoder(input_rgb[1])

        # for the 1/4 res
        input_d14_crop = F.avg_pool2d(input_d[1], 4, 4) / (F.avg_pool2d(C, 4, 4) + 0.0001)
        enc_d14_crop = self.depth_encoder1(input_d14_crop)
        dcd_d14_crop = self.depth_decoder1(enc_d14_crop, enc_c_crop[2:5])

        # for the 1/2 res
        input_d12_crop = F.avg_pool2d(input_d[1], 2, 2) / (F.avg_pool2d(C, 2, 2) + 0.0001)
        predict_d12_crop = F.interpolate(dcd_d14_crop[3], scale_factor=2, mode='bilinear', align_corners=True)
        input_12_crop = torch.cat((input_d12_crop, predict_d12_crop), 1)

        enc_d12_crop = self.depth_encoder2(input_12_crop, 2, dcd_d14_crop[0], dcd_d14_crop[1], dcd_d14_crop[2])
        dcd_d12_crop = self.depth_decoder2(enc_d12_crop, enc_c_crop[1:4])

        # for the 1/1 res
        predict_d11_crop = F.interpolate(dcd_d12_crop[3] + predict_d12_crop, scale_factor=2, mode='bilinear', align_corners=True)
        input_11_crop = torch.cat((input_d[1], predict_d11_crop), 1)

        enc_d11_crop = self.depth_encoder3(input_11_crop, 2, dcd_d12_crop[0], dcd_d12_crop[1], dcd_d12_crop[2])

        with torch.no_grad():
            enc_d11_crop = F.avg_pool2d(enc_d11_crop[-1], 4, 4)
            enc_d11_crop = enc_d11_crop.reshape(B, dim, -1).transpose(1, 2).reshape(-1, dim)

            enc_d11_crop = self.proj(enc_d11_crop)
            emb = self.pred(enc_d11_crop)

        return emb, ref.detach(), output_d11

    def _prepare_head(self, mode=''):
        '''
        Initialize the self-supervised MLP heads (projection, prediction layers)
        Inputs:
            mode: str
                Rules the training mode, choices: [None, ts]
        '''
        layer_dim = self.rgb_encoder.layers
        print(layer_dim)
        if 'selfsup' in mode:
            if 'ema' in mode:
                import copy
                self.proj = self.MLP(32, 512, 512).to(torch.device('cuda'))
                self.proj_t = copy.deepcopy(self.proj)
                self.pred = self.MLP(512, 512, 512).to(torch.device('cuda'))
            else:
                self.proj = self.MLP(32, 512, 512).to(torch.device('cuda'))
                self.pred = self.pred = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 512),
                    nn.BatchNorm1d(512)).to(torch.device('cuda'))
        elif mode == 'rgbd':
            self.proj = self.MLP(layer_dim*16, layer_dim*16, layer_dim*32).cuda()
            # Simsiam Pred
            self.pred = nn.Sequential(
                nn.Linear(layer_dim*16, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024)
            ).cuda()
        else:
            self.proj_t = None
        if mode is not None and 'meta' in mode:
            if 'seq' in mode:
                if '1layer' in mode:
                    self.conv1_rgb_meta = nn.Conv2d(32, 32, 3, 1, 1).cuda()

                    nn.init.kaiming_normal_(self.conv1_rgb_meta.weight, mode='fan_out', nonlinearity='relu')
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                    # import torchvision.models.resnet
                elif '2layers' in mode:
                    self.conv1_rgb_meta = Res_Conv(32, 128, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                elif 'resblock' in mode:
                    self.conv1_rgb_meta = BasicBlock(32, 32, 1).cuda()
                    self.meta_bn_rgb = nn.Identity()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()
                elif '1conv' in mode:
                    self.conv1_rgb_meta = nn.Conv2d(32, 32, 3, 1, 1).cuda()
                    self.meta_bn_rgb = nn.BatchNorm2d(32).cuda()
                    self.conv1_dep_meta = nn.Identity()
                    self.meta_bn_dep = nn.Identity()

    def MLP(self, dim, projection_size, hidden_size=4096):
        '''
        Return projection layer
        '''
        return nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
