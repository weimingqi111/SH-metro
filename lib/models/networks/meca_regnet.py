import sys
sys.path.append('.')
import numpy as np
import torch
import torch.nn as nn
from src.lib.models.networks.DCNv2.dcn_v2 import DCN
import math
import os
savepath = 'features'
if not os.path.exists(savepath):
    os.mkdir(savepath)
BN_MOMENTUM = 0.1
__all__ = ['g_ghost_regnetx_002', 'g_ghost_regnetx_004', 'g_ghost_regnetx_006', 'g_ghost_regnetx_008',
           'g_ghost_regnetx_016', 'g_ghost_regnetx_032',
           'g_ghost_regnetx_040', 'g_ghost_regnetx_064', 'g_ghost_regnetx_080', 'g_ghost_regnetx_120',
           'g_ghost_regnetx_160', 'g_ghost_regnetx_320']


# 定义Mish类
class Mish(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class m_eca(nn.Module):
    def __init__(self, channel, k_size=3):
        super(m_eca, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = Mish()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)
import cv2
def draw_features1(self,width, height, x, savename):
        import time
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        tic = time.time()
        sub_output=x
        #plt.imshow(image, alpha=1)
        plt.axis('off')
        mask = np.zeros((width, height))
        b, c, h, w = np.shape(sub_output)
        sub_output = np.transpose(sub_output, [0, 2, 3, 1])[0]
        score = np.max(sigmoid(sub_output[..., 5:]), -1) * sigmoid(sub_output[..., 4])
        score = cv2.resize(score, (width, height))
        normed_score = (score * 255).astype('uint8')
        mask = np.maximum(mask, normed_score)
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(savename, dpi=200)
        print("Save to the " + savename)
        plt.cla()
        print("time:{}".format(time.time() - tic))

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=1,
                 dilation=1, norm_layer=None, k_size=3):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = planes * self.expansion
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, width // min(width, group_width), dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.eca = m_eca(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def fill_up_weights(up):
  w = up.weight.data
  f = math.ceil(w.size(2) / 2)
  c = (2 * f - 1 - f % 2) / (2. * f)
  for i in range(w.size(2)):
    for j in range(w.size(3)):
      w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
  for c in range(1, w.size(0)):
    w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
  for m in layers.modules():
    if isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight, std=0.001)
      # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
      # torch.nn.init.xavier_normal_(m.weight.data)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Stage(nn.Module):

    def __init__(self, block, inplanes, planes, group_width, blocks, stride=1, dilate=False, cheap_ratio=0.5):
        super(Stage, self).__init__()
        norm_layer = nn.BatchNorm2d
        downsample = None
        self.dilation = 1
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )

        self.base = block(inplanes, planes, stride, downsample, group_width,
                          previous_dilation, norm_layer)
        self.end = block(planes, planes, group_width=group_width,
                         dilation=self.dilation,
                         norm_layer=norm_layer)

        group_width = int(group_width * 0.75)
        raw_planes = int(planes * (1 - cheap_ratio) / group_width) * group_width
        cheap_planes = planes - raw_planes
        self.cheap_planes = cheap_planes
        self.raw_planes = raw_planes

        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes + raw_planes * (blocks - 2), cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cheap_planes, cheap_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(cheap_planes, cheap_planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cheap_planes),
        )
        self.cheap_relu = nn.ReLU(inplace=True)

        layers = []
        downsample = nn.Sequential(
            LambdaLayer(lambda x: x[:, :raw_planes])
        )

        layers = []
        layers.append(block(raw_planes, raw_planes, 1, downsample, group_width,
                            self.dilation, norm_layer))
        inplanes = raw_planes
        for _ in range(2, blocks - 1):
            layers.append(block(inplanes, raw_planes, group_width=group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x0 = self.base(input)

        m_list = [x0]
        e = x0[:, :self.raw_planes]
        for l in self.layers:
            e = l(e)
            m_list.append(e)
        m = torch.cat(m_list, 1)
        m = self.merge(m)

        c = x0[:, self.raw_planes:]
        c = self.cheap_relu(self.cheap(c) + m)

        x = torch.cat((e, c), 1)
        x = self.end(x)
        return x


class GGhostRegNet(nn.Module):

    def __init__(self, block, layers, widths, num_classes=1000, zero_init_residual=True,
                 group_width=1, replace_stride_with_dilation=None,
                 norm_layer=None, head_conv=64):
        self.deconv_with_bias = False
        super(GGhostRegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.group_width = group_width
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, widths[0], layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.inplanes = widths[0]                               
        if layers[1] > 2:
            self.layer2 = Stage(block, self.inplanes, widths[1], group_width, layers[1], stride=2,
                          dilate=replace_stride_with_dilation[1], cheap_ratio=0.5) 
        else:      
            self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        
        self.inplanes = widths[1]
        self.layer3 = Stage(block, self.inplanes, widths[2], group_width, layers[2], stride=2,
                      dilate=replace_stride_with_dilation[2], cheap_ratio=0.5)
        
        self.inplanes = widths[2]
        if layers[3] > 2:
            self.layer4 = Stage(block, self.inplanes, widths[3], group_width, layers[3], stride=2,
                          dilate=replace_stride_with_dilation[3], cheap_ratio=0.5) 
        else:
            self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[3])
        
        self.inplanes = widths[3]
        self.deconv_layers = self._make_deconv_layer(3, [256, 128, 64], [4, 4, 4])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if head_conv > 0:
      # heatmap layers
            self.hmap_p = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            #draw_features1(1,1,self.hmap_p.cpu().numpy(),"{}/1.png".format(savepath))
            self.hmap_p[-1].bias.data.fill_(-2.19)
      # regression layers
            self.regs_p = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.w_h_p = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
      # heatmap layers
            self.hmap_h = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.hmap_h[-1].bias.data.fill_(-2.19)
      # regression layers
            self.regs_h = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
            self.w_h_h = nn.Sequential(nn.Conv2d(64, head_conv, kernel_size=3, padding=1, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(head_conv, 2, kernel_size=1, bias=True))
        else:
      # heatmap layers
            self.hmap_p = nn.Conv2d(64, 1, kernel_size=1, bias=True)
      # regression layers
            self.regs_p = nn.Conv2d(64, 2, kernel_size=1, bias=True)
            self.w_h_p = nn.Conv2d(64, 2, kernel_size=1, bias=True)
      # heatmap layers
            self.hmap_h = nn.Conv2d(64, 1, kernel_size=1, bias=True)
      # regression layers
            self.regs_h = nn.Conv2d(64, 2, kernel_size=1, bias=True)
            self.w_h_h = nn.Conv2d(64, 2, kernel_size=1, bias=True)
        fill_fc_weights(self.regs_p)
        fill_fc_weights(self.w_h_p)
        fill_fc_weights(self.regs_h)
        fill_fc_weights(self.w_h_h)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.group_width,
                            previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
               kernel_size=(3, 3), stride=1,
               padding=1, dilation=1, deformable_groups=1)
      # fc = nn.Conv2d(self.inplanes, planes,
      #         kernel_size=3, stride=1,
      #         padding=1, dilation=1, bias=False)
      # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(in_channels=planes,
                              out_channels=planes,
                              kernel_size=kernel,
                              stride=2,
                              padding=padding,
                              output_padding=output_padding,
                              bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        ret = [{'hm_p': self.hmap_p(x), 'wh_p':  self.w_h_p(x), 'reg_p': self.regs_p(x),
                'hm_h': self.hmap_h(x), 'wh_h':  self.w_h_h(x), 'reg_h': self.regs_h(x)}]
        return ret

    def forward(self, x):
        return self._forward_impl(x)


def g_ghost_regnetx_002(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 1, 4, 7], [24, 56, 152, 368], group_width=8, **kwargs)


def g_ghost_regnetx_004(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 2, 7, 12], [32, 64, 160, 384], group_width=16, **kwargs)


def g_ghost_regnetx_006(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 3, 5, 7], [48, 96, 240, 528], group_width=24, **kwargs)


def g_ghost_regnetx_008(**kwargs):
    return GGhostRegNet(Bottleneck, [1, 3, 7, 5], [64, 128, 288, 672], group_width=16, **kwargs)


def g_ghost_regnetx_016(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 4, 10, 2], [72, 168, 408, 912], group_width=24, **kwargs)


def g_ghost_regnetx_032(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 6, 15, 2], [96, 192, 432, 1008], group_width=48, **kwargs)


def g_ghost_regnetx_040(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 5, 14, 2], [80, 240, 560, 1360], group_width=40, **kwargs)


def g_ghost_regnetx_064(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 4, 10, 1], [168, 392, 784, 1624], group_width=56, **kwargs)


def g_ghost_regnetx_080(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 5, 15, 1], [80, 240, 720, 1920], group_width=120, **kwargs)


def g_ghost_regnetx_120(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 5, 11, 1], [224, 448, 896, 2240], group_width=112, **kwargs)


def g_ghost_regnetx_160(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 6, 13, 1], [256, 512, 896, 2048], group_width=128, **kwargs)


def g_ghost_regnetx_320(**kwargs):
    return GGhostRegNet(Bottleneck, [2, 7, 13, 1], [336, 672, 1344, 2520], group_width=168, **kwargs)


if __name__ == '__main__':
    model = g_ghost_regnetx_002()
    model.eval()
    print(model)
    input = torch.randn(32, 3, 512, 512)
    y = model(input)
    print(y)
