"""
    ResNet from 'Deep Residual Learning for Image Recognition', Kaiming He et al, CVPR 2015

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from model.model_utils import register


# resnet variants
__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'wide_resnet50_2', 'wide_resnet101_2'
]

# urls to pretrained resnet models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# define a conv-3x3 layer with padding
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)


# define a conv-1x1 layer -> used often, for simplicity
def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic res-block for resnet

    structure: conv-3x3 -> bn -> relu -> conv-3x3 -> bn -> skip-connect -> relu

    input dimension = batch_size x inplanes x H x H
    output dimension = batch_size x planes x H/stride x H/stride

    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            planes: (int) number of output channels
            stride: (int) stride
            downsample: () downsamples output fmaps -> require dimension matching for skip connection; must set if stride > 1
            groups: (int) BasicBlock only supports default=1
            base_width: (int) BasicBlock only supports default=64
            dilation: (int) dilated convolution; only supports default=1
            norm_layer: (nn.Module) normalization; default=BatchNorm2d

        base_width & groups are interfaces for BottleNeck block, here it is fixed to width=64 and groups=1 for no bottleneck

        """

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock class only supports groups=1 and base_width=64')
        if dilation > 1:
            raise ValueError('BasicBlock class only supports dilation=1')

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.dropout = nn.Dropout(p=0.2)

    def _forward_imp1(self, x):
        """ forward method: no dropout """
        identity = x                                            # batch_size x inplanes x H x H

        out = self.bn1(self.conv1(x))                           # batch_size x planes x H/stride x H/stride -> downsamples if stride > 1
        out = F.relu(out, inplace=True)
        out = self.bn2(self.conv2(out))                         # batch_size x planes x H/stride x H/stride -> maintain dimensions

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                                         # skip connection
        out = F.relu(out, inplace=True)

        return out                                              # batch_size x planes x H/stride x H/stride

    def _forward_imp2(self, x):
        """ forward method: dropout after each conv filter """
        identity = x                                            # batch_size x inplanes x H x H

        out = self.bn1(self.conv1(x))                           # batch_size x planes x H/stride x H/stride -> downsamples if stride > 1
        out = F.relu(out, inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))                         # batch_size x planes x H/stride x H/stride -> maintain dimensions
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                                         # skip connection
        out = F.relu(out, inplace=True)

        return out                                              # batch_size x planes x H/stride x H/stride

    def forward(self, x):
        """ forward method """

        return self._forward_imp2(x)


class BottleNeck(nn.Module):
    """
    Bottle-necked block

    structure: conv-1x1 (decrease width) -> bn/relu -> conv-3x3 (downsample?) -> bn/relu -> conv-1x1 (restore width) -> bn -> skip-connect concat -> relu

    input dimension = batch_size x inplanes x H x H
    output dimension = batch_size x planes*self.expansion x H/stride x H/stride

    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            planes: (int) number of output channels = planes * self.expansion
            stride: (int) stride
            downsample: (nn.Module) downsamples output fmaps -> require dimension matching for skip connection; must set if stride > 1
            groups: (int) number of groups
            base_width: (int) number of channels per group
            dilation: (int) dilated convolution
            norm_layer: (nn.Module) normalization; default=BatchNorm2d

        bottleneck width = planes * base_width / 64 * groups
        - implemened as grouped convolution
        - base_width and groups are used to accomodate implementation of ResNeXt and wide_ResNets
            - for ResNeXt, specify bottleneck width = base_width and cardinality = groups
            - for wide_ResNet, double base_width
            - for vanilla ResNet, set base_width = 64, groups = 1, then bottleneck width = planes -> no bottleneck like BasicBlock
                - can tweak base_width, groups to alter bottleneck widths

        """

        super(BottleNeck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # bottleneck width, implemented from grouped convolution
        # in ResNeXt paper, base_width=4, groups=32, planes=256 -> width=128
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.dropout = nn.Dropout(p=0.2)

    def _forward_imp1(self, x):
        """ forward method for BottleNeck class: no dropout """

        identity = x                                                # batch_size x inplanes x H x H

        out = self.relu(self.bn1(self.conv1(x)))                    # batch_size x width x H x H
        out = self.relu(self.bn2(self.conv2(out)))                  # batch_size x width x H/stride x H/stride
        out = self.bn3(self.conv3(out))                             # batch_size x planes*self.expansion x H/stride x H/stride

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                                             # skip connection
        out = self.relu(out)

        return out                                                  # batch_size x planes*self.expansion x H/stride x H/stride

    def _forward_imp2(self, x):
        """ forward method for BottleNeck class: dropout after each conv filter """

        identity = x                                                # batch_size x inplanes x H x H

        out = self.relu(self.bn1(self.conv1(x)))                    # batch_size x width x H x H
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))                  # batch_size x width x H/stride x H/stride
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))                             # batch_size x planes*self.expansion x H/stride x H/stride
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity                                             # skip connection
        out = self.relu(out)

        return out                                                  # batch_size x planes*self.expansion x H/stride x H/stride

    def forward(self, x):
        """ forward method """

        return self._forward_imp2(x)

class ResNet(nn.Module):
    """
    ResNet

    An abstract network builder class to generate ResNet variants

    Common structure:

        - conv-7x7-s2 -> bn -> relu -> max_pool-3x3-s2
        - res-block (basic / bottleneck) stack: 1
        - res-block (basic / bottleneck) stack: 2
        - res-block (basic / bottleneck) stack: 3
        - res-block (basic / bottleneck) stack: 4
        - global average pool -> flatten -> fc

    """

    def __init__(self, block, stacks, num_classes=10, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """
        Constructor

        Args:
            block: (BasicBlock or BottleNeck class) type of residual block to use as building block for res-stacks
            stacks: (list) a list of 4 integers, each specifying the number of layers (residual blocks) in each of the 4 res-stacks
            num_class: (int) number of final fc layer outputs
            zero_init_residual: (bool) if true initialize the weights of the last BN layer in each residual block to zero
            groups: (int)
            width_per_group: (int)
            replace_stride_with_dilation: (tuple of 3 boolean values) specify whether or not to use dilated conv for
                                residual stack 1~3 (strided stacks)
            norm_layer: (nn.Module) nomalization layer; default = nn.BatchNorm2d

        """

        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None"
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # construt residual stacks
        # 1st stack use no striding, no channel doubling (except block expansion)
        # subsequent stacks use stride=2 in its 1st layer, double channels at final stack output
        # to avoid bottleneck in information flow
        self.stack1 = self._make_stack(block, 64, stacks[0], stride=1)
        self.stack2 = self._make_stack(block, 128, stacks[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.stack3 = self._make_stack(block, 256, stacks[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.stack4 = self._make_stack(block, 512, stacks[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        self.dropout = nn.Dropout(p=0.2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # zero-initiate the last BN layer in each residual block (basic or bottleneck)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_stack(self, block, planes, blocks, stride=1, dilate=False):
        """
        Construct a res-block stack

        Args:
            block: (BasicBlock or BottleNeck class) building block for res-stack
            planes: (int) number of output channels (equal for all layers in the stack) = planes * block.expansion
            blocks: (int) number of building blocks (layers) in the stack
            stride: (int) if > 1, apply striding to (only) the first layer in the stack
            dilate: (bool) if True enable dilation instead of striding

        Structure:
            - 1st layer:
                - if stride > 1, enable striding
                - input = batch_size x self.inplanes x H x H
                - output = batch_size x planes * block.expansion x H/stride x H/stride
                - requires downsampling by conv-1x1
            - subsequent layers:
                - layer_stride =1
                - input = output = batch_size x planes * block.expansion x H/stride x H/stride
                - downsample = None

        """

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        # use dilation instead of striding if true
        if dilate:
            self.dilation *= stride
            stride = 1

        # apply conv-1x1 to input identity if stride > 1 or output channels != input channels for dim. matching
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        # first layer
        # input = batch_size x self.inplanes x H x H
        # output = batch_size x planes * block.expansion x H/stride x H/stride
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # subsequent layers
        for _ in range(1, blocks):
            # input = output = batch_size x planes * block.expansion x H' x H'
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_imp1(self, x):
        """ forward method: no dropout """
                                                                        # batch_size x 3 x H x H -> 32 x 32 on cifar-10
        x = self.bn1(self.conv1(x))                                     # batch_size x 64 x H x H -> 32 x 32

        x = self.stack1(x)                                              # batch_size x 64*block.expansion x H/2 x H/2 -> 16 x 16
        x = self.stack2(x)                                              # batch_size x 128*block.expansion x H/4 x H/4 -> 8 x 8
        x = self.stack3(x)                                              # batch_size x 256*block.expansion x H/8 x H/8 -> 4 x 4

        x = self.avgpool(x)                                             # batch_size x 256*block.expansion x 1 x 1
        x = torch.flatten(x, 1)                                         # batch_size x 256*block.expansion*1*1
        out = self.fc(x)                                                # batch_size x num_classes

        return out

    def _forward_imp2(self, x):
        """ forward method: dropout=0.5 """
                                                                        # batch_size x 3 x H x H -> 32 x 32 on cifar-10
        x = self.bn1(self.conv1(x))                                     # batch_size x 64 x H x H -> 32 x 32
        x = self.dropout(x)

        x = self.stack1(x)                                              # batch_size x 64*block.expansion x H/2 x H/2 -> 16 x 16
        x = self.dropout(x)
        x = self.stack2(x)                                              # batch_size x 128*block.expansion x H/4 x H/4 -> 8 x 8
        x = self.dropout(x)
        x = self.stack3(x)                                              # batch_size x 256*block.expansion x H/8 x H/8 -> 4 x 4
        x = self.dropout(x)

        x = self.avgpool(x)                                             # batch_size x 256*block.expansion x 1 x 1
        x = torch.flatten(x, 1)                                         # batch_size x 256*block.expansion*1*1
        out = self.fc(x)                                                # batch_size x num_classes

        return out

    def forward(self, x):
        """ forward method """
        return self._forward_imp1(x)



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    """
    Abstract model generator interface

    Args:
        arch: (str) architecture of pretrained model
        block: (BasicBlock or BottleNeck) residual block type
        layers: (list) a list of 4 integers each specifying the number of residual blocks for each of the 4 residual stacks
        pretrained: (bool) if true load the weights from pretrained models on ImageNet
        progress: (bool) if true displays a progress bar of the download to stderr
        **kwargs: pointer to additional arguments (e.g., groups, stride, etc.)

    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

@register(name='resnet18')
def resnet18(pretrained=False, progress=True, **kwargs):
    """ ResNet-18 """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

@register(name='resnet34')
def resnet34(pretrained=False, progress=True, **kwargs):
    """ ResNet-34 """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

@register(name='resnet50')
def resnet50(pretrained=False, progress=True, **kwargs):
    """ ResNet-50 """
    return _resnet('resnet50', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)

@register(name='resnet101')
def resnet101(pretrained=False, progress=True, **kwargs):
    """ ResNet-101 """
    return _resnet('resnet101', BottleNeck, [3, 4, 23, 3], pretrained, progress, **kwargs)

@register(name='resnet152')
def resnet152(pretrained=False, progress=True, **kwargs):
    """ ResNet-152 """
    return _resnet('resnet152', BottleNeck, [3, 8, 36, 3], pretrained, progress, **kwargs)

@register(name='resnext50-32x4d')
def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """
    ResNeXt-50 32x4d
    layer = 50
    cardinality = 32
    bottleneck base_width = 4 (width_per_group)
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4', BottleNeck, [3, 4, 6, 4], pretrained, progress, **kwargs)

@register(name='resnext101-32x8d')
def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """
    ResNeXt-101 32x8d
    layer = 101
    cardinality = 32
    bottleneck base_width = 8
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', BottleNeck, [3, 4, 23, 3], pretrained, progress, **kwargs)

@register(name='wide-resnet50-2')
def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    """
    wide ResNet-50-2
    model is the same as ResNet-50, except bottleneck base_width is doubled
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', BottleNeck, [3, 4, 6, 3], pretrained, progress, **kwargs)

@register(name='wide-resnet101-2')
def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    """
    wide ResNet-101-2

    model is the same as ResNet-101, except bottleneck base_width is doubled

    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', BottleNeck, [3, 4, 23, 3], pretrained, progress, **kwargs)
