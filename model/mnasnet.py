"""
    MnasNet from the paper
    "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
    by Mingxing Tan & Quoc V.Le, Google, CVPR 2019
"""

# imports
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# mnasnet variants
__all__ = [
    'MnasNet', 'mnasneta1',
    'mnasneta1_3x3mbconv3', 'mnasneta1_3x3mbconv3se',
    'mnasneta1_5x5mbconv3', 'mnasneta1_5x5mbconv3se',
    'mnasneta1_3x3mbconv6', 'mnasneta1_3x3mbconv6se',
    'mnasneta1_5x5mbconv6', 'mnasneta1_5x5mbconv6se',
]

# pretrained models
model_urls = {

}

# 5x5-conv filter
def conv5x5(in_planes, out_planes, stride=1, groups=1):
    """
    5x5-conv filter
    - preserves fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
    - requires padding=2, dilation=1, kernel_size=5
    - becomes depthwise convolution when in_planes = out_planes = groups
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, groups=groups,
                     padding=2, dilation=1, bias=False)


# 3x3-conv filter
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """
    3x3-conv filter
    - preserves fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
    - requires padding=dilation=1, kernel_size=3
    - becomes depthwise convolution when in_planes = out_planes = groups
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=1, dilation=1, bias=False)

# 1x1-conv filter
def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """
    1x1-conv filter
    - preserves fmap dimensions if stride=1
    - exactly halves fmap dimensions if stride=2
    - requires padding=0, dilation=arbitray, kernel_size=1
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups,
                     padding=0, dilation=1, bias=False)


class DWSepConv(nn.Module):
    """
    depthwise-separable convolution block

    structure:
    - conv-dw > bn > relu > 1x1-conv > bn

    notes:
    - kernel_size of conv-dw should be parametried, could be 3x3 or 5x5
    - output channels = input channels * stride
    """

    def __init__(self, inplanes, kernel_size, stride=1, dropout=0, norm_layer=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels to the block
            kernel_size: (int) kernel_size of conv-dw filter, either 3x3 or 5x5 is supported
            stride: (int) stride of conv-dw filter
            dropout: (float) p = dropout; default = 0 (no dropout effect)
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d

        """
        super(DWSepConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if kernel_size == 3:
            conv_dw = conv3x3
        elif kernel_size == 5:
            conv_dw = conv5x5
        else:
            raise ValueError("DWSepConv class only supports kernel size 3x3, 5x5")

        self._outplanes = inplanes * stride

        self.convdw1 = conv_dw(inplanes, inplanes, stride=stride, groups=inplanes)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv1x1(inplanes, self._outplanes, stride=1)
        self.bn2 = norm_layer(self._outplanes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """ forward method """

        x = self.relu(self.bn1(self.convdw1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))

        return x


class SEopt(nn.Module):
    """
    squeeze-and-excitation optimization layer

    structure:
    - global pooling > fc > relu > fc > sigmoid > skip connect: scale

    notes:
    - global pooling = AdaptiveAvgPool((1,1)) -> reduces fmap dimensions to 1x1
    - reduction ratio
        - 1st fc squeeze channels to c/r, where r = reduction ratio
        - 2nd fc recovers channels to c
    - scale
        - there is a skip connection from input to sigmoid output
        - then scales the input fmaps (hxhxc) by multiplying with the sigmoid output (1x1xc)
          for each activation location in channel-wise manner

    """

    def __init__(self, inplanes, reduction=8, nonlinearity=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            reduction: (int) reduction ratio
            nonlinearity: (nn.Module) non-linearity used in SE module; default = nn.sigmoid
        """

        super(SEopt, self).__init__()

        if nonlinearity is None:
            nonlinearity = nn.Sigmoid

        self._reduced_planes = int(inplanes / reduction)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(inplanes, self._reduced_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self._reduced_planes, inplanes)
        self.sigmoid = nonlinearity()

    def forward(self, x):
        """ forward method """

        identity = x                                # batch_size x C x H x H

        s = self.avgpool(x)                         # batch_size x C x 1 x 1
        s = torch.flatten(s, 1)                     # batch_size x C*1*1
        s = self.fc1(s)                             # batch_size x C/r*1*1
        s = self.relu(s)                            # batch_size x C/r*1*1
        s = self.fc2(s)                             # batch_size x C*1*1
        s = self.sigmoid(s)                         # batch_size x C*1*1

        s = s.view(s.shape[0], s.shape[1], 1, 1)    # batch_size x C x 1 x 1
        out = identity * s

        return out



class MBConv(nn.Module):
    """
    mobile-inverted-bottleneck block

    structure:
    - 1x1-conv > bn > relu > 3x3/5x5-conv-dw > bn > relu > (optional) SE > 1x1-conv > bn

    options:
    - SE optimization
    - kernel_sizes: 3x3, 5x5
    - expansion factor > 1

    notes:
    - input: H x H x K; bottleneck: H/s x H/s x tK; output: H/s x H/s x K'
        - K: input channels; K': output channels; t: expansion factor; s: stride
    """

    def __init__(self, inplanes, outplanes, expansion=1, kernel_size=3, stride=1, dropout=0,
                 SE=False, downsample=None, norm_layer=None, act_layer=None):
        """
        Constructor

        Args:
            inplanes: (int) number of input channels
            outplanes: (int) number of output channels
            expansion: (int) expansion factor for inverted residuals
            kernel_size: (int) 3x3 or 5x5 conv-dw filter
            stride: (int) stride for conv-dw filter
            dropout: (float) p = dropout; default = 0 (no dropout effect)
            SE: (bool) if True add SE optimization layer after conv-dw layer
            downsample: (nn.Module) downsamples input tensors for skip connection if stride > 1
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            act_layer: (nn.Module) activation layer; default = nn.ReLU6
        """

        super(MBConv, self).__init__()

        # default normalization
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # default activation
        if act_layer is None:
            act_layer = nn.ReLU6
        # conv-dw kernel
        if kernel_size == 3:
            conv_dw = conv3x3
        elif kernel_size == 5:
            conv_dw = conv5x5
        else:
            raise ValueError("MBConv class only supports kernel size 3x3, 5x5")

        self._bottleneck_width = int(inplanes * expansion)
        self._SE_enabled = SE

        self.conv1 = conv1x1(inplanes, self._bottleneck_width, stride=1)
        self.bn1 = norm_layer(self._bottleneck_width)
        self.convdw2 = conv_dw(self._bottleneck_width, self._bottleneck_width, stride=stride,
                               groups=self._bottleneck_width)
        self.bn2 = norm_layer(self._bottleneck_width)

        if self._SE_enabled:
            self.seopt = SEopt(self._bottleneck_width)

        self.conv3 = conv1x1(self._bottleneck_width, outplanes, stride=1)
        self.bn3 = norm_layer(outplanes)

        self.relu = act_layer(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.downsample = downsample


    def forward(self, x):
        """ forward method """

        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.convdw2(out)))
        out = self.dropout(out)
        if self._SE_enabled:
            out = self.seopt(out)
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)

        out += identity

        return out



class MnasNet(nn.Module):
    """
    MnasNet

    Structure:

    > ImageNet
    - 3x3-conv s=2
    - 3x3-DWSepConv s=1
    - 3x3-MBConv-t6 x2 (s=2 first block)
    - 5x5-MBConv-t3-SE x3 (s=2 first block)
    - 3x3-MBConv-t6 x4 (s=2 first block)
    - 3x3-MBConv-t6-SE x2 s=1
    - 5x5-MBConv-t6-SE x3 (s=2 first block)
    - 3x3-MBConv-t6 x1 s=1
    - global pooling > fc

    > Cifar-10/100
    - 3x3-conv s=1
    - 3x3-DWSepConv s=1
    - 3x3-MBConv-t6 x2 (s=2 first block)
    - 5x5-MBConv-t3-SE x3 (s=2 first block)
    - 3x3-MBConv-t6 x4 s=1
    - 3x3-MBConv-t6-SE x2 s=1
    - 5x5-MBConv-t6-SE x3 (s=2 first block)
    - 3x3-MBConv-t6 x1 s=1
    - global pooling > fc
    """

    def __init__(self, block, layers, SE, kernel_sizes, expansions, num_classes=10,
                 dropout=0, norm_layer=None, act_layer=None):
        """
        Constructor

        Args:
            block: (nn.Module) building block; default = MBConv
            layers: (list of int) list of integers specifying number of layers for each stack
            SE: (list of bool) list of boolean values specifying whether to apply SE optimization for each stack
            kernel_sizes: (list of int) list of integers (3 or 5) specifying kernel size for each stack
            expansions: (list of int) list of expansion factors for each MBConv stack
            num_classes: (int) number of classes in dataset = number of output channels of network
            dropout: (float) p = dropout; default = 0 (no dropout effect)
            norm_layer: (nn.Module) normalization layer; default = nn.BatchNorm2d
            act_layer: (nn.Module) activation layer; default = nn.ReLU6
        """

        super(MnasNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        if act_layer is None:
            act_layer = nn.ReLU6
            self._act_layer = act_layer

        self._dropout = dropout

        self.conv1 = conv3x3(3, 16, stride=1)
        self.bn1 = norm_layer(16)
        self.stack1 = DWSepConv(16, kernel_size=3, stride=1)

        self.stack2 = self._make_stack(block=block, num_layers=layers[0], inplanes=16, outplanes=24,
                                       kernel_size=kernel_sizes[0], SE=SE[0], expansion=expansions[0],
                                       stride=2)

        self.stack3 = self._make_stack(block=block, num_layers=layers[1], inplanes=24, outplanes=40,
                                       kernel_size=kernel_sizes[1], SE=SE[1], expansion=expansions[1],
                                       stride=2)

        self.stack4 = self._make_stack(block=block, num_layers=layers[2], inplanes=40, outplanes=80,
                                       kernel_size=kernel_sizes[2], SE=SE[2], expansion=expansions[2],
                                       stride=1)
                                       
        self.stack5 = self._make_stack(block=block, num_layers=layers[3], inplanes=80, outplanes=112,
                                       kernel_size=kernel_sizes[3], SE=SE[3], expansion=expansions[3],
                                       stride=1)

        self.stack6 = self._make_stack(block=block, num_layers=layers[4], inplanes=112, outplanes=160,
                                       kernel_size=kernel_sizes[4], SE=SE[4], expansion=expansions[4],
                                       stride=2)

        self.stack7 = self._make_stack(block=block, num_layers=layers[5], inplanes=160, outplanes=320,
                                       kernel_size=kernel_sizes[5], SE=SE[5], expansion=expansions[5],
                                       stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(320, num_classes)

        self.relu = self._act_layer(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_stack(self, block, num_layers, inplanes, outplanes, kernel_size=3,
                    SE=False, expansion=3, stride=1):
        """
        build a stack of blocks

        Args:
            block: (nn.Module) building block; default = MBConv
            num_layers: (int) number of layers in the stack
            inplanes: (int) number of input channels to the stack
            outplanes: (int) number of output channels to the stack
            kernel_size: (int) filter size = 3x3 or 5x5
            SE: (bool) if True apply the SE optimization layer
            expansion: (int) expansion factor for MBConv block
            stride: (int) stride for first block in the stack; for other layers stride=1
        """

        norm_layer = self._norm_layer
        act_layer = self._act_layer
        downsample = None

        # if stride > 1
        # or if block input planes != block output planes (only possible for first block in stack)
        # downsamples skip connection by 1x1-conv filter
        if stride != 1 or inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride=stride),
                norm_layer(outplanes)
            )

        layers = []

        # first block in stack can have stride > 1
        layers.append(block(inplanes, outplanes, expansion=expansion, kernel_size=kernel_size,
                            SE=SE, stride=stride, dropout=self._dropout, downsample=downsample,
                            norm_layer=norm_layer, act_layer=act_layer))

        # other layers in stack
        # for each layer: inplanes = outplanes, stride=1, downsample=None
        for _ in range(1, num_layers):
            layers.append(block(outplanes, outplanes, expansion=expansion, kernel_size=kernel_size,
                          SE=SE, stride=1, dropout=self._dropout, norm_layer=norm_layer,
                          act_layer=act_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        """ foward method """

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.stack4(out)
        out = self.stack5(out)
        out = self.stack6(out)
        out = self.stack7(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out



def _mnasnet(arch, block, layers, expansions, kernel_sizes, SE, dropout=0,
             pretrained=False, progress=False, **kwargs):
    """
    Abstractor generator to build mnasnet variants

    Args:
        arch: (str) architecture of the pretrained model
        block: (nn.Module) building block for network; default=MBConv
        layers: (list of int) list of integers specifying number of layers per stack
        expansions: (list of int) list of integers specifying expansion factor per stack
        kernel_sizes: (list of int) list of integers specifying the filter sizes per stack
        SE: (list of bool) list of boolean values; if True, enables SE optimization for the stack
        dropout: (float) p = dropout; default = 0 (no dropout effect)
        pretrained: (bool) if True, download pretrained models
        progress: (bool) if True, display download progress of pretrained models
        **kwargs: pointer to additional parameters (e.g., norm_layer, act_layer)
    """
    model = MnasNet(block, layers=layers, expansions=expansions, kernel_sizes=kernel_sizes,
                    SE=SE, dropout=dropout, **kwargs)
    if pretrained:
        if arch in model_urls.keys():
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
    return model


def mnasneta1(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1

    structure:
    - 3x3-MBConv-6
    - 5x5-MBConv-3-SE
    - 3x3-MBConv-6
    - 3x3-MBConv-6-SE
    - 5x5-MBConv-6-SE
    - 3x3-MBConv-6
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[6, 3, 6, 6, 6, 6],
                    kernel_sizes=[3, 5, 3, 3, 5, 3], SE=[False, True, False, True, True, False],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_3x3mbconv3(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 3x3-MBconv-3 block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[3, 3, 3, 3, 3, 3],
                    kernel_sizes=[3, 3, 3, 3, 3, 3], SE=[False, False, False, False, False, False],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_3x3mbconv3se(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 3x3-MBconv-3-SE block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[3, 3, 3, 3, 3, 3],
                    kernel_sizes=[3, 3, 3, 3, 3, 3], SE=[True, True, True, True, True, True],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_5x5mbconv3(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 5x5-MBconv-3 block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[3, 3, 3, 3, 3, 3],
                    kernel_sizes=[5, 5, 5, 5, 5, 5], SE=[False, False, False, False, False, False],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_5x5mbconv3se(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 5x5-MBconv-3-SE block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[3, 3, 3, 3, 3, 3],
                    kernel_sizes=[5, 5, 5, 5, 5, 5], SE=[True, True, True, True, True, True],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_3x3mbconv6(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 3x3-MBconv-6 block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[6, 6, 6, 6, 6, 6],
                    kernel_sizes=[3, 3, 3, 3, 3, 3], SE=[False, False, False, False, False, False],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_3x3mbconv6se(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 3x3-MBconv-6-SE block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[6, 6, 6, 6, 6, 6],
                    kernel_sizes=[3, 3, 3, 3, 3, 3], SE=[True, True, True, True, True, True],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_5x5mbconv6(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 5x5-MBconv-6 block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[6, 6, 6, 6, 6, 6],
                    kernel_sizes=[5, 5, 5, 5, 5, 5], SE=[False, False, False, False, False, False],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)


def mnasneta1_5x5mbconv6se(pretrained=False, progress=False, **kwargs):
    """
    mnasneta1 w.t. 5x5-MBconv-6-SE block only
    """
    return _mnasnet('mnasneta1', MBConv, layers=[2, 3, 4, 2, 3, 1], expansions=[6, 6, 6, 6, 6, 6],
                    kernel_sizes=[5, 5, 5, 5, 5, 5], SE=[True, True, True, True, True, True],
                    dropout=0, pretrained=pretrained, progress=progress, **kwargs)
