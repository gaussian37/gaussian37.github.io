---
layout: post
title: Dilated Residual Network for Image Classification and Semantic Segmentation
date: 2020-01-30 00:00:00
img: dl/concept/dilated_residual_network/0.png
categories: [dl-concept]
tags: [dilated residual network, DRN] # add tag
---

<br>

- 참조 : https://towardsdatascience.com/review-drn-dilated-residual-networks-image-classification-semantic-segmentation-d527e1a8fb5
- 이번 글에서는 `Dilated Residual Network`에 대하여 다루어 보려고 합니다. 
- 기존에 `Residual Network`에 `Dilated Convolution`을 접목한 형태의 딥러닝 네트워크입니다. 

<br>

## **목차**

<br>

- ### Dilated Convolution 이란
- ### Dilated Convolution 적용 이유
- ### Dilated Residual Network와 Pytorch 코드

<br>

## **Dilated Convolution 이란**

<br>
<center><img src="../assets/img/dl/concept/dilated_residual_network/1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 이미지를 보면 왼쪽이 일반적인 convolution 연산이고 오른쪽이 dilated convolution 연산입니다.
- 위 이미지의 파란색이 입력이고 초록색이 출력인데 오른쪽의 dilated convolution을 보면 왼쪽의 일반적인 convolution과 비교하였을 때, 필터간의 간격이 있는 것을 확인할 수 있습니다.
- dilation에 적용된 상수를 통해서 표현하면 왼쪽의 일반적인 convolution은 1이고 오른쪽의 dilated convolution은 2가 됩니다. 필터에서의 간격에 해당합니다.

<br>
<center><img src="../assets/img/dl/concept/dilated_residual_network/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- dilated convolution을 적용하면 소위 말하는 `receptive field`가 넓어지게 됩니다.
- `receptive field`는 convolution filter가 커버하는 영역이라고 생각하면 됩니다. 말 그대로 필터가 받아들이는 영역입니다.

<br>

## **Dilated Convolution 적용 이유**

<br>

- 기존의 Residual Network에 Dilated Convolution을 적용한 이유는 무엇일까요?
- Semantic Segmentation에서 output의 성능을 높이려면 큰 output feature map이 필요합니다. 반대로 말하면 Semantic Segmentation의 output에서 output feature map의 크기가 작으면 accuracy가 감소합니다.
- [FCN](https://gaussian37.github.io/vision-segmentation-fcn/)에서 `32x upsampling`만 하게 되면 성능이 좋지 못한 segmentation 결과를 얻을 수도 있는데, 이런 이유로 `16x upsampling` 또는 `8x upsampling` 등을 적용하여 좀 더 큰 (resolution이 높은) output feature map을 얻게 됩니다.
- 좀 더 큰 output feature map을 얻기 위해서 Segmentation을 위한 Encoder에서 단순히 subsampling을 제거하는 방법도 사용할 수 있지만, 이렇게 하면 receptive field가 감소하게 되고 그 결과 네트워크의 이미지 context를 이해 능력이 떨어질 수 있습니다. 즉 성능이 떨어지게 됩니다.
- 이러한 이유로 `dilated convolution`을 적용하여 receptive field를 늘릴 수 있습니다.
- 또한 Segmentation 성능을 높이기 위하여 사용된 `dilated convolution`이 image classification을 할 때에도 더 좋은 성능을 내었다고 알려져 있습니다.

<br>

## **Dilated Residual Network와 Pytorch 코드**

<br>

- Pytorch 코드를 살펴보면서 Dilated Convolution이 어떻게 적용되는 지 살펴보겠습니다.

```python
import math
import torch.utils.model_zoo as model_zoo
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    '''
    - 입력 : 입력 채녈 수, 출력 채널 수, stride
    - 출력 : convolution 필터를 적용한 feature
    - 3x3 필터를 사용하는 기본적인 convolution 필터 함수
    - #filter = 3x3, #padding = 1로 고정
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    '''
    ResNet BasicBlock
    - 입력 : in_planes(입력 채널 수), out_planes(출력 채널 수), stride, dilation, downsample, previous_dilation
    - 출력 : BasicBlock 객체 
    - Convolution - BatchNorm - ReLU 2번을 하면서 skip connection을 만든다. 필요 시 downsample도 수행함
    '''
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, downsample=None, previous_dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''
    ResNet Bottleneck
    - 입력 : in_planes(입력 채널 수), out_planes(출력 채널 수), stride, dilation, downsample, previous_dilation
    - 출력 : Bottleneck 객체 
    - Convolution - BatchNorm - ReLU 3번을 하면서 Bottleneck 구조와 skip connection을 만든다. 필요 시 downsample도 수행함
    '''
    expansion = 4
    def __init__(self, inplanes, out_planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(
            out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which reduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, dilated=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

```