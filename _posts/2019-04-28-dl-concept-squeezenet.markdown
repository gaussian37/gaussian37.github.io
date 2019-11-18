---
layout: post
title: SQUEEZENET, ALEXNET-LEVEL ACCURACY WITH 50X FEWER PARAMETERS AND < 0.5MB MODEL SIZE
date: 2019-04-24 00:00:00
img: dl/concept/squeezenet/squeezenet.png
categories: [dl-concept] 
tags: [deep learning, squeezenet, 스퀴즈넷] # add tag
---

<br>

- 이번 글에서는 경량화 모델 중 하나인 스퀴즈넷에 대하여 다루어 보도록 하겠습니다.
- 출처 : https://youtu.be/eH5O5nDiFoY, https://youtu.be/ge_RT5wvHvY

<br>

## **목차**

<br>

- ### 스퀴즈넷의 의의 및 배경
- ### 스퀴즈넷 관련 연구
- ### 스퀴즈넷 아키텍쳐 설명
- ### 스퀴즈넷 pytorch 코드

<br>

## **스퀴즈넷의 의의 및 배경**

<br>

- 먼저 스퀴즈넷의 contribution을 살펴보면
    - ImageNet에서 AlexNet 수준의 accuracy를 달성하였지만 AlexNet 보다 50배 작은 파라미터 수를 이용하였다는 것에 의미가 있습니다.
    - hyperpar parameter를 어떻게 조합하는냐에 따라서 성능과 파라미터의 수가 어떤 관계가 있는지를 확인한 것 또한 의미가 있습니다. 

<br>

- 스퀴즈넷이 나온 배경들을 살펴보면
    - 시간이 지날수록 딥러닝 네트워크의 accuracy 성능이 좋아지고 있는 반면에 모델의 사이즈가 점점 더 커지는 문제가 있었습니다.
    - 하지만 PC 레벨이 아닌 다양한 어플리케이션에서 딥러닝 모델을 사용하려면 사이즈가 너무 커지면 사용하기 어려운 한계가 있습니다.
    - 이러한 배경 속에서 quantization이나 모델 경량화에 많은 관심이 생기게 되었습니다.

<br>

## **스퀴즈넷 관련 연구**

<br>

- 스퀴즈넷에서는 `micro architecture`와 `macro architecture` 측면으로 나누어서 전체 내용을 설명하고 있습니다.
- `micro architecture`에서 다룰 내용은 다음과 같고
    - **1x1 convolution**
    - **3x3 convolution**
    - **skip connection**
- `macro architecture`에서는 **block**과 같은 내용을 언급해보겠습니다.

<br>

- 먼저 다룰 내용은 `micro architecture`로 이 논문에서 다루는 내용은 **1x1 convolution filter가 channel 방향의 multi-layer perceptron과 같은 역할**을 한다는 것입니다. 

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 1x1 convolution의 경우 각각의 채널에 element-wise로 multiplication 한 후에 합하게 되므로 이 연산은 채널 방향의 multi-layer perceptron 즉, FC layer와 같다는 뜻입니다.

<br>

- 그 다음으로 어떤 필터를 사용할 것인가에 대한 연구가 되어왔었는데 **5x5 convolution filter가 4개의 3x3 convolution filter로 분해 된다**는 것입니다. 
    - 즉, 큰 convolution filter가 3x3 convolution filter로 인수분해 될 수 있다는 뜻입니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 인수분해 되는 것을 위의 그림과 같이 표현할 수 있습니다. 

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 효과 또한 1개 layer의 5x5 필터는 2개 layer의 3x3 필터로 대체될 수 있음을 나타냅니다.

<br>

- 마지막으로 skip connection` 입니다. 먼저 `skip connection`의 관점은 ResNet과 Segmentation에서 볼 수 있습니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- Segmentation에서의 skip connection을 살펴보면 resolution 정보가 굉장히 중요한데 depp layer로 내려갈수록 resolution 정보가 맞지 않는 문제가 있습니다.
- 이 문제를 보완하기 위하여 skip connection을 이용하여 deep layer에도 resolution 정보를 전달해 줍니다. 

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- ResNet에서 같은 컨셉으로 사용되었고 특히 layer가 깊어질수록 발생할 수 있는 vanishing gradient 문제를 skip connection을 통하여 개선한 것이 의미가 있습니다. 

<br>

- 다음으로는 `macro architecture` 관련 내용으로 `block`을 쌓는 컨셉에 해당합니다.
    - 인셉션, 모바일넷, 셔플넷 등에서 어떤 단위의 block들을 쌓아서 네트워크 전체를 구성하는 것과 같이 스퀴즈넷에서도 유사한 개념이 사용됩니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

## **스퀴즈넷 아키텍쳐 설명**

<br>

- 스퀴즈넷의 첫번째 전략은 **3x3 필터를 1x1 필터로 대체하는 것**입니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\6.gif" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위와 같이 3개의 채널이 있을 때, 3개의 convolution filter를 사용하여 multiplication을 하고 그 결과를 모두 합하는 것을 한다고 앞에서 다루었습니다. 
- 이 경우에는 파라미터의 수가 `3 x 3 x #channel`이 됩니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 하지만 1x1 convolution filter에서는 `#channel`이 파라미터의 수가 됩니다. 따라서 파라미터의 수가 굉장히 줄어들게 됩니다.

<br>

- 스퀴즈넷의 두번째 전략은  **3x3 필터의 채널 수를 줄이는 것**입니다.
- 만약 인풋이 3채널이면 인풋을 처리하는 데 필터의 수가 3개가 필요합니다. 이 때, 3x3 필터의 채널이 늘게되면 필터의 수가 늘어난 것에 3x3 만큼 곱한 갯수가 늘어나게 되어 파라미터의 수가 급증하게 됨을 알 수 있습니다.

<br>

- 스퀴즈넷의 세번째 전략은 convolution layer의 **activation map을 크게** 가지기 위하여 **downsample을 늦게** 한다는 것입니다.
- 네트워크의 앞쪽에서 downsample을 하게 되면 activation map의 크기가 줄어 연산량이 줄기 때문에 처리 속도가 빨라지지만 그만큼 activation map에서의 정보가 손실되기 때문에 accuracy가 떨어지게 됩니다. 따라서 downsample을 늦게 하여 accuracy를 높이는 전략을 취합니다.

<br>

- **전략 1,2에서는 필터를 효율적으로 사용하여 최대한 파라미터 수를 줄이는 것에 목적을 가지고 있고 그 전략을 취하는 조건 속에서 전략 3에서는 최대한 성능을 내기 위해 activation map의 크기를 유지하는 것으로 스퀴즈넷의 전체 전략을 정리할 수 있습니다.**

<br>

- 마지막으로 스퀴즈넷에서 사용한 `block`인 `Fire Module`을 살펴보겠습니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

-  먼저 `squueze layer`는 1x1 convolution filter를 통하여 채널을 압축하고 `expand layer`는 1x1 convolution filter와 3x3 convolution filter를 통하여 다시 팽창시켜주는 역할을 하게 됩니다. 
- activation은 주로 ReLU를 사용하였습니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\8.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 예를 들어 위와 같이 인풋이 128 채널이 들어오면 1x1을 통해서 16 채널로 줄였다가 다시 1x1 채널로 64개, 3x3 채널로 64개를 만듭니다. 이것을 concatenate를 하여 다시 128 채널의 아웃풋을 만듭니다. 

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\9.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 아키텍쳐들은 논문에서 나온 내용들입니다.
- 첫번째가 가장 기본적인 스퀴즈넷의 형태이고 가운데가 skip connection을 적용한 형태입니다. 오른쪽 형태는 조금 복잡한 skip connection 형태까지 적용해본 아키텍쳐입니다.

<br>
<center><img src="..\assets\img\dl\concept\squeezenet\10.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 마지막으로 스퀴즈넷 논문에서 알렉스넷과 비교하여 실험한 결과를 나타냈습니다.
- 결과적으로 보면 파라미터의 수는 50배 이상을 줄일 수 있으면 accuracy 성능은 유사하게 나올 수 있었습니다.
    - 그 중 simple skip connection을 해본 모델의 성능이 가장 잘 나왔습니다.
- 여기서 squeeze layer와 expand layer의 비율에 따라서 성능이 어떻게 변하는 지 또한 실험을 하였으며 논문을 통해 확인하실 수 있습니다.

<br>

## **스퀴즈넷 pytorch 코드**

<br>

```python
import torch
import torch.nn as nn
import torch.nn.init as init
from .utils import load_state_dict_from_url

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        # squeeze layer에서는 1x1 convolution을 통하여 채널을 줄여줍니다.
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        # expand layer에서는 채널을 다시 늘려줍니다. 
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # squeeze layer에서 채널을 줄이고 expand layer에서 다시 채널을 눌려주는 것 확인 
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                # 첫 인풋을 처리할 때에만 예외적인 Convolution 연산과 MaxPool을 적용한 뒤
                # 이후에는 Fire Module을 계속 적용합니다. 
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                # (입력채널, squeeze 채널, 1x1 expand 채널, 3x3 expand 채널)
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                # late downsampling(MaxPool)을 적용함
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version, pretrained, progress, **kwargs):
    model = SqueezeNet(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)

```