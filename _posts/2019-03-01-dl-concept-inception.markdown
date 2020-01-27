---
layout: post
title: Inception Module과 GoogLeNet
date: 2019-03-01 00:00:00
img: dl/concept/inception/0.png
categories: [dl-concept] 
tags: [딥러닝, inception, 인셉션] # add tag
---

<br>

- 이번 글에서는 GoogLeNet 또는 Inception이라는 딥 뉴럴 네트워크에 대하여 알아보도록 하겠습니다.

<br>

**목차**

<br>

- ### Inception
- ### Pytorch 코드

<br>

## **Pytorch 코드**

<br>

- 앞에서 살펴본 `Inception Module`에 대한 내용을 다시 한번 정리하면서 Pytorch 코드에 대하여 살펴보도록 하겠습니다.
- 살펴본 것 처럼 네트워크 전체의 구조가 다소 복잡한 감이 있지만 자세히 살펴보면 반복적인 형태를 띄기 때문에 하나씩 모듈별로 하나씩 살펴보면 코드를 이해하힐 수 있을 것입니다.

<br>
<center><img src="../assets/img/dl/concept/inception/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 기본적인 Inception 모듈에 해당합니다. 이전 레이어의 출력에 다양한 필터 크기로 합성곱 연산을 한 것을 확인 할 수 있습니다.
- 여기서 사용한 convolution 필터는 1x1, 3x3, 5x5 이고 추가적으로 3x3 맥스 풀링을 사용하였습니다.

<br>
<center><img src="../assets/img/dl/concept/inception/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번에 다루어 볼 코드는 기본적인 Inception 모듈에서 각각의 연산에 1x1 convolution이 추가된 형태를 알아보려고 합니다.
- 이 코드 형태를 이해하면 다른 버전의 Inception을 이해하는 것에도 어려움이 없을 것으로 판단됩니다.
- 먼저 1x1 convolution을 사용한 이유를 살펴보면 앞에서도 설명한 것과 같이 메모리 사용 절약의 측면이 큰 이유가 됩니다.

<br>
<center><img src="../assets/img/dl/concept/inception/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 1x1 convolution 연산을 이용하면 위 그림과 같이 width와 height의 크기는 유지한 상태에서 channel의 크기만 변경 가능합니다.
- channel의 크기를 줄이게 되면 메모리를 절약할 수 있습니다.
- 또한 위 그림을 기준으로 보면 1x1 convolution은 이미지의 width height 기준으로 하나의 픽셀만을 입력으로 받지만 채널 관점에서 봤을 때에는 채널의 갯수만큼의 입력을 받아서 하나의 결과를 생성하고 이를 목표 채널 수 만큼 반복합니다.
- 위 그림에서 왼쪽을 1 x 1 x 3 크기의 feature라고 하고 오른쪽을 1 x 1 x 5 크기의 feature라고 한다면 1 x 1 x 3 feature의 width와 height가 같은 위치의 (e.g. (1, 1)) feature들을 이용하여 같은 위치의 1 x 1 x 5 feature를 만들어 냅니다.
- 또한 각 width, height 위치의 채널들이 서로 대응되므로 Fully Connected Layer의 성격도 띄게 됩니다. 물론 FC layer처럼 filter의 위치 속성을 없애지 않고 계속 유지한다는 장점도 있습니다.
- 즉 정리하면, 1 x 1 convolution을 이용하여 메모리를 절약할 수도 있고 FC layer와 같이 filter 간의 연결도 하면서 feature에서의 공간을 무너뜨리지 않기에 성능 향상에 도움이 됩니다.

<br>

- 실제 Inception 모듈에 적용해 보면 기본적인 Inception 모듈에서는 128개 채널에서 256개 채널로 늘어나는 연산이었다면 1x1 convolution이 적용된 Inception 모듈에서는 1x1 convoltuion을 통해 채널의 수를 128개에서 32개로 줄인 다음에 다시 convolution 연산을 통하여 256개 채널로 늘립니다.
- 이렇게 적용하면 기존 연산보다 메모리 사용을 줄일 수 있습니다.

<br>

- 인셉션 모듈에는 1x1 연산, 1x1 연산 + 3x3 연산, 1x1 연산 + 5x5 연산, 3x3 맥스 풀링 + 1x1 연산 이렇게 4가지 연산이 있고 각각의 연산들을 채널 차원으로 붙여줍니다.
- 참고로 `nn.Conv2d` 함수의 파라미터는 각각 #input_channels, #output_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode 순서이고 아래 코드에서 사용하는 것은 **#input_channels, #output_channels, kernel_size, stride, padding** 까지 입니다.

<br>

```python
def conv_1(in_dim, out_dim):
    # 1x1 연산
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 1, 1)
        nn.ReLU(),
    )
    return model

def conv_1_3(in_dim, mid_dim, out_dim):
    # 1x1 연산 + 3x3 연산
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),
        nn.ReLU()
        nn.Conv2d(mid_dim, out_dim, 3, 1, 1),
        nn.ReLU()
    )
    return model

def conv_1_5(in_dim, mid_dim, out_dim):
    # 1x1 연산 + 5x5 연산
    model = nn.Sequential(
        nn.Conv2d(in_dim, mid_dim, 1, 1),
        nn.ReLU(),
        nn.Conv2d(mid_dim, out_dim, 5, 1, 2),
        nn.ReLU(),
    )
    return model

def max_3_1(in_dim, out_dim):
    # 3x3 맥스 풀링 + 1x1 연산
    model = nn.Sequential(
        nn.MaxPool2d(3, 1, 1),
        nn.Conv2d(in_dim, out_dim, 1, 1),
        nn.ReLU(),
    )
    return model
```

<br>

- 위 코드들을 조합하면 인셉션 모듈을 만들 수 있습니다.
- 위 코드의 함수명을 잘 살펴보면 앞에서 언급한 인셉션 모듈의 4개 부분에 해당하는 것을 확인하실 수 있습니다.

<br>

- 위 코드를 이용하여 `Inception module`을 만들어 보도록 하겠습니다.
- 아래 코드는 입력이 들어오면 4가지 연산을 따로 진행하고 마지막에 채널 방향으로 이를 붙여주는 모듈입니다.

<br>

```python
class inception_module(nn.Module):
    def __init__(self, in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool):
        super(inception_module, self).__init__()

        self.conv_1 = conv_1(in_dim, out_dim_1)
        self.conv_1_3 = conv_1_3(in_dim, mid_dim_3, out_dim_3),
        self.conv_1_5 = conv_1_5(in_dim, mid_dim_5, out_dim_5),
        self.max_3_1 = max_3_1(in_dim, pool)

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x)
        output = torch.cat([out_1, out_2, out_3, out_4], 1)
        return output

```

<br>

- 마지막으로 위 `Inception` 모듈을 이용하여 `GoogLeNet`을 작성하면 다음과 같습니다.
- 처음 입력은 컬러 이미지 라는 가정 하에 3채널을 입력받습니다.
- 각 `inception module`의 input의 크기는 이전 inception module의 **out_dim_1 + out_dim_3 + out_dim_5 + pool**과 같습니다.

<br>

```python
class GoogLeNet(nn.Module):
    def __init__(self, base_dim, num_classes = 2):
        super(GoogLeNet, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base_dim, base_dim * 3, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer_2 = nn.Sequential(
            # inception_module(in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool)
            inception_module(base_dim * 3, 64, 96, 128, 16, 32, 32),
            inception_module(base_dim * 4, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer_3 = nn.Sequential(
            # inception_module(in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool)
            inception_module(480, 192, 96, 208, 16, 48, 64),
            inception_module(512, 160, 112, 224, 24, 64, 64),
            inception_module(512, 128, 128, 256, 24, 64, 64),
            inception_module(512, 112, 144, 288, 32, 64, 64),
            inception_module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),
        )

        self.layer_4 = nn.Sequential(
            # inception_module(in_dim, out_dim_1, mid_dim_3, out_dim_3, mid_dim_5, out_dim_5, pool)
            inception_module(832, 256, 160, 320, 32, 128, 128),
            inception_module(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, 1),
        )

        self.layer_5 = nn.Dropout2d(0.4)
        self.fc_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out
```