---
layout: post
title: DenseNet (Densely connected convolution networks)
date: 2020-06-06 00:00:00
img: dl/concept/densenet/0.png
categories: [dl-concept] 
tags: [딥러닝, densenet, densely connected convolution networks] # add tag
---

<br>

[deep learning 관련 글 목차]()https://gaussian37.github.io/dl-concept-table/

<br>

- 참조 : https://pytorch.org/hub/pytorch_vision_densenet/
- 참조 : https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
- 참조 : https://youtu.be/fe2Vn0mwALI?list=WL

<br>

## **목차**

<br>

- ### DenseNet 설명
- ### Pre-activation
- ### Pytorch 코드 설명

<br>

## **DenseNet 설명**

<br>

- 이번 글에서는 2017 CVPR에서 소개된 `DenseNet`에 대하여 다루어 보도록 하겠습니다.

<br>

- [Residual Network](https://gaussian37.github.io/dl-concept-resnet/)를 잘 이해하고 있다면 DenseNet을 이해하기는 상당히 쉽습니다. 만약 ResNet을 잘 모른다면 앞에 연결해 놓은 링크를 통해 먼저 학습을 하고 이 글을 읽기를 권장드립니다.

<br>
<center><img src="../assets/img/dl/concept/densenet/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 먼저 ResNet과 DenseNet의 공통점은 Skip Connection을 이용한다는 점입니다. 즉, 정보를 더 깊은 layer 까지 전달할 수 있는 path를 만들어 학습이 잘 되도록 하였습니다.
- 반면 차이점은 위 그림과 같이 ResNet은 Skip Connection이 덧셈을 통해 이루어지는 반면 DenseNet은 `concatenation`을 통하여 이루어 집니다. 그 차이점은 수식을 보면 이해할 수 있습니다.
- 먼저 `ResNet`의 skip connection 부분입니다. 설명한 바와 같이 `+` 연산을 통해 합쳐집니다.

<br>

$$ x_{l} = H_{l}(x_{l-1}) + x_{l-1} $$

<br>

- 반면 `DenseNet`에서는 concatenation을 합니다. 이 방법이 ResNet과의 차이점 입니다.

<br>

$$ x_{l} = H_{l}([x_{0}, x_{1}, \cdots, x_{l-1}]) $$

<br>

- 기존의 ResNet에서는 `Bottleneck` 구조를 만들기 위하여 **1x1 convolution**으로 **dimension reduction**을 한 다음에 다시 **1x1 convolution**을 이용하여 **expansion**을 하였습니다. dimension이 축소되었다가 확대되는 구조가 있기 때문에 bottleneck 형상을 만들어 내었다고 생각하면 됩니다.
- 반면 DenseNet에서는 **1x1 convolution**을 이용하여 **dimension reduction**을 하지만 expansion은 하지 않습니다. 대신에 feature들의 `concatenation`을 이용하여 expansion 연산과 같은 효과를 만들어 냅니다.
- 이 때, 얼맡큼 feature가 늘어날 지에 대한 값을 `growth rate` 라고 합니다.

<br>
<center><img src="../assets/img/dl/concept/densenet/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 DenseNet의 핵심인 `dense block`을 설명합니다. 5 layer dense block을 나타내며 growth rate는 4입니다. 각각의 layer는 이전의 모든 layer의 feature map들을 input으로 받는 것을 볼 수 있습니다.
- growth rate가 4이기 때문에 channel의 수가 등차수열 처럼 4씩 늘어나는 것을 확인할 수 있습니다.

<br>

## **Pre-activation**

<br>







<br>

[deep learning 관련 글 목차]()https://gaussian37.github.io/dl-concept-table/

<br>