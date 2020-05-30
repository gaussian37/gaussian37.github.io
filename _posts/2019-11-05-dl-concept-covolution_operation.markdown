---
layout: post
title: Convolution 연산 정리 (w/ Pytorch)
date: 2019-11-05 00:00:00
img: dl/concept/conv/conv.gif
categories: [dl-concept] 
tags: [convolution operation, 컨볼루션 연산] # add tag
---

<br>

- 출처 : https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
- 출처 : https://github.com/vdumoulin/conv_arithmetic

<br>

## **목차**

<br>

- ### Convolution 연산 소개
- ### Kernel 이란
- ### Trainable parameter와 bias 란
- ### Input, output channel의 갯수란
- ### Kernel의 size 란
- ### Stride 란
- ### Padding 이란
- ### Dilation 이란
- ### Group 이란
- ### Output Channel Size 란

<br>

## **Convolution 연산 소개**

<br>

- 이번 글에서는 영상 처리를 위한 딥러닝을 사용할 때 기본적으로 사용되는 `convolution` 연산에 대하여 다루어 보려고 합니다.
- `convolution` 연산은 예를 들면 아래 그림과 같습니다.

<br>
<center><img src="../assets/img/dl/concept/conv/1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 영상 처리의 입력에 사용되는 영상은 일반적으로 3차원의 형태를 가집니다. 즉, 영상의 width, height, channel 3가지가 기본 요소이며 Framework에 따라 (**channel**, height, width)의 순서 또는 (height, width, **channel**) 순서로 영상을 표시합니다. 이번 글에는 (height, width, **channel**)의 순서로 영상을 표시해 보도록 하겠습니다. channel은 예를 들어 R,G,B 값이 될 수 있습니다.
- 그러면 위 이미지의 가장 왼쪽이 Input 일 때, (7, 7, 3)의 크기를 가집니다. 그리고 Output은 (3, 3, 2)의 크기를 가집니다. 연산의 속성들을 정리하면 다음과 같습니다.

<br>

- Input shape : (7, 7, 3)
- Output shape : (3, 3, 2)
- Kernel : (3, 3)
- Padding : (1, 1)
- Dilation : (2, 2)
- Group : 1

<br>

- Input, output을 제외한 `Kernel, Padding, Dilation, Group`은 Convolution 영상 기법에 해당합니다. 이 내용에 대하여 간략하게 알아보도록 하겠습니다.
- 먼저 `Pytorch`의 `Conv2d` 모듈의 파라미터들을 살펴보겠습니다. 위의 convolution 연산과 동일한 파라미터가 필요합니다.

<br>

- `in_channels` (int) — Number of channels in the input image
- `out_channels` (int) — Number of channels produced by the convolution
- `kernel_size` (int or tuple) — Size of the convolving kernel
- `stride` (int or tuple, **optional**) — Stride of the convolution. Default: 1
- `padding` (int or tuple, **optional**) — Zero-padding added to both sides of the input. Default: 0
- `dilation` (int or tuple, **optional**) — Spacing between kernel elements. Default: 1
- `groups` (int, **optional**) — Number of blocked connections from input channels to output channels. Default: 1
- `bias` (bool, **optional**) — If True, adds a learnable bias to the output. Default: True

<br>

- 그러면 위 키워드 들을 하나 하나 다루어 보도록 하겠습니다.

<br>

## **Kernel 이란**

<br>
<center><img src="../assets/img/dl/concept/conv/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- kernel 또는 filter라고 하는 convolution matrix는 input image의 전 영역들에 걸쳐서 연산 됩니다.
- 일반적으로 **left-top** → **right-bottom** 방향으로 sliding window를 하면서 연산이 됩니다. 이 때 각 영역에서 `convolution product`가 연산됩니다. 이 연산을 거치기 전을 Input image 이라고 한다면 연산 후는 `filtered image` 라고 합니다.
- 위 그림에서 `convolution product`를 살펴보면 kernel의 각 원소와 1 대 1 대응이 되는 Input 영역이 있고 Input의 값과 kernel의 값을 곱한 뒤 그 결과를 모두 더하면 output 픽셀 하나가 완성됨을 볼 수 있습니다. 즉, **element-wise multiplication** 연산을 합니다.
- 그러면 이 연산을 수식으로 표현해 보겠습니다.

<br>

$$  G(x, y) = w * F(x, y) = \sum_{\delta x = -k_{i}}^{k_{i}} \sum_{\delta y = -k_{j}}^{k_{j}} w(\delta x, \delta y) \cdot F(x + \delta x, y + \delta y) $$

$$ w \text{ is kernel and } -k_{i} \ge \delta x \ge k_{i}, \ \ -k_{j} \ge \delta y \ge k_{j} $$

<br>

- 위 수식에서 $$ \detla x, \delta y $$ 각각은 현재 convolution product가 되는 kernel의 중앙 위치가 이미지 좌표계 원점에서 얼마큰 떨어져 있는지를 나타냅니다.

<br>
<center><img src="../assets/img/dl/concept/conv/3.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 연산의 과정을 위 예제를 통해 정리해 보겠습니다. 위 예제에서는 Input image의 크기는 (9, 9, 1) 입니다. kernel의 크기는 (3, 3) 입니다. kernel은 Input image의 전 영역을 sliding window 하면서 **element-wise multiplication** 연산을 하게 되고 각 연산은 (1, 1)의 scalar 값을 가지게 됩니다. 

<br>

## **Trainable parameter와 bias 란**

<br>

- parameter들은 학습 과정 동안 update 됩니다. `Conv2d`에서 학습이 되는 parameter는 무엇일까요? 바로 `kernel` 입니다.
- 앞의 예제에서 (3, 3) 크기의 kernel을 이용하여 convolution 연산을 하였습니다. 이 때 사용된 9 개의 원소가 학습할 때 업데이트가 되는 parameter 입니다.

<br>

$$ w * F(x, y) = \Biggl( \sum_{\delta x = -k_{i}}^{k_{i}} \sum_{\delta y = -k_{j}}^{k_{j}} w(\delta x, \delta y) \cdot F(x + \delta x, y + \delta y) \Biggr) + w_{bias}$$

$$ \text{where } w_{bias} \in \mathbb R \text{ is the bias of the kernel } w $$

<br>

- 만약 `bias` 까지 포함하여 생각한다면 위 수식처럼 표현할 수 있습니다. `bias`는 convolution product 연산한 값에 덧셈을 해주는 trainable parameter 입니다.
- 만약 앞에서 다룬 예제에서 (3, 3) kernel에 bias가 추가된다면 parameter의 갯수는 총 몇개가 될까요? 9 + 1 =  10개가 됩니다.

<br>

## **Input, output channel의 갯수란**

<br>


- ### Kernel의 size 란
- ### Stride 란
- ### Padding 이란
- ### Dilation 이란
- ### Group 이란
- ### Output Channel Size 란













<br>

- 아래 애니메이션의 `파란색`이 `인풋`이고 `청록색`이 `아웃풋`입니다.

<br>

## **Basic Convolution Operation**

<br>

- Convolution 연산을 이용하면 input의 feature를 압축하게 되므로 convolution 연산 이후의 feature map의 크기는 더 줄어들게 됩니다.
- 아래 애니메이션들도 보면 파란색의 인풋이 convolution 연산을 거치면서 청록색 아웃풋 처럼 사이즈가 작아지게 된 것을 볼 수 있습니다.

<br>

<br>
<center><img src="../assets/img/dl/concept/conv/no_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/arbitrary_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/same_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/full_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/no_padding_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/padding_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/padding_strides_odd.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

