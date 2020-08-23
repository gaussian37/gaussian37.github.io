---
layout: post
title: Transposed Convolution과 Checkboard artifact
date: 2019-02-13 00:00:00
img: dl/concept/checkboard_artifact/0.png
categories: [dl-concept] 
tags: [deep learning, convolution, transposed, checkboard artifact] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

<br>

- 참조 : https://distill.pub/2016/deconv-checkerboard/

<br>

## **목차**

<br>

- ### Checkboard pattern 이란
- ### Transposed Convolution과 Overlap
- ### 더 좋은 Upsampling 방법

<br>

## **Checkboard pattern 이란**

<br>

- 딥러닝에서 feature를 Upsampling 할 때, 사용하는 방법 중 하나인 `Transposed Convolution`을 사용할 때 발생하는 문제인 `Checkboard artifact`에 대하여 다루어 보겠습니다.


<br>
<center><img src="../assets/img/dl/concept/checkboard_artifact/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 뉴럴 네트워크에 의해 생성된 이미지들을 자세히 들여다 보면 위 그림처럼 인공적인 체크보드 패턴을 가지는 결과를 종종 가집니다.
- 신기하게도, 체크보드 패턴은 강한 색상을 가진 이미지에서 가장 두드러지는 경향이 있습니다.

## **Transposed Convolution과 Overlap**

<br>

- 뉴럴 네트워크가 이미지를 생성하였을 때, 이 이미지는 종종 해상도는 낮으면서 높은 수준의 description을 가지는 경우가 있습니다.
- 이 경우 네트워크는 낮은 해상도에서 대략적인 이미지를 설명(describe)하고 그리고 높은 해상도로 이미지를 키워 나가면서 상세 정보들을 채워나아갑니다.
- 이를 위해서는 저해상도 이미지에서 고해상도 이미지로 변환하는 방법이 필요합니다. 이러한 방법 중 [Transposed Convolution](https://gaussian37.github.io/dl-concept-transposed_convolution/) 이 있습니다. Transposed Convolution을 사용하면 작은 이미지의 모든 점을 사용하여 큰 이미지를 만들어 낼 수 있습니다.
- Transposed Convolution 또한 Convolution 연산이므로 커널이 슬라이딩 윈도우 방식으로 이동하면서 연산이 진행됩니다. 특히, **kernel의 크기와 stride의 크기에 따라서 transposed convolution 연산의 overlap 영역이 발생할 수 있습니다.** 이 때 Transposed Convolution을 어떻게 사용하느냐에 따라서 overlap이 없을 수도 있고 또는 많이 생길 수도 있습니다.

<br>
<center><img src="../assets/img/dl/concept/checkboard_artifact/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림의 4 종류를 살펴 보면  `stride = 2`로 고정하고 `kernel_size`를 변경하였을 때, overlap 영역의 변화를 살펴볼 수 있습니다.
- 각 그림은 **입력 → Transposed Convolution → 출력**으로 변환되는 과정이며 **overlap 영역**은 진한 보라색 영역으로 나타납니다. overlap이 되는 영역은 출력이 중복되어 나타나며 이 값들은 모두 더해집니다. 즉, overlap 되는 영역의 출력 크기가 커질 가능성이 높습니다.
- 위 그림에서 `stride`는 입력에서 선택된 사각형의 간격이고 `kernel_size`는 입력에서 출력으로 매칭되는 영역의 너비입니다.
- kernel_size가 stride로 나뉘어지지 않을 때, 중복되어 겹쳐지는 부분이 더 발생하는 것을 확인 할 수 있습니다.예를 들어 size가 3, 5일 때 추가적으로 겹치는 부분이 발생합니다. 특히 많이 사용되는 `stride = 2, kernel_size = 3`의 경우를 살펴보면 가장 많이 겹친 곳은 2번 겹친 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/checkboard_artifact/2.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이러한 overlap 패턴은 2차원에서도 같은 원리로 발생합니다. 심지어 2개의 차원에서 동시에 겹치는 영역이 발생하여 앞의 1차원에서 보다 배로 겹치게 됩니다. 위 그림과 같이 `stride = 2, kernel_size = 3`인 경우 가장 많이 겹친 곳은 4번 겹친 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/checkboard_artifact/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 특히 Transposed Convolution은 원하는 해상도로 복원을 하기 위해 여러 층의 layer가 계속 쌓이는 것이 일반적입니다.
- 만약 앞에서 사용한 `stride = 2, kernel_size = 3`이 여러 층의 layer로 계속 쌓인다면 위 그림과 같이 지속적으로 overlap 되는 영역이 생길 수도 있습니다.

<br>

- 지금 까지 내용을 정리하면 **Transposed Convolution을 이용하여 Upsampling을 할 때,** $$ \text{kernel_size  } MOD \text{  stride} \ne 0 $$ **인 경우에 어떤 영역의 중복 연산이 배로 발생하게 되고 그 영역은 계속 누적되어 큰 값을 가지게 됩니다.** 데이터의 차원이 늘어날수록 겹치는 구간의 겹치는 횟수가 배로 늘어나게 됩니다.
- 이렇게 **연산이 누적되는 횟수의 차이가 발생하게 되어 checkboard artifact가 발생**하게 됩니다.

<br>

## **더 좋은 Upsampling 방법**

<br>

- Transposed Convolution을 사용하면서 checkboard artifact 문제를 개선하는 방법들이 연구되었고 대표적으로  $$ \text{kernel_size  } MOD \text{  stride} = 0 $$을 만족하도록 설계하여 checkboard가 생기는 중복 연산 구간을 줄이는 방법이 있습니다. 그럼에도 불구하고 발생하는 overlap 구간 및 Transposed Convolution의 한계로 인하여 Transposed Convolution을 사용하지 않고 Upsampling 하는 방법도 고안되었습니다.
- 대표적으로 `interpolation + convolution` 연산입니다. Transposed Convolution은 interpolation과 convolution 연산을 한번에 하는 역할을 하지만 `interpolation + convolution`에서는 `bilinear, bicubic, nearest-neight` 등의 방법을 이용하여 먼저 interpolation을 하고 그 뒤에 convolution 연산을 추가하는 방식입니다. 즉, **interpolation 역할과 convolution 역할을 분리**하는 것입니다.
    - nearest-neighbor interpolation : [https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation](https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
    - bilinear interpolation : [https://en.wikipedia.org/wiki/Bilinear_interpolation](https://en.wikipedia.org/wiki/Bilinear_interpolation)
- 따라서 현재 네트워크에 checkboard artifact 문제가 있고 개선하기가 어렵다면 Transposed Convolution 대신 `interpolation + convolution` 연산을 사용하는 것도 좋은 방법이 될 수 있습니다.


<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>