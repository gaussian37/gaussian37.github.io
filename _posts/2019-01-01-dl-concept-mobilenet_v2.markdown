---
layout: post
title: MobileNetV2(모바일넷 v2), Inverted Residuals and Linear Bottlenecks
date: 2019-01-01 00:00:00
img: dl/concept/mobilenet_v2/0.png
categories: [dl-concept] 
tags: [딥러닝, 모바일넷 v2, mobilenet v2] # add tag
---

<br>

- 출처 : 
- https://arxiv.org/abs/1801.04381
- https://www.youtube.com/watch?v=mT5Y-Zumbbw&t=1397s

<br>

- 이번 글에서는 mobilenet v2에 대하여 알아보도록 하겠습니다.
- 이 글을 읽기 전에 `mobilenet`과 `depth-wise separable` 연산에 대해서는 반드시 알아야 하기 때문에 모르신다면 아래 글을 읽으시길 추천드립니다.
- `mobilenet` : https://gaussian37.github.io/dl-concept-mobilenet/
- `depth-wise separable 연산` : https://gaussian37.github.io/dl-concept-dwsconv/

<br>

- Mobilenet v2의 제목은 **Inverted Residuals and Linear Bottlenecks**입니다. 
- 즉, 핵심적인 내용 2가지인 `Inverted Residuals`와 `Linear Bottlenecks`가 어떻게 사용되었는 지 이해하는 것이 중요하겠습니다.

<br>

## **목차**

<br>

- ### Mobilenet v2 전체 리뷰
- ### Linear Bottlenecks
- ### Inverted Residuals
- ### Pytorch 코드 리뷰

<br>

## **Mobilenect v2 전체 리뷰**

<br>

- 먼저 mobilenet v2 전체를 간략하게 리뷰해 보도록 하겠습니다.
- 앞선 mobilenet v1에서는 **Depthwise Separable Convolution** 개념을 도입하여 연산량과 모델 사이즈를 줄일 수 있었고 그 결과 모바일 디바이스와 같은 제한된 환경에서도 사용하기에 적합한 뉴럴 네트워크를 제시한 것에 의의가 있었습니다.
- mobilenet v2에서의 핵심 개념은 **Inverted Residual** 구조입니다.
- 논문에서는 이 구조를 이용한 네트워크를 backbone으로 하였을 때, object detection과 segmentation에서의 성능이 이 시점 (CVPR 2018)의 다른 backbone 네트워크 보다 더 낫다고 설명하였습니다.
- `Linear Bottleneck`과 `Inverted Residual`에 대한 자세한 내용은 전체 리뷰가 끝난 다음에 자세히 알아보고 먼저 큰 숲을 살펴보도록 하겠습니다.

<br>

### Mobilenet_v1 Vs. Mobilenet_v2

<br>

- 먼저 `mobilenet v2`의 전체적인 구조를 mobilenet v1에 비교해 보겠습니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet_v2/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 `mobilenet v1`입니다.
- mobilenet v1에서의 block은 2개의 layer로 구성되어 있습니다.
- 첫번째 layer가 depthwise convolution이고 두번째 layer가 pointwise convolution 입니다.
- 첫번째 layer인 depthwise convolution은 각 인풋 채널에 single convolution filter를 적용하여 네트워크 경량화를 하는 작업이었습니다.
- 두번째 layer인 pointwise convolution 또는 1x1 convolution에서는 depthwise convolution의 결과를 pointwise convolution을 통하여 다음 layer의 인풋으로 합쳐주는 역할을 하였습니다.
- 그리고 activation function으로 ReLU6를 사용하여 block을 구성하였습니다.
    - 참고로 ReLU6는 `min(max(x, 0), 6)` 식을 따르고 음수는 0으로 6이상의 값은 6으로 수렴시킵니다.

<br>

- 다음으로 `mobilenet v2`입니다.
- mobilenet v2에는 2가지 종류의 block이 있습니다. **첫번째는 stride가 1인 residual block**이고 **두번째는 downsizing을 위한 stride가 2인 block**입니다. 이 각각의 block은 3개의 layer를 가지고 있습니다.
- 두 block 모두 첫 번째 layer는 pointwise(1x1) convolution + ReLU6 입니다.
- 두번째 layer는 depthwise convolution 입니다. 첫번째 block은 여기서 stride가 1로 적용되고 두번째 block은 stride가 2가 적용되어 downsizing됩니다.
- 세번째 layer에서는 다시 pointwise(1x1) convolution이 적용됩니다. **단, 여기서는 activation function이 없습니다.** 즉, non-linearity를 적용하지 않은 셈입니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet_v2/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 입출력의 크기를 보면 상수 `t`가 추가되어 있습니다. 이 값은 `expansion factor`란 이름으로 도입하였고 논문의 실험에서는 모두 6으로 정하여 사용하였습니다.
- 예를 들어 입력의 채널이 64이고 t = 6이라면 출력의 채널은 384가 됩니다.

<br>

### Architecture

<br>

- `mobilenet v2`의 전체적인 아키텍쳐에 대하여 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet_v2/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 표에서 사용된 상수들의 의미를 먼저 살펴보겠습니다.
- `t` : expansion factor
- `c` : output channel의 수
- `n` : 반복 횟수
- `s` : stride
- 위와 같은 입력 resolution을 기준으로 multiply-add에 대한 연산 비용은 약 300M 이고 3.4M개의 파라미터를 사용하였습니다. 
    - 참고로 **width multiplier**는 **layer의 채널수를 일정 비율로 줄이는 역할**을 합니다. 기본이 1이고 0.5이면 반으로 줄이는 것입니다.
    - 그리고 **resuolution multiplier**는 **입력 이미지의 resolution**을 일정한 비율로 줄이는 것입니다. 
    - mobilenet v1에서 도입된 width multiplier는 1로 설정하였습니다.
    - performance와 파라미터등의 trade off는 입력 resolution과 width multiplier 조정등으로 가능합니다.
    

<br>
<center><img src="../assets/img/dl/concept/mobilenet_v2/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 테이블에서는 각 인풋 사이즈에 따른 `mobilenet`, `mobilenet v2`, `shufflenet`에 대한 **채널의 수/메모리(Kb 단위)**의 최대 크기를 기록하였습니다.
- 논문에서는 mobilenet v2의 크기가 가장 작다고 설명하고 있습니다.

<br>

## **Linear Bottlenecks**

<br>

- 그러면 `Linear Bottleneck`에 대한 내용을 알아보도록 하겠습니다.
- 먼저 `manifold` 라는 개념을 알아야 하는데, 간략히 말하면 뉴럴 네트워크들은 일반적으로 고차원 → 저차원으로 압축하는 `Encoder` 역할의 네트워크 부분이 발생하게 됩니다. 이 과정에서 feature extraction을 수행하게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet_v2/manifold.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림처럼 고차원의 데이터가 저차원으로 압축되면서 특정 정보들이 저차원의 어떤 영역으로 매핑이 되게 되는데, 이것을 `manifold`라고 이해하고 아래 글을 이해하시면 되겠습니다.
- 따라서 뉴럴 네트워크의 manifold는 저차원의 subspace로 매핑이 가능하다고 가정해 보겠습니다.
- 이런 관점에서 보면 어떤 데이터에 관련된 manifold가 `ReLU`를 통과하고 나서도 입력값이 음수가 아니라서 0이 되지 않은 상태라면, `ReLU`는 `linear transformation` 연산을 거친 것이라고 말할 수 있습니다. 즉, ReLU 식을 보면 알 수 있는것 처럼, identity matrix를 곱한것과 같아서 단순한 linear transformation과 같다고 봐집니다.  
- 그리고 네트워크를 거치면서 저차원으로 매핑이 되는 연산이 계속 되는데, 이 때, (인풋의 manifold가 인풋 space의 저차원 subspace에 있다는 가정 하에서) ReLU는 양수의 값은 단순히 그대로 전파하므로 즉, **linear transformation**이므로, manifold 상의 정보를 그대로 유지 한다고 볼 수 있습니다.
- 즉, 저차원으로 매핑하는 bottleneck architecture를 만들 때, **linear transformation** 역할을 하는 **linear bottleneck layer**를 만들어서 **차원은 줄이되 manifold 상의 중요한 정보들은 그대로 유지**해보자는 것이 컨셉입니다.
- 여기 까지는 일단 가설이고, 실제로 실험을 하였는데 bottleneck layer를 사용하였을 때, `ReLU`를 사용하면 오히려 성능이 떨어진다는 것을 확인하였습니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet_v2/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- ### Inverted Residuals

<br>

- ### Pytorch 코드 리뷰

<br>