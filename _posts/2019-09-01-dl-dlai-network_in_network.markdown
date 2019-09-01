---
layout: post
title: Network In Network and 1x1 Convolutions
date: 2019-09-01 00:00:00
img: dl/dlai/dlai.png
categories: [dl-dlai] 
tags: [python, deep learning, CNN, Network In Network, 1 x 1] # add tag
---

- 이번 글에서는 `Network In Network`과 `1 x 1 Convolution`에 대하여 다루어 보도록 하겠습니다.
- 아키텍쳐 설계에 있어서 `1 x 1 Convolution`을 사용하는 것은 꽤 도움이 많이 되는 방법입니다.
- 결과적으로 `1 x 1 Convolution`은 다음과 같은 역할을 하게됩니다.
    - 행과 열의 크기 변환없이 Channel의 수를 조절할 수 있습니다.
    - 또는 행과 열 그리고 Chaneel의 수를 변화하지 않고 단순히 **weight 및 비선형성을 추가**하는 역할로 사용할 수 있습니다.

<br>
<center><img src="../assets/img/dl/dlai/network_in_network/1.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 슬라이드의 위 그림을 먼저 살펴보도록 하겠습니다.
- 단순히 (6 x 6) 매트릭스에 2라는 스칼라 값을 곱하여 2배의 결과를 출력하는 매트릭스를 만들었습니다.
    - 정확히는 (6 x 6 x 1) 이미지를 (1 x 1 x 1)이미지와 convoltuion 한 결과입니다. 
- 만약에 아래 그림의 입체형태와 같은 (6 x 6 x 32) 인풋에 이러한 convolution 연산을 하게 되면 어떻게 될까요? 
- 즉 (1 x 1 x #channel)의 convolution연산을 하면 어떻게 될까요? 이 연산은 꽤나 유용한 역할을 하게 됩니다.
- 특히 (1 x 1) convolution이 수행할 작업은 여기에 있는 36가지 위치(행과 열의 곱에 의한 픽셀 갯수)를 각각 살펴보고 **왼쪽의 32개의 숫자(채널)와 필터의 32개 숫자**사이에 요소간 곱셈을 하는 것입니다. 
- 그리고 나서 여기에 `ReLU` 비선형성을 적용합니다. 

<br>

- 정리하면 (1 x 1) convolution 연산을 할 때, 필요한 필터는 `(1 x 1 x #channel x #filter)`가 됩니다.
- 여기서 `#channel`은 입력단의 channel 수와 동일해야 하고 `#filter`는 원하는 출력단의 channel 수로 지정해야 합니다.
- 이렇게 연산하는 방법을 `(1 x 1 Convolution)`이라고 합니다. 또는 `Network In Network`라고도 합니다.
- 이 컨셉은 딥러닝 네트워크를 구축할 때 많이 사용되는 방법입니다. 특히 `Inception`에서 사용되기도 하였는데 그 내용은 다음 글에서 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/dl/dlai/network_in_network/2.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 특히, `1 x 1 Convolution`이 유용하게 사용되는 예는 뒤와 같습니다.
- 일반적으로 행과 열의 사이즈를 줄이고 싶다면 Pooling을 사용하면 됩니다. 그런데 채널의 수를 줄이고 싶다면 어떻게 해야 할까요?
- 그 답은 바로 `1 x 1 Convolution`입니다. 
- 많은 채널 중의 하나가 과하게 커서 이걸 줄이고 싶다면, 예를 들어 (28 x 28 x 192)의 인풋을 어떻게 (28 x 28 x 32)로 줄일 수 있을까요?
- 그 방법은 (1 x 1 x 192) 필터를 32개 사용하여 convolution 연산을 하는 것입니다. 
    - 즉, 이 방법은 Channel의 수를 줄이는 방법으로 사용될 수 있습니다.
- 물론 `1 x 1 Convolution`을 사용하면서 channel의 갯수를 유지하는 경우도 있습니다. 
- 이 때 `1 x 1 Convolution`을 사용하는 목적은 weight의 갯수를 늘리고 비선형성을 더 부과하여 더 복잡한 네트워크를 학습할 수 있도록 도와줍니다.
- 이 기능은 `Inception` 네트워크에서 유용하게 사용되는데 그 내용은 다음 글에서 한번 살펴보도록 하겠습니다.