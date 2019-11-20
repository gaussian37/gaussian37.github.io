---
layout: post
title: Fully Convolutional Networks for Semantic Segmentation
date: 2019-08-21 00:00:00
img: vision/segmentation/fcn/fcn.jpg
categories: [vision-segmentation] 
tags: [vision, segmentation, fcn] # add tag
---

- 이번 글에서 다루어 볼 논문은 `FCN`으로 유명한 **Fully Convolutional Networks for Semantic Segmentation** 입니다.

<br>

## **목차**

- ### FCN의 배경
- ### FCN 구조 설명 - downsampling
- ### FCN 구조 설명 - upsampling
- ### FCN 구조 설명 - skip connection
- ### convolutional and deconvolutional network
- ### pytorch 코드

<br>

- 이 글에서는 FCN의 배경과 전체적인 네트워크 구조를 살펴보고 내용의 핵심이라 할 수 있는 Deconvolution 연산에 대하여 자세히 다루어 보도록 하겠습니다. 마지막으로 pytorch 코드 까지 살펴보겠습니다.

<br>

## **FCN의 배경**

<br>

- 이미지 처리를 위한 딥러닝 네트워크의 시작은 `CNN` 기반의 알렉스넷이 대회에서 성능을 거두면서 부터 시작되었습니다.
- '12년도에 `알렉스넷`을 시작으로 다양한 딥러닝 기반의 `classification`을 위한 네트워크가 고안되기 시작하였고
- '14년도에 `VGG`와 `GoogLeNet` 그리고 '15년도에 대망의 `ResNet` 이 나오면서 `classification` 성능에 비약적인 발전이 있어왔습니다.
- 이렇게 `classification`에서 좋은 성능을 보인 CNN 기반의 딥러닝 네트워크를 `Localization` 이나 `Segmentation`에도 적용시켜서 성능 향상을 해보려는 시도가 있었고 그 결과 확인하게 된것들이 `FCN`에 반영됩니다.

<br>

## **FCN 구조 설명 - downsampling**

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 `classification`을 할 때, 마지막 FC layer를 softmax로 출력을 하게 되면 확률 값이 나타나게 되고 가장 큰 확률값에 해당하는 클래스가 이미지에 해당하는 것으로 판단하게 됩니다.
- 이와 같은 classification 작업에서는 물체의 **위치 정보는 없고** 단지 물체가 어떤 물체인지에 대한 확률 값만 가지게 됩니다. 즉, 물체의 위치 정보를 잃어버리게 되는 것이지요.
- 하지만 입력 값을 보면 이미지 내의 어떤 물체에 대한 **공간 정보를 가지고 있었고** 하위 레벨로 내려가면서도 그 공간 정보는 계속 가지고 있었습니다. 
- 이런 공간 정보를 중간에 잃어버리게 되는 데 그 시점이 바로 `fully connected layer`가 되게 됩니다.
    - 왜냐하면 `fully connected layer`에서는 모든 노드들이 연결되어 버리기 때문입니다. (모든 노드들이 서로 곱해져서 더해지는 형태가 되지요)
- 따라서 classifier 용도로 사용한 `fully connected layer`를 사용하면 안되겠다고 생각하게 됩니다.

<br>

- 그래서 `fully connected layer`를 대신하여 [NIN(Network In Network, 1x1 Network)](https://gaussian37.github.io/dl-dlai-network_in_network/)를 사용하게 됩니다.
    - `NIN`은 현재 효율적인 네트워크 설계를 위해 많이 사용되었고(**차원 축소 및 연산량 감소**) 위에서 언급한 `Inception`에서도 사용되었습니다.
    - `NIN`은 이름 그대로 네트워크 에서 Multi layer perceptron의 역할을 수행하고 있습니다. (위 그림 참조)
- `Segmentation`을 처리하기 위해서 공간 정보를 유지해야 하기 때문에 `fully connected layer` 자리 대신 `NIN`을 넣게 되면 위 그림의 아래와 같이 volume 형태의 출력을 얻을 수 있습니다.
- 이 결과를 heatmap으로 그려보면 **공간 정보가 유지되고 있음을 확인**할 수 있습니다.

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 다룬 내용을 좀 더 입체적으로 표현한 것입니다.
- 위 그림의 아래 네트워크가 `FCN`에서 사용하는 방식의 앞부분 입니다. 즉, `fully connected layer`가 사라진 것이지요. 대신에 `NIN`을 사용해서 차원을 축소하였습니다. 
- 이것을 이미지 크기로 복원하려면 다시 `upsample` 하는 작업이 필요한데, 그것은 아래 글에서 계속 알아보겠습니다.
- 먼저 여기까지 한 작업을 보면 마치 정보를 압축하는 `encoder` 역할을 한것으로 볼 수 있습니다. `featuer`를 추출한 것으로 볼 수 있습니다. 

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 참고로 segmentation에서 나타나는 전체적인 구조에서 네트워크의 사이즈가 줄어들었다가 다시 입력크기로 크게 만들때, 각 layer의 feature들의 해상도가 다른데, 이것들을 마지막 layer에서 concat을 하는 등의 합치는 작업을 하면 성능 개선에 도움이 되는 트릭을 사용하였습니다.
    - 이 트릭을 `fuse feature into depp jet` 이라고 하며 object detection의 `ssd`에서도 사용되었습니다. 
- 해상도 관련 문제는 segmentaion 결과가 뭉게진 형태인 것으로 나타나는 문제인데 관련 그림은 아래 `skip connection`에서 확인해 보시면 됩니다.

<br>

## **FCN 구조 설명 - upsampling**

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 지금까지 얘기한 것이 `downsampling`이었고 이제 downsample한 feature를 이미지 크기만큼 다시 `upsampling` 하는 방법에 대하여 다루어 보겠습니다.
- feature의 크기를 다시 크게 하고 싶을 때 가장 쉽게 생각 할 수 있는 것은 bilinear interpolation 같은 방법일 수 있습니다.
    - 물론 이런 간단한 방법으로는 성능이 나오지 않기 때문에 다른 방법이 고안되었는데요..
- encoder 단에서 convolution 연산을 하여 feature를 압축 시킬 때 필터의 parameter를 학습하듯이 decoder 단에서 `deconvolution` 연산 이란 것을 해보고 그 결과 feature를 다시 팽창 시킬 때에도 **parameter를 학습**해보자는 것이 컨셉입니다.
    - inference 결과를 보면 이 방법이 훨씬 효과적인 것을 확인할 수 있고 직관적으로도 좀 더 딥러닝 네트워크에 가까운 것을 알 수 있습니다. 

<br>

## **FCN 구조 설명 - skip connection**

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 설명한 해상도 문제가 위 그림의 아래 부분과 같습니다.
- 가장 왼쪽의 segmentation 결과를 보면 클래스별로 잘 구분해서 segmentation 하였지만 저해상도와 같이 뭉개져 있는 느낌이 들고 세세한 부분은 부정확하게 segmentation이 되어있습니다.
- 그래서 고해상도의 이미지 정보를 deconvolution 할 때 사용할 수 있도록 바로 전달해 주는 `skip connection`을 만들어 문제를 개선하였습니다. 
- 위 그림을 보면 각 `대칭`되는 네트워크 구조에 따라 각 단계별로 `skip connection`이 만들어 지는 것을 볼 수 있고 skip connection을 여러개 넣을 때 더 성능이 좋아지는 것을 볼 수 있습니다.
- 그러면 `skip connection`을 하는 자세한 방법에 대하여 바로 아래에서 다루어 보도록 하겠습니다.

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 

<br>
<center><img src="..\assets\img\vision\segmentation\fcn\6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

