---
layout: post
title: MobileNets - Efficient Convolutional Neural Networks for Mobile Vision Applications
date: 2017-04-17 00:00:00
img: dl/concept/mobilenet/mobilenet.PNG
categories: [dl-concept] 
tags: [python, deep learning, dl, MobileNet] # add tag
---

- 이번 글에서는 경량화 네트워크로 유명한 `MobileNet`에 대하여 알아보도록 하겠습니다.
    - 이번글은 `MobileNet` 초기 버전(v1) 입니다.
- 아래 글의 내용은 [PR12 모바일넷 설명](https://youtu.be/7UoOFKcyIvM) 내용을 글로 읽을 수 있게 옮겼습니다.

<br>

### **목차**

<br>

- 1.논문 리뷰
- 2.추가 설명 자료
- 3.Pytorch 코드 리뷰 

<br>

## **1. 논문 리뷰**

<br>


<br>

## **2. 추가 설명 자료**

<br> 


### 경량화 네트워크의 필요성

<br>

- 먼저 딥러닝의 상용화를 위하여 필요한 여러가지 제약 사항을 개선시키기 위하여 경량화 네트워크에 대한 연구가 시작되었습니다.
- 딥러닝을 이용한 상품들이 다양한 환경에서 사용되는데 특히, 고성능 컴퓨터가 아닌 상황에서 가벼운 네트워크가 필요하게 됩니다. 
- 예를 들어 데이터 센터의 서버나 스마트폰, 자율주행자동차 또는 드론과 같이 가격을 무작정 높일 수 없어서 제한된 하드웨어에 딥러닝 어플리케이션이 들어가는 경우입니다.
    - 이러한 경우에 실시간 처리가 될 정도 성능의 뉴럴넷이 필요하고 또한 얼마나 전력을 사용할 지도 고려를 해야합니다.    
- 이러한 제약 사항을 충분히 만족하면서 또한 아래와 같은 성능이 꽤 괜찮아야 어플리케이션에 적용을 할 수 있습니다.
    - 충분히 납득할만한 정확도
    - 낮은 계산 복잡도
    - 저전력 사용
    - 작은 모델 크기

<br>

- 그러면 왜 `Small Deep Neural Network`가 중요하게 되었을까요?
    - 네트워크를 작게 만들면 학습이 빠르게 될것이고 임베디드 환경에서 딥러닝을 구성하기에 더 적합해집니다.
    - 그리고 무선 업데이트로 딥 뉴럴 네트워크를 업데이트 해야한다면 적은 용량으로 빠르게 업데이트 해주어야 업데이트의 신뢰도와 통신 비용등에 도움이 됩니다. 
    
<br>

### Small Deep Neural Network 기법

<br>

- `Channel Reduction` : MobileNet 적용
    - Channel 숫자룰 줄여서 경량화
- `Depthwise Seperable Convolution` : MobileNet 적용
    - 이 컨셉은 `Xception`에서 가져온 컨셉이고 이 방법으로 경량화를 할 수 있습니다.
- `Distillation & Compression` : MobileNet 적용
- Remove Fully-Connected Layers
    - 파라미터의 90% 정도가 FC layer에 분포되어 있는 만큼 FC layer를 제거하면 경량화가 됩니다. 
    - CNN기준으로 필터(커널)들은 파라미터 쉐어링을 해서 다소 파라미터의 갯수가 작지만 FC layer에서는 파라미터 쉐어링을 하지 않기 때문에 엄청나게 많은 수의 파라미터가 존재하게 됩니다. 
- Kernel Reduction (3 x 3 → 1 x 1)
    - (3 x 3) 필터를 (1 x 1) 필터로 줄여서 연산량 또는 파라미터 수를 줄여보는 테크닉 입니다. 
    - 이 기법은 대표적으로 `SqueezeNet`에서 사용되었습니다.
- Evenly Spaced Downsampling
    - Downsampling 하는 시점과 관련되어 있는 기법입니다.
    - Downsampling을 초반에 많이 할 것인지 아니면 후반에 많이할 것인지 선택하게 되는데 그것을 극단적으로 하지 않고 균등하게 하자는 컨셉입니다.
    - 왜냐하면 초반에 Downsampling을 많이하게 되면 네트워크 크기는 줄게 되지만 feature를 많이 잃게 되어 accuracy가 줄어들게 되고
    - 후반에 Downsampling을 많이하게 되면 accuracy 면에서는 전자에 비하여 낫지만 네트워크의 크기가 많이 줄지는 않게 됩니다.
    - 따라서 이것의 절충안으로 적절히 튜닝하면서 Downsampling을 하여 Accuracy와 경량화 두 가지를 모두 획득하자는 것입니다.
- Shuffle Operation

<br>

- 특히 `MobileNet`에서 사용하는 핵심 아이디어는 `Depthwise Seperable Convolution`입니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet/1.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 `MobilNet`을 다루기 전에 간단하게 **Convolution Operation**에 대하여 다루어 보겠습니다.
- 위와 같이 인풋의 채널이 3개이면 convolution 연산을 하는 필터의 채널 또한 3개이어야 합니다.
- 이 때 필터의 갯수가 몇 개 인지에 따라서 아웃풋의 채널의 숫자가 결정되게 됩니다.
- 즉, 위의 오른쪽 그림과 같이 입력 채널에서는 필터의 크기 만큼 모든 채널의 값들이 **element-wise 곱**으로 연산하여 한 개의 값으로 모두 더해지게 됩니다.

<br>

## **3. Pytorch 코드 리뷰**

<br>
