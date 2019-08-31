---
layout: post
title: MobileNet
date: 2019-08-31 00:00:00
img: dl/concept/mobilenet/mobilenet.PNG
categories: [dl-concept] 
tags: [python, deep learning, dl, MobileNet] # add tag
---

- 이번 글에서는 경량화 네트워크로 유명한 `MobileNet`에 대하여 알아보도록 하겠습니다.
    - 이번글은 `MobileNet` 초기 버전(v1) 입니다.
- 아래 글의 내용은 [이진원님의 모바일넷 설명](https://youtu.be/7UoOFKcyIvM) 내용을 글로 읽을 수 있게 옮겼습니다.

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
- `Depthwise Seperable Convolution` : MobileNet 적용
- `Distillation & Compression` : MobileNet 적용
- Remove Fully-Connected Layers
    - 파라미터의 90% 정도가 FC layer에 분포되어 있는 만큼 FC layer를 제거하면 경량화가 됩니다. 
    - CNN기준으로 필터(커널)들은 커널 
- Kernel Reduction (3 x 3 → 1 x 1)
- Evenly Spaced Downsampling
- Shuffle Operation
