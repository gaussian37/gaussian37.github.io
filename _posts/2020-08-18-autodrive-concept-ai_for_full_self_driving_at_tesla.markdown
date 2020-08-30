---
layout: post
title: AI for Full-Self Driving at Tesla
date: 2020-08-18 00:00:00
img: autodrive/concept/ai_for_full_self_driving_at_tesla/0.png
categories: [autodrive-concept] 
tags: [테슬라, tesla, 자율주행, 자율주행 자동차, autodrive, self-driving] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/hx7BXih7zx8" frameborder="0" allowfullscreen="true" height="400px" width="600px"> </iframe>
</div>
<br>

- 이 글은 안드레이 카파시가 [scaledml2020](http://scaledml.org/2020/)에서 테슬라의 자율주행에 관해 발표한 내용을 정리한 것입니다.

<br>

## **목차**

<br>

- ### 테슬라의 오토파일럿이란?
- ### (라이다 방식이 아닌) 컴퓨터 비전 기반의 테슬라 방식
- ### 뉴럴 네트워크의 역할
- ### fleet 으로 부터 까다로운 케이스에 해당하는 이미지 획득
- ### 테스트를 위해선 loss function과 accruracy 평균만으로는 부족함
- ### HydraNet (48 network, 1,000 prediction, 70,000 hours train)
- ### Full self-driving을 위한 neural network
- ### Self-supervised learning을 이용하여 이미지에서 depth를 예측하고 실제 거리를 측정하는 방법
- ### 다른 self-supervised learning의 사용 사례
- ### Q & A

<br>

## **테슬라의 오토파일럿이란?**

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 테슬라의 오토파일럿 기능에 대하여 설명합니다. 현재 사용 중인 오토파일럿의 이름은 `NoA(Navigation on Autopilot)`으로 내비게이션 기반의 오토파일럿입니다.
- 테슬라에서 모은 고객 데이터를 보면 전체 NoA를 사용한 이력이 10억 마일이 넘으며 20만건 이상의 자동 차선 변경이 있었으며 50개 이상의 나라에서 NoA를 사용중입니다.
- 위 기능은 테슬라 이외의 다른 제조업에서도 운영중에 있으며 비슷한 수준의 자율주행을 구현하고 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `Smart Summon`은 주차되어 있는 자동차를 나의 위치까지 호출하는 기능입니다. 다른 **제조사에 없는 이 기능을 테슬라는 제공**하고 있으며 테슬라의 다른 자료를 통해 살펴보면 SLAM 기술을 통하여 실시간 지도를 그려서 이 기능을 대응하는 것으로 보입니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞에서 다룬 자율주행을 위한 편의 기능 이외에도 안전 기능들도 지원하고 있으며 안전 기능의 경우 다른 제조사들도 많이 제공 하는 기능들인 긴급 제동, 충돌 방지, 회피 등의 기능을 제공하고 있습니다. 위 기능군들을 보면 다른 제조사 와의 차이점도 없어 보입니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 물론 이런 안전 기능은 법규 뿐 아니라 EURO NCAP과 같은 인증 기관으로 부터 좋은 점수를 받기 위한 것이 있습니다. 테슬라에서도 EURO NCAP에서 5점 받은 것을 자랑하였습니다.
- 이 외에도 보행자를 감지하여 긴급 정지하거나 테슬라가 주장하는 Full Self-Driving에 관한 영상을 통해 테슬라의 기술력을 자랑합니다. (3:25 ~ 5:30)

<br>

## **(라이다 방식이 아닌) 컴퓨터 비전 기반의 테슬라 방식**

<br>

- 앞에서 보여준 Full-Self Driving 데모와 같은 사례는 웨이모에서 훨씬 이전에 보여주었습니다.
- 웨이모에 비해 시기적으로 뒤쳐짐에도 불구하고 테슬라에서 구현한 기술을 자랑할 수 있는 이유는 라이다 없이 컴퓨터 비전 기반으로 구현한 점에 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 웨이모의 차에서는 차 윗쪽에 lidar가 달려 있고 lidar 즉, laser를 송/수신 하여 주위를 인식합니다.
- 라이다를 이용하면 lidar point cloud 라는 것을 만들게 됩니다. lidar point cloud는 라이다가 인식한 물체의 좌표들을 점의 형태로 나타낸 것입니다. lidar point cloud를 통하여 주변의 지도를 그릴 수 있기 때문에 주위 상황을 잘 파악할 수 있고 자차의 현재 위치를 찾는 Localize 문제를 보다 잘 해결할 수 있습니다.
- 또한 테슬라는 웨이모에서 적용한 HD(High Definition) Map 또한 사용하지 않고 실시간 컴퓨터 비전 기반으로 주변 상황을 인지하는 방법을 취합니다.
- ※ [테슬라와 웨이모의 차이점](https://gaussian37.github.io/autodrive-concept-waymo_vs_tesla/)은 저의 다른 글에서 좀 더 자세하게 다루었습니다.

<br>

## **뉴럴 네트워크의 역할**

<br>

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 테슬라에서는 현재 위 그림과 같은 대상 군들을 카메라를 이용하여 인식하고 있습니다.
- 이미지의 각 대상들의 클래스를 살펴 보면 교통 신호, 표지판, 차선, 연석 등과 같은 교통과 관련된 물체들을 인식하며 움직이는 물체와 정적인 물체들도 분리하여 구분하는 것을 알 수 있습니다. (물론 위 클래스는 발표 예시이므로 내부적으로 더 세세하게 분류하여 사용할 수 있을 것 같습니다.)
- 위와 같이 클래스들을 분류하여 인식하는 목적은 **① 자율주행을 위함** 과 **② 디스플레이에 표시**하여 사용자에게 보여주기 위함입니다.
- 인식 하는 예시는 다음 동영상 부터 참조하시면 됩니다. (7:55)

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/hx7BXih7zx8?t=474" frameborder="0" allowfullscreen="true" height="400px" width="600px"> </iframe>
</div>
<br>

- 동영상을 참조하면 테슬라에서 카메라로 인식하는 내용들을 참조하실 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 동영상에서 캡쳐한 위 이미지는 테슬라의 많은 카메라 중에서 전방 카메라에서 물체들을 인식하는 것을 나타낸 것입니다.
- 자세히 보면 앞에서 소개한 클래스들을 다양한 형태로 인식하는 것을 확인할 수 있습니다.

<br>

## **fleet 으로 부터 까다로운 케이스에 해당하는 이미지 획득**

<br>

- 뉴럴 네트워크를 이용하여 물체 인식에 대한 성능을 높였음에도 불구하고 성능을 떨어뜨리는 코너 케이스가 종종 발생하곤 합니다.
- 다음은 정지 표지판 (Stop)을 예시로 어떤 코너 케이스들이 발생할 수 있는 지 살펴보겠습니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 보면 다양한 코너 케이스의 정지 표지판 모양이 있을 수 있습니다. 상황에 따라 다르게 나타날 수 있는 정지 표지판 데이터들을 수집할 수 있어야 뉴럴 네트워크의 성능을 높일 수 있습니다.
- 테슬라에서는 이와 같은 코너 케이스들을 어떻게 다루는 지 소개합니다.

<br>
<center><img src="../assets/img/autodrive/concept/ai_for_full_self_driving_at_tesla/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이것은 Data Engine 이라는 프로세스 입니다. 위 그림의 구조와 같은 형태를 계속 반복하면서 데이터를 구축해 나아가는 방법입니다.
- 예를 들어 뉴럴 네트워크의 인식기 (detector)가 동작을 잘못하게되면 그 데이터를 전송 받게 되고 그 데이터의 정답 데이터를 만든 다음 재학습을 하게 합니다. 학습 이후 OTA(Over The Air)로 학습 결과를 배포하여 코너 케이스를 대응하도록 합니다.