---
layout: post
title: An Alternative to the EKF - The Unscented Kalman
date: 2020-02-04 00:00:00
img: autodrive/sdcc/sdcc.png
categories: [autodrive-sdcc] 
tags: [self driving cars, unscented kalman, 무향 칼만 필터] # add tag
---

<br>

- 참조 : self-driving cars specialization, coursera
- 이번 글에서는 `Unscented Kalman Filter`에 대하여 다루어 보도록 하겠습니다.

<br>

- 이전 글에서 선형화 오류로 인해 `EKF`가 상태의 실제 값과 매우 다른 상태 추정치 및 상태의 불확실성을 정확하게 포착하지 못하는 공분산을 생성하는 방법을 보았습니다.
- 자율주행차와 같은 안전에 중요한 응용 분야에서 EKF에 의존 할 때 이는 큰 문제가 될 수 있습니다.
- 이번 글에서는 비선형 함수를 통해 확률 분포를 전달하기 위해 `Unscented Transform`이라고하는 비선형 칼만 필터링에 대한 대체 접근 방식인 `Unscented Kalman Filter`에 대해 학습합니다.
- 살펴보겠지만 `Unscented Transform`은 (EKF보다는 계산량이 많지만) 비슷한 양의 계산을 위해 `Jacobian`을 계산하지 않고도 `EKF` 스타일 선형화보다 훨씬 높은 정확도를 제공합니다.
- 이번 글의 목적은 Unscented Transform을 사용하여 비선형 함수를 통해 확률 분포를 전달하고 Unscented Kalman Filter 또는 `UKF`가 **prediction** 및 **correction** 단계에서 Unscented Transform을 사용하는 방법을 설명하고 이해하는 것입니다. 더 나아가 EKF에 비해 UKF의 장점뿐만 아니라 UKF를 단순한 비선형 tracking 문제에 적용해 보는 것입니다.
- Unscented Transform은 매우 직관적인데 왜냐하면 임의의 비선형 함수를 근사화하는 것보다 확률 분포를 근사화하는 것이 일반적으로 훨씬 쉽기 때문입니다.

<br>
<center><img src="../assets/img/autodrive/sdcc/ukf/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 왼쪽에있는 것과 같은 가우시안 분포가 비선형 함수를 통해 오른쪽에 있는 것과 같은 더 복잡한 분포로 변환되는 간단한 예를 생각해 봅시다.
