---
layout: post
title: 베이즈 필터 (Bayes Filter)
date: 2020-02-01 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [Optimal State Estimation, 최정 상태 이론, 베이즈 필터, Bayes filter] # add tag
---

<br>

- 선수 지식 1 : [상태 방정식 (state equation)](https://gaussian37.github.io/autodrive-ose-state_equation/)
- 선수 지식 2 : [자율주행에서의 Localization과 Tracking](https://gaussian37.github.io/autodrive-ose-localization_and_tracking/)
- 선수 지식 3 : [베이지안 통계](https://gaussian37.github.io/math-pb-easy_bayes/)

<br>

- 참조 1 : Probabilistic Robotics (Sebastian Thrun)
- 참조 2 : 순천향 대학교 박성근 교수님 강의

- 이번 글에서는 Bayes Filter에 대하여 다루어 보도록 하겠습니다. Bayes Filter는 Kalman Filter와 Particle Filter의 개념의 기초가 되는 간단하지만 매우 중요한 Filter입니다. 따라서 Kalman Filter나 Particle Filter를 배우기 전 단계라면 베이즈 필터를 통하여 어떻게 동작하는 지 그 메커니즘을 먼저 배우기를 추천합니다.
- 만약 Kalman Filter와 Particle Filter에 대하여 잘 아신다면 아래 Bayes Filter의 개념과 비교해 보는 것을 추천 드립니다.

<br>

## **목차**

<br>

- ### [Bayes Filter 개념](#bayes-filter-개념-1)
- ### [로봇 이동 예제를 통한 Bayes Filter 이해](#로봇-이동-예제를-통한-bayes-filter-이해-1)
- ### [Bayes Filter 알고리즘의 이해](#bayes-filter-알고리즘의-이해-1)
- ### [Bayes Filter 수식의 이해](#bayes-filter-수식의-이해-1)
- ### [예제를 통한 Bayes Filter 수식의 이해](#예제를-통한-bayes-filter-수식의-이해-1)
- ### [자동차 이동 예제를 이용한 Bayes Filter의 구체적인 이해](#자동차-이동-예제를-이용한-bayes-filter의-구체적인-이해-1)
- ### [Bayes Filter의 한계와 개선 방법](#bayes-filterd의-한계와-개선-방법-1)

<br>

## **Bayes Filter 개념**

<br>

- 먼저 이 글에서 다룰 `Bayes Filter`는 `Bayes Theorem`을 `Recursive`하게 따릅니다. 
- 즉, **확률 기반의 Recursive한 Filter**라고 보시면 됩니다. Bayes Filter의 식은 다음과 같습니다.

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이러한 Bayes Filter의 목적을 단 한줄로 표현하면 다음과 같습니다.

<br>

- $$ bel(x_{t}) = p(x_{t} \vert z_{1:t}, u_{1:t}) $$

<br>

- 앞의 사전 지식을 잘 이해하고 오셨으면 위 식의 뜻은 충분히 이해가 가실 것으로 생각됩니다.
- 위 식을 다시 한번 해석하면 $$ x_{t} $$는 t시점의 `상태 값`을 $$ z_{t} $$는 `센서 입력 값`을 뜻하고 $$ u_{t} $$는 `제어 입력값`을 뜻합니다.
- 따라서 **알고리즘의 시작부터 현재 시점 $$ t $$ 까지의 센서 입력값과 제어 입력값을 이용하여 현재 상태를 확률적으로 추정하는 것**으로 해석할 수 있습니다.

<br>

- Bayes Filter를 조금 더 자세하게 들여다 보면 Bayes Filter는 2가지 역할을 합니다.
- 첫번째는 **이전 상태를 이용하여 현재 상태를 예측**하는 역할이고 두번째는 **Bayes Rule을 이용하여 센서 입력값을 현재 상태에 반영**하는 역할입니다.
- 첫번째 케이스를 `물리 모델을 통한 예측`이라고 하겠습니다. 왜냐하면 이전 상태에 물리 모델의 수식을 통해 현재 상태를 예측하기 때문입니다. 이 때, `제어값`이 사용됩니다. 그리고 두번째 케이스는 `센서값`이라고 하겠습니다.
- 자동차가 이동하는 사례를 이용하여 예를 들어보겠습니다. 이 때, 상태의 정의는 자동차의 위치라고 가정하겠습니다. 
    - `물리 모델을 통한 예측` : 고속도로에서 이동 중인 자차의 1초 전의 상태에서 현재 상태를 예측할 때, 자동차의 1초 전의 위치와 자동차의 속도 정보 및 추가적으로 입력된 엑셀 및 브레이크 `제어값`을 이용하여 자동차의 현재 위치를 예측할 수 있습니다.
    - `센서값` : GPS를 이용하여 자동차의 현재 위치를 알 수 있습니다.
- Bayes Filter는 위 값 중 하나만을 사용하는 것이 아니라 두가지 값을 모두 사용하여 하나의 값으로 퓨전하여 사용합니다. 예를 들면 ① 현재 위치를 이전 상태와 속도 물리 모델을 이용하여 먼저 예측하고 ② 그 위치에 센서값을 통해 보완합니다.
- 이 퓨전 방법을 통하여 `물리 모델을 통한 예측`만 사용하거나 `센서값`만 사용하는 것 보다 더 좋은 성능으로 현재 상태를 추정할 수 있습니다.
- 위 두가지 값을 퓨전할 때, 노이즈가 작은 값을 더 신뢰하여 반영하는 것이 높은 성능을 확보하기 위해 고려할 점입니다. 경우에 따라서 물리 모델이 정확할 수 있고 센서의 성능이 좋을 수도 있기 때문에 이것은 상황에 맞춰서 적용해야합니다.
    - 이 내용은 선수 지식 1의 상태 방정식 내용을 자세히 읽어보시길 권장드립니다.

<br>

## **로봇 이동 예제를 통한 Bayes Filter 이해**

<br>

- 앞에서 설명한 컨셉을 Probabilistic Robotics의 로봇 예제를 이용하여 알아보도록 하겠습니다.

<br>

#### **Step 1**

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 로봇이 어느 위치에 있는 지 확률 분포 값으로 표현합니다. 위 그래프에서 $$ x $$ 가로축과 $$ bel(x) $$ 세로축으로 이루어진 확률 분포를 살펴보면 모든 위치가 동일한 확률 값을 아주 낮게 가집니다. 즉, 초기값은 어떠한 정보가 없으므로 uniform 분포를 가지게 됩니다.
- 각 문 앞에는 센서가 있습니다. 문 앞의 센서는 로봇이 지나갈 때, 3개의 문 중 하나에 로봇이 있다고 판단을 할 수 있으나 어느 문 앞에 있는 지는 판단할 수는 없습니다. 즉, **로봇이 문 앞에 있을 때, 3개의 문 모두 동일한 분포를 가지게 됩니다.**
- 로봇은 왼쪽에서 오른쪽으로 이동 중에 있습니다.

<br>

#### **Step 2**

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 새로운 확률 분포에 대한 개념을 정의해 보도록 하겠습니다.
- 　$$ p(z) $$ : 센서가 인지한 로봇 위치의 확률 분포
- 　$$ p(z \vert x) $$ : 로봇의 위치 상태가 $$ x $$로 주어졌을 때, 센서가 인지한 로봇 위치의 확률 분포
- 위 그림에서 $$ p(z \vert x) $$는 문앞에 센서가 있고 로봇이 문앞을 지나면 문앞에 로봇이 있을 확률이 증가하게 됩니다.
- 따라서 현재 시점의 최종 로봇 위치 상태에 대한 확률 분포는 문 앞에 있을 확률이 동등하게 분포하게 됩니다.
- 센서가 인지하였음에도 불구하고 문 앞의 확률 분포가 매우 좁게 (분산이 작게) 분포하지 못하는 이유는 **센서에 존재하는 노이즈 때문**입니다.

<br>

#### **Step 3**

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Step 2의 상태에서 Step 3로 로봇이 오른쪽으로 이동함에 따라 Step2에서 얻은 위치의 확률 분포가 로봇이 오른쪽으로 이동한 만큼 그대로 이동한 것을 확인할 수 있습니다.
- 이 결과는 앞에서 설명한 `물리 모델을 통한 예측`을 한 것으로 파악할 수 있습니다. 위치 속도 모델과 엑셀 및 브레이크와 같은 제어값을 이용하여 물체의 위치 상태를 예측한 것입니다.
- 그리고 단순히 오른쪽으로 분포가 이동한 것 뿐만 아니라 분포에 분산이 조금 더 커진 것을 알 수 있습니다. 빨간색 동그라미의 확률 분포에 비해서 파란색 동그라미의 확률 분포가 조금 더 퍼져 있는 것을 확인하시면 됩니다.
- 이와 같이 분산이 더 커진 이유는 제어값에 존재하는 노이즈가 추가되었기 때문입니다. 그 결과 Step 2의 센서 노이즈에 Step 3 의 제어 노이즈 까지 추가됩니다.

<br>

#### **Step 4**

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번 Step에서는 Step 3까지 반영한 $$ bel(x) $$에 센서가 인지한 로봇의 위치에 대한 확률 분포를 결합합니다.
- 위 그림과 같이 기존의 $$ bel(x) $$와 센서값 $$ p(z \vert x) $$가 결합하여 새로운 $$ bel(x) $$를 계산합니다.
- 계산 결과 빨간색 동그라미의 확률과 파란색 동그라미의 확률이 겹치게 되어 새로운 $$ bel(x) $$에서는 로봇의 위치 확률이 커지고 분산이 작아지게 됩니다.

<br>

#### **Step 5**

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번 Step에서도 Step 3과 같이 `물리 모델을 통한 예측`을 합니다. 따라서 로봇이 이동한 거리 만큼 확률 분포가 이동한 것을 확인할 수 있습니다.
- 또한 Step 3에서와 같이 물리 모델에 사용된 제어값에 대한 노이즈가 기존 확률 분포에 추가되어 각 위치 별 분포의 분산이 커진 것을 확인 할 수 있습니다.

<br>

- 지금까지 확인한 내용이 Bayes Filter의 가장 기본적인 컨셉입니다. 
- 상태가 변화할 때, `물리 모델을 통한 예측`을 통하여 현재 상태를 예측하고, 이와 연관된 **센서값이 있으면** 센서값을 통하여 현재 상태를 보완해 줍니다.
- 이 과정을 좀 더 상세하게 알아보도록 하겠습니다.

<br>

## **Bayes Filter 알고리즘의 이해**

<br>

- 로봇 이동 예제를 통하여 확인한 Bayes Filter 알고리즘을 정형화 하여 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 핵심이 되는 알고리즘은 for문 안의 2줄 밖에 되지 않을 정도로 매우 간단합니다. 
- for 문의 첫번째 줄은 `contorl update` 또는 `prediction`이라고 불립니다. 앞의 예제에서 `물리 모델을 통한 예측` 부분에 해당하는 알고리즘 입니다. `control update` 라고 불리는 이유는 제어값을 사용하여 상태를 업데이트 하였기 때문이고 `prediction` 이라고 하는 이유는 물리 모델과 제어값을 이용하여 `예측` 하였기 때문입니다.
- for 문의 두번째 줄은 `measurement update` 또는 `correction`이라고 불립니다. 앞의 예제에서 `센서값`을 통한 현재 상태 보정에 해당하는 알고리즘 입니다. `measurement update` 라고 불리는 이유는 센서를 통해 `측정`한 값을 사용하여 업데이트 하였기 때문이고 `correction`이라고 하는 이유는 prediction한 결과를 센서값을 통하여 보정하기 때문입니다.

<br>

## **Bayes Filter 수식의 이해**

<br>

<br>

## **예제를 통한 Bayes Filter 수식의 이해**

<br>

<br>

## **자동차 이동 예제를 이용한 Bayes Filter의 구체적인 이해**

<br>

## **Bayes Filter의 한계와 개선 방법**

<br>
