---
layout: post
title: 선형 칼만 필터로 human tracking (w/ opencv)
date: 2019-11-19 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [칼만 필터, kalman filter, tracking, 트래킹] # add tag
---

<br>

- 이번 글에서는 opencv를 이용하여 칼만필터를 사용하는 방법에 대하여 알아보겠습니다.
- 전체적으로는 2D 이미지에서 컴퓨터 비전 알고리즘으로 Detection한 포인트를 선형 칼만 필터 알고리즘으로 tracking 할 예정입니다.
- 선형 칼만 필터의 자세한 원리를 이해하고 싶으면 제 블로그의 [이 링크](https://gaussian37.github.io/ad-ose-lkf_basic/)를 참조하시기 바랍니다. 이 글에서는 간략하게만 이해하고 넘어갈 예정입니다.
- 이 글에서는 opencv를 이용하여 어떻게 칼만 필터를 사용하는 지 관점으로 접근해 보도록 하겠습니다.

<br>

## **목차**
    - ### 칼만 필터의 역할
    - ### State Transition Equation
    - ### Uncertainties 모델링
    - ### Control Input (optional)
    - ### Uncontrolled uncertainties
    - ### Measurements
    - ### Fusing information
    - python & opencv 코드
    - c++ & opencv 코드 

<br>

## **칼만 필터의 역할**

<br>

- 칼만 필터는 2가지 전략을 따릅니다.
- 1) **predict** : 시스템의 `내부 상태`에 대하여 예측을 합니다. (e.g. 드론의 위치와 속도) 이 예측은 `이전 내부 상태`와 `센서 등으로 부터 입력`된 값 (e.g. 드론의 프로펠러 force)을 기반으로 만들어집니다.
- 2) **update** : 내부 상태에 관련된 새로운 계측값 (e.g. GPS 정보)이 들어오면 이 값을 사용하여 **predict**에 반영합니다.

<br>

## **State Transition Equation**

<br>

- 먼저 칼만 필터에서 **predict**에 관련된 과정부터 이해해 보도록 하겠습니다. 드론의 예를 들어서 한번 접근해 보도록 하겠습니다.
- 드론이 비행을 할 때, 드론의 위치를 정확히 알아야 합니다. $$ k $$ 번째 스텝에서의 드론의 위치를 알고 싶으면 이전 스텝인 $$ k - 1 $$ 스텝에서의 정보를 이용하여 predict 해야 합니다.
    - 먼저 $$ k - 1 $$ 번째의 드론의 `위치`
    - 그리고 $$ k - 1 $$ 번째의 드론의 `속도` 
    - 그리고 $$ k - 1 $$ 번째와 $$ k $$ 번째 사이의 `시간 간격`
- 위 3가지 정보를 이용하여 설계를 해보겠습니다. 문제를 단순화 시키기 위하여 acceleration은 고려하지 않고 등속으로 간주하겠습니다.

<br>

$$ \begin{equation} P_{k}  = P_{k-1} + v_{k-1} \Delta t \end{equation} \tag{1} $$

<br>

$$ v_{k} = v_{k-1} \tag{2} $$ 

<br>

- 여기서 $$ P $$는 3차원의 `위치`이고 $$ v $$는 3D `속도`벡터이고 $$ \Delta t $$는 `시간 간격` 입니다.
- 그리고 $$ P_{k} $$와 $$ v_{k} $$는 시스템의 **state(상태)**라고 하는데 $$ P_{k} $$와 $$ v_{k} $$는 벡터이고 각각의 원소가 무엇을 의미하는 지 알아보겠습니다.
- 먼저 $$ P = [x, y, z]^{T} $$는 각 $$ x, y, z$$ 축의 위치를 나타냅니다.
- 그리고 $$ v = [u, v, w]^{T} $$는 각 $$ x, y, z$$축의 방향 속도를 나타냅니다.
- 예를 들어 $$ u $$는 $$ x $$ 방향으로의 속도를 뜻합니다.

<br>

- 위의 (1), (2) 식을 결합해서 한 개의 **state transition equation**을 만들 수 있는데 다음 식과 같습니다.

<br>

$$ \begin{bmatrix} P_{k} \\ v_{k} \end{bmatrix}  = F_{k} \begin{bmatrix} P_{k-1} \\ v_{k-1} \end{bmatrix} \tag{3} $$

<br>

- (3) 식의 의미를 보면 $$ k-1 $$ 상태에서 $$ F_{k} $$를 곱하면 $$ k $$ 상태가 된다라고 해석할 수 있습니다.
- 그러면 $$ F_{k} $$는 무슨 뜻일까요? $$ F_{k} $$는 **state transition matrix** 라고 하고 그 의미는 아래 식을 통하여 확인할 수 있습니다.

<br>

$$ \begin{bmatrix} x_{k} \\ y_{k} \\ z_{k} \\ u_{k} \\ v_{k} \\ w_{k} \\ \end{bmatrix}  = 
\begin{bmatrix}
1 & 0 & 0 & \Delta t & 0 & 0 \\
0 & 1 & 0 & 0 & \Delta t & 0 \\
0 & 0 & 1 & 0 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix} x_{k-1} \\ y_{k-1} \\ z_{k-1} \\ u_{k-1} \\ v_{k-1} \\ w_{k-1} \\ \end{bmatrix}
\tag{4} $$

<br>

- 위 매트릭스를 보면 각 축의 이전 스텝의 위치에서 시간 간격 $$ \Delta t $$에 그 축의 속도 만큼 곱하여 이동 거리를 구하면 현재 스텝의 위치를 예측 할 수 있다는 것으로 해석할 수 있습니다.
- 그러면 식을 간단하게 표현하기 위하여 다음과 같이 정리하겠습니다.

<br>

$$ x_{k} = \begin{bmatrix} p_{k} \\ v_{k} \end{bmatrix} $$
$$ x_{k} = F_{k} x_{k-1} \tag{5} $$

<br>

## **Uncertainties 모델링**

<br>

- 앞의 과정에서 위치와 속도를 **predict** 하였는데 그럼에도 이 예측은 다소 확률적입니다. 왜냐하면 우리의 예측에 다소 불확실적인 요소(Uncertainties, 불확실성))들이 있기 때문입니다. 바로 `노이즈` 들입니다.
- 불확실성은 주로 가우시안 분포를 이용하여 모델링 됩니다. (가우시안 분포에는 평균 $$ \mu $$와 분산 $$ \Sigma $$가 있습니다.)
- 먼저 드론의 예에서 상태 벡터는 6개의 요소를 가지고 있습니다. 세개는 속도였고 세개는 위치였습니다. 
- 이 경우에 (6 x 1)의 `평균 벡터`와 (6 x 6)의 `공분산 행렬`다변량 가우시안 분포를 이용하여 모델링 해야 합니다. 
    - 여기서 (6 x 1) 의 분산 벡터를 사용하지 않고 (6 x 6) 크기의 공분산 행렬을 사용하는 이유는 correlation을 고려하기 위해서 입니다. 
    - 예를 들어 $$ x $$차원의 위치는 $$ x $$ 차원의 속도와 correlation이 있는데 공분산 행렬의 **대각선 이외의 요소**가 이런 correlation 관계를 캡쳐합니다.

<br>

- 앞에서 다루었듯이 $$ x_{k} $$를 **predict** 하는 방법은 확인하였는데 공분산 행렬 $$ P_{k} $$는 어떻게 **predict**하면 될까요? 
- 공분산 행렬의 성질에 따라서 랜덤 확률 변수 $$ y $$가 공분산 $$ \Sigma $$를 가질 때, 랜덤 확률 변수 $$  y $$에 임의의 행렬 $$ A $$를 곱하여 $$ Ay $$를 만든다면 공분산 행렬은 $$ A\Sigma A^{T} $$가 됩니다. (이유는 [링크](https://gaussian37.github.io/ad-ose-lkf_basic/)에서 참조 하시기 바랍니다.)
- 따라서 이 논리에 따라 $$ P_{k} = F_{K} P_{k-1} F_{k}^{T} \tag{6} $$로 업데이트 될 수 있습니다.
- 여기 까지 정리하면 **predict**하는 방법은 다음과 같습니다.

<br>

$$ x_{k} = F_{k} x_{k-1} \tag{5} $$
$$ P_{k} = F_{K} P_{k-1} F_{k}^{T} $$

<br>

## **Control Input (optional)**

<br>

- 여기 까지 이전 스텝의 상태를 이용하여 드론의 위치와 속도를 **predict**하는 방법 까지 다루었습니다.
- 