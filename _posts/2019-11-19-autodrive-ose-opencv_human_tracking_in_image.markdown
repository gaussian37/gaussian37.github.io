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

$$ \begin{equation} p_{k}  = p_{k-1} + v_{k-1} \Delta t \end{equation} \tag{1} $$

<br>

$$ v_{k} = v_{k-1} \tag{2} $$ 

<br>

- 여기서 $$ p $$는 3차원의 `위치`이고 $$ v $$는 3D `속도`벡터이고 $$ \Delta t $$는 `시간 간격` 입니다.
- 그리고 $$ p_{k} $$와 $$ v_{k} $$는 시스템의 **state(상태)**라고 합니다.
- 위의 (1), (2) 식을 결합해서 한 개의 **state transition equation**을 만들 수 있는데 다음 식과 같습니다.

<br>

$$ \begin{bmatrix} p_{k} \\ v_{k} \end{bmatrix}  = F_{k} \begin{bmatrix} p_{k-1} \\ v_{k-1} \end{bmatrix} \tag{3} $$

<br>


