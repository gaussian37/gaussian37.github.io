---
layout: post
title: 상태 방정식 (state equation)
date: 2020-02-01 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [Optimal State Estimation, 최정 상태 이론, 상태 방정식] # add tag
---

<br>

[Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>

- 최적 상태를 추적하기 위하여 칼만 필터와 같은 estimator를 사용할 때, 모델링을 하기 위하여 상태 방정식 (state equation)을 작성할 필요가 있습니다.
- 이번 글에서는 나름 쉬운(?) 상태 방정식 하나를 예를 들어 상태 방정식을 어떻게 작성하는 지 살펴보겠습니다.
- 이 글에서 다루는 상태는 **1차원에서 직선으로 움직이는 어떤 물체의 움직임**입니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이 글에서 최종적으로 살펴볼 상태 방정식은 위와 같습니다.
- 바로 보면 이해가 안되므로 각 성분에 대해서 살펴보고 다시 이 식을 살펴보도록 하겠습니다.
 
<br>
<center><img src="../assets/img/autodrive/ose/state_equation/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 각 식에 들어가 있는 모든 성분의 의미를 분석해 보면 위와 같습니다.
- 여기서 $$ k $$는 몇 번째 step인 지에 해당합니다. 
- 　$$ x(k) $$는 k번째 step의 물체의 위치 상태에 해당합니다.
- 　$$ \dot{x}(k) $$는 k번째의 step의 물체의 속도에 해당합니다.
- 　$$ \ddot{x}(k) $$는 k번째의 step의 물체의 가속도에 해당합니다.
- 　$$ T $$는 각 step 간 시간 간격을 뜻합니다.

<br>

- 그러면 첫번째 식부터 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이 식은 이전 스텝 (k-1)의 위치 상태인 $$ x(k-1) $$에 이전 스텝 (k-1)과 현재 스텝의 시간 간격($$ T $$)에 이전 스텝의 속도($$ \dot{x}(k-1) $$)를 곱한 값을 더하여 현재 위치를 추정합니다. 이것은 **이동 거리 = 속도 * 시간**을 이용한 것입니다. ($$ T\dot{x}(k-1) $$)
- 이 때, 이전 스텝의 가속도를 이용하여 위치 상태를 좀 더 정확하게 추정할 수 있습니다. 이것은 **1/2 * 가속도 * 시간^2**을 이용하는 것입니다. ($$ \frac{T^{2}}{2}\ddot{x}(k-1)$$)
- 마지막으로 현재 스텝에서 주어지는 `control input` 까지 추가하여 현재 위치를 추정할 수 있습니다.
- `contol input`은 Localization 문제에서는 주어질 수 있으나 Tracking 문제에서는 알 수 없는 한계가 있습니다. 관련 내용은 이 [링크](https://gaussian37.github.io/autodrive-ose-localization_and_tracking/)를 살펴보시길 바랍니다.
- control input이 주어졌다고 하였을 때, 이 값은 $$ u(k) $$로 나타내겠습니다. 위 식에서는 이번 스텝에서 추가로 입력된 가속도라고 생각하시면 됩니다.
- 그러면 여기서 $$ w(k) $$는 무엇일까요? 바로 이 `control input`의 `noise`입니다.
- 위 물체가 자동차라고 하였을 때, 가속도를 주기 위해 엑셀을 밟아야 하는데 정확히 $$ u(k) $$ 만큼의 가속도를 줄 정도로 엑셀을 밟는 것은 사실 매우 어렵습니다. 따라서 이런 `contorl input`에는 `noise`가 발생하는 데 이 노이즈를 고려해 주어야 하기 때문에 $$ w(k) $$를 같이 고려해 주어야 합니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 다음은 속도에 대한 식을 다루어 보겠습니다. 위 식의 각 항의 의미는 앞에서 다 다루었기 때문에 생략하겠습니다.
- **속도 = 현재 속도 + 가속도 * 시간**의 물리 법칙을 따르기 떄문에 위 식과 같이 적을 수 있습니다. 
- 물론 앞에서 다루었듯이 contrl input에 대한 노이즈 까지 같이 고려하였습니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 마지막으로 가속도에 대한 식이 위와 같습니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 식에서 $$ z(k) $$는 위치를 가리키는 센서 데이터를 나타냅니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞에서 다룬 식을 행렬로 묶으면 위와 같이 묶을 수 있습니다. 여기서 $$ A $$ 행렬을 `state matrix`라고 부릅니다. state equation에서 state를 모델링 할 때 핵심이 되는 행렬입니다.
- 그리고 $$ B $$ 행렬을 `control matrix`라고 합니다. control input을 어떻게 줄 지에 대한 정의를 내려주기 때문입니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 식의 $$ C $$ 행렬을 `transition matrix` 라고 합니다. 이 행렬은 **상태 추정식에 존재하는 차원 3개(위치, 속도, 가속도)를 센서 데이터의 차원(위치)과 맞추어** 줍니다. 일반적으로 모델링한 식의 차원의 수가 더 많고 센서 데이터의 차원의 수가 더 작으므로 높은 차원을 낮은 차원으로 매핑 하는게 쉬우므로 모델링 식의 차원을 센서 데이터의 차원으로 맞춰줍니다.


<br>
<center><img src="../assets/img/autodrive/ose/state_equation/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

[Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>