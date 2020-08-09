---
layout: post
title: 자율주행에서의 Localization과 Tracking
date: 2020-02-01 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [Optimal State Estimation, 최정 상태 이론, Localization, Tracking] # add tag
---

<br>

[Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>

- 이번 글에서는 간단하게 자율주행에서의 Localization과 Tracking의 차이에 대하여 알아보도록 하겠습니다.
- 먼저 `Localization` 문제는 자차의 키네마틱 정보를 인식하는데 이 때, `센서값`과 `컨트롤 입력값`을 사용하는 케이스를 뜻합니다.
- 이 때 사용하는 센서값과 컨트롤 입력에 노이즈가 발생할 수 있다는 것을 전제로 하며 이 센서값과 컨트롤 입력값을 이용하여 더 정확하게 자차의 키네마틱 정보를 예측하는 것이 목표이고 이를 재귀적으로 구현합니다.

<br>

- 반면 `Tracking` 문제는 위의 `Localization` 문제에서 정보 하나가 빠지게 됩니다. 바로 `컨트롤 입력값` 입니다.
- `Tracking`이라고 하면 자차의 상태가 아니라 주변 차 또는 자차 이외의 무언가의 상태를 추정하는 것이기 때문에 `Localization` 때와 같이 컨트롤 입력값을 받을 수 없습니다.
- 예를 들어 `Localization`은 **컨트롤 입력값**을 받기 때문에 예를들어 Accelerator 페달을 얼만큼 밟았는 지, 휠을 몇 도 움직였는 지 등을 알 수 있습니다.
- 반면 `Tracking`에서는 컨트롤 입력값이 빠지기 때문에 `Localization`에 비해 다소 문제가 어려워 집니다.

<br>

- 이를 확률적인 관적에서 다루어 보겠습니다. 먼저 `Localization` 입니다.
- $$ x_{t} $$는 t 시점의 위치, $$ z_{t} $$는 t 시점의 센서값, $$ u_{t} $$는 t 시점의 컨트롤 입력값 입니다.

<br>

- $$ p(x_{t} \vert x_{0:t-1}, z_{1:t-1}, u_{1:t}) = p(x_{t} \vert x_{t-1}, u_{t}) $$

<br>

- 위 식의 좌변은 처음 시점 부터 직전 시점(t-1)의 위치와 센서값 그리고 컨트롤 입력값을 이용하여 현재 시점 t에서의 위치를 추정하는 확률식입니다.
- 우변을 보면 조건부에서 위치값인 $$ x_{t-1} $$ 이외에 $$ x_{0} $$ ~ $$ x_{t-1} $$까지는 모두 생략되었는데 그 이유는 **독립이라는 가정하에 직전 데이터인 $$ x_{t-1} $$와 $$ u_{t} $$를 제외하고 모두 생략**하였습니다. (물론 상황에 따라 종속적인 경우가 있다면 종속적이라 생각되는 케이스는 추가로 곱해주어도 상관없습니다.)
- 이런 성질을 `Markov Property` 라고 합니다. 즉 **현재 상태는 직전 상태에 영향을 받는다**는 가정을 통하여 문제를 단순화 시킨 것입니다.
    - 즉, $$ P(x_{t} \vert x_{0}, x_{1}, ..., x_{t-1} ) \approx P(x_{t} \vert x_{t-1}) $$로 가정하는 것입니다.
    - 간단하게 생각하면 직전 상태인 $$ P(x_{t-1}) $$은 **그 이전의 모든 상태를 함축**하고 있다고 보는 것입니다.

<br>

- $$ p(z_{t} \vert x_{0:t}, z_{1:t-1}, u_{1:t}) = p(z_{t} \vert x_{t}) $$

<br>

- 위 식은 앞에서 구한 $$ x_{t} $$를 기반으로 센서값인 $$ z_{t} $$를 업데이트 합니다.
- 필요한 $$ x_{t} $$와 $$ z_{t} $$를 구하였기 때문에, 최종적으로 믿고 싶은 위치($$ \text{bel} $$은 belief를 뜻함)인 $$ bel(x_{t}) $$는 다음과 같습니다.

<br>

- $$ bel(x_{t}) = p(x_{t} \vert z_{1:t}, u_{1:t}) $$

<br>

- 그러면 `Localization`과 `Tracking`의 차이는 컨트롤 입력값을 직접적으로 받는 지 아닌지이므로 위에서 다룬 식에서 $$ u $$만 지우면 됩니다.

<br>

- $$ p(x_{t} \vert x_{0:t-1}, z_{1:t-1}) = p(x_{t} \vert x_{t-1}) $$

<br>

- $$ p(z_{t} \vert x_{0:t}, z_{1:t-1}) = p(z_{t} \vert x_{t}) $$

<br>

- $$ bel(x_{t}) = p(x_{t} \vert z_{1:t}) $$

<br>

- 최적 상태 추정을 위해 사용되는 `칼만 필터`에서 Contorl Matrix와 Control Input이 있습니다.
- `Localization` 문제를 풀 때에는 주어진 Control Input이 있기 때문에 이 값이 존재하지만, `Tracking` 문제에서는 이 값이 존재하지 않으므로 생략됩니다.
- 따라서 칼만 필터에서의 `Tracking`에서는 Control Input이 명시되지 않기 때문에 따로 입력되지는 않지만 그만큼 Control Input에 대한 `state variance`를 어떻게 선정하는 지가 상당히 중요해집니다.

<br>

[Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>