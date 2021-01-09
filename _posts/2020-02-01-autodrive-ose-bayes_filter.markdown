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

- 이번 글에서는 Bayes Filter에 대하여 다루어 보도록 하겠습니다. Bayes Filter는 Kalman Filter와 Particle Filter의 개념의 기초가 되는 간단하지만 매우 중요한 Filter입니다. 따라서 Kalman Filter나 Particle Filter를 배우기 전 단계라면 베이즈 필터를 통하여 어떻게 동작하는 지 그 메커니즘을 먼저 배우기를 추천합니다.
- 만약 Kalman Filter와 Particle Filter에 대하여 잘 아신다면 아래 Bayes Filter의 개념과 비교해 보는 것을 추천 드립니다.

<br>

## **목차**

<br>



<br>

- 먼저 이 글에서 다룰 `Bayes Filter`는 `Bayes Theorem`을 `Recursive`하게 따릅니다. 
- 즉, **확률 기반의 Recursive한 Filter** 중 하나라고 보시면 됩니다. Bayes Filter의 식은 다음과 같습니다.

<br>
<center><img src="../assets/img/autodrive/ose/bayes_filter/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이러한 Bayes Filter의 목적을 단 한줄로 표현하면 다음과 같습니다.

<br>

- $$ bel(x_{t}) = p(x_{t} \vert z_{1:t}, u_{1:t}) $$

<br>

- 앞의 사전 지식을 잘 이해하고 오셨으면 위 식의 뜻은 충분히 이해가 가실 것으로 생각됩니다.
- 위 식을 다시 한번 해석하면 $$ x{t} $$는 t시점의 상태 값을 $$ z{t} $$는 센서 입력 값을 뜻하고 $$ u_{t} $$는 제어 입력값을 뜻합니다.
- 따라서 **알고리즘의 시작부터 현재 시점 $$ t $$ 까지의 센서 입력값과 제어 입력값을 이용하여 현재 상태를 확률적으로 추정하는 것**으로 해석할 수 있습니다.