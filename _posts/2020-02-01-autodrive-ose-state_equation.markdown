---
layout: post
title: 상태 방정식 (state equation)
date: 2020-02-01 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [Optimal State Estimation, 최정 상태 이론, 상태 방정식] # add tag
---

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
- 　$$ \dot{x(k)} $$는 k번째의 step의 물체의 속도에 해당합니다.
- 　$$ \ddot{x(k)} $$는 k번째의 step의 물체의 가속도에 해당합니다.
- 　$$ T $$는 각 step 간 시간 간격을 뜻합니다.

<br>
<center><img src="../assets/img/autodrive/ose/state_equation/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

