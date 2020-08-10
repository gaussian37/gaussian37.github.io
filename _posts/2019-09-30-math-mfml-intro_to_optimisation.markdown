---
layout: post
title: Intro to Optimisation
date: 2019-09-30 01:00:00
img: math/mfml/mfml.jpg
categories: [math-mfml] 
tags: [calculus, multivariate chain rule, application] # add tag
---

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>

- 이번 글에서는 `최적화(Optimisation)` 하는 방법에 대하여 다루어 보려고 합니다. 이 글에서 다루는 최적화 방법은 최적화의 기본이 되는 간단한 방법들입니다.

<br>

## **목차**

<br>

- ### newton-raphson method
- ### gradient descent
- ### constrained optimisation methodof lagrange multipliers

<br>

## **newton-raphson method**

<br>

- 지금 부터 살펴볼 `newton-raphson method`는 derivative를 이용하여 방정식을 풀어 보는 방법입니다.

<br>
<center><img src="../assets/img/math/mfml/intro_to_optimisation/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `newton-raphson method`는 위 그림과 같이 반복적인 탐색을 통하여 방정식의 해를 찾는 방법입니다. 이 때 사용하는 점화식은 다음과 같습니다.

<br>

- $$ x_{i+1} = x_{i} - \frac{f(x_{i})}{f'(x_{i})} $$

<br>

- 임의의 $$ x_{0} $$에서 시작하여 위 점화식을 풀어갈 때, $$ x_{i} $$가 수렴하게 되면 그 해는 $$ f(x_{i}) = 0 $$ 을 만족하게됩니다.
- 위 식에서 업데이트 되는 $$ - f(x_{i}) / f'(x_{i}) $$ 를 살펴보면 분자의 $$ f(x_{i}) $$는 함수 값으로 $$ x_{i} $$가 실제 해에 가까워질 수록 $$ f(x_{i}) $$는 0에 수렴하게 됩니다. 따라서 분모인 $$ f'(x_{i}) $$ 값 크기를 조정해줍니다.
- 반면 $$ f'(x_{i}) $$는 업데이트 할 방향과 관계있습니다. $$ x_{i} $$ 지점에서 함수값이 증가하면 기울기가 양수이고 감소하면 기울기가 음수이기 때문입니다.
- 위 예제에서는 -2를 $$ x $$의 초깃값으로 시작해서 -1.769 근처에서 수렴시킵니다.
- 위 과정을 볼 때 `newton-raphson method`의 핵심은 초깃값을 어디서 부터 시작하는 지에 따라서 수렴 성능에 영향을 미칩니다.

<br>
<center><img src="../assets/img/math/mfml/intro_to_optimisation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 예제에서는 $$ x_{0} = 0 $$으로 설정하였습니다. 이 경우에는 `newton-raphson method`가 수렴하지 않고 `진동`하게 됩니다. 업데이트 되는 부분이 해를 찾아갈 정도로 $$ x_{i} $$의 값을 업데이트 해주지 못하기 때문입니다.

<br>
<center><img src="../assets/img/math/mfml/intro_to_optimisation/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 문제와 유사하게 다른 곳에서 수렴하는 문제가 발생하는 경우가 생깁니다. 이 또한 업데이트가 되는 $$ - f(x_{i}) / f'(x_{i}) $$ 이 부분에서 해에 해당하는 $$ x $$ 값으로 적당한 크기와 방향만큼 업데이트 되지 못하기 때문입니다. 

<br>

- `newton-raphson method`는 위 경우와 같이 정확하게 방정식의 해를 못찾는 경우가 발생하긴 합니다. 하지만 많은 경우에 정확한 해를 근사화 할 수 있기 때문에 사용되고 있습니다.


<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>