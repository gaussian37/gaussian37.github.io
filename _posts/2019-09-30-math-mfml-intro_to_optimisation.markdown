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

- 위 식에 대한 유도는 `newton-raphson method`의 마지막 부분에서 `테일러 급수`를 이용하여 유도해 보도록 하겠습니다.
- 임의의 $$ x_{0} $$에서 시작하여 위 점화식을 풀어갈 때, $$ x_{i} $$가 수렴하게 되면 그 해는 $$ f(x_{i}) = 0 $$ 을 만족하게됩니다.
- 위 식에서 업데이트 되는 $$ - f(x_{i}) / f'(x_{i}) $$ 를 살펴보면 분자의 $$ f(x_{i}) $$는 함수 값으로 $$ x_{i} $$가 실제 해에 가까워질 수록 $$ f(x_{i}) $$는 0에 수렴하게 됩니다. 따라서 분모인 $$ f'(x_{i}) $$ 값 크기를 조정해줍니다.
- 또한 $$ f'(x_{i}) $$도 업데이트 할 크기와 방향에 영향을 줍니다. $$ x_{i} $$ 지점에서 함수값이 증가하면 기울기가 양수이고 감소하면 기울기가 음수이기 때문에 업데이트 할 방향에 영향을 주고 기울기 값에 따라서 업데이트 할 크기에도 영향을 줍니다.
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
- 특히 `변곡점`에서는 $$ f'(x_{i}) $$ 가 0에 가까워 져서 $$ - f(x_{i}) / f'(x_{i}) $$가 아주 큰 값을 가지기 때문에 `변곡점`에서의 `newton-raphson method`는 매우 취약합니다.

<br>

- `newton-raphson method`는 위 경우와 같이 정확하게 방정식의 해를 못찾는 경우가 발생하긴 합니다. 그럼에도 초깃값 설정이 잘 되면 정확한 해를 근사할 수 있기 때문에 많이 사용되곤 합니다.

<br>

- 그러면 `newton-raphson method`가 어떤 방식으로 유도되었는 지 아래 예제를 통해 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/mfml/intro_to_optimisation/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- $$ f(x) = \frac{x^{6}}{6} - 3x^{4} + \frac{2x^{3}}{3} + \frac{27x^{2}}{2} + 18x -30 $$

<br>

- 위 식과 그래프를 통해 확인할 수 있는 것은 $$ x = -4 $$와 $$ x = 1 $$ 부근에서 해를 찾을 수 있다는 것입니다. 이 정보와 `newton-raphson method`를 이용하여 해를 구해보도록 하겠습니다.
- 먼저 앞에서 다룬 `newton-raphson method`의 식을 유도하기 전에 [테일러 급수](https://gaussian37.github.io/math-mfml-taylor_series_and_linearisation/)에서 $$ f(x + \Delta x) $$ 형태로 식 변형한 것에서 부터 시작해보도록 하겠습니다.
- 만약 $$ x_{0} $$라는 임의의 값에서 선형화를 한다면 $$ f(x + \Delta x) $$ 값을 다음과 같이 근사화 할 수 있습니다.

<br>

- $$ f(x_{0} + \Delta x) = f(x_{0}) + f'(x_{0}) \Delta x $$

<br>

- 만약 어떤 점에서 $$ f(x_{0} + \Delta x) $$가 0으로 수렴한다고 가정하겠습니다. 그러면 $$ \Delta x $$가 $$ x_{0} $$에서 얼만큼 떨어져 있는 값인지 알 수 있습니다.

<br>

- $$ f(x_{0} + \Delta x) = 0 = f(x_{0}) + f'(x_{0}) \Delta x $$

- $$ \Delta x = -\frac{f(x_{0})}{f'(x_{0})} $$

<br>

- 위에서 $$ f(x_{0} + \Delta x) = 0 $$ 이라는 가정을 통해 이 함수의 해는 $$ x_{0} + \Delta x $$가 됩니다. 따라서 $$ x_{0} + \Delta x  = x_{1} $$ 이라고 새로운 $$ x_{1} $$를 도입하여 정의할 수 있습니다.
- 여기서 $$ x $$의 인덱스를 일반화 하여 $$ i $$로 나타내면 다음과 같습니다.

<br>

- $$ x_{i+1} = x_{i} + \Delta x $$

<br>

- `테일러 급수`를 통하여 근사화 하는 대부분의 함수는 비선형 함수 입니다. 그래서 선형화로 한번에 정확한 값을 근사화 하는것은 어렵습니다. 따라서 임의의 점 $$ x_{0} $$에서 함수의 해가 되는 $$ x_{0} + \Delta x $$ 를 한번에 구하는 것은 어렵습니다. 하지만 임의의 점 $$ x_{0} $$ 보다는 $$ x_{0} + \Delta x $$가 좀 더 해에 가까운 값이 되는 것을 이용할 수 있습니다. 즉, 계속 반복하여 점점 실제 해에 가까워지도록 하는 것입니다.

<br>

- 위 예제를 이용하여 해를 구해보도록 하겠습니다.

<br>

- $$ f(x) = \frac{x^{6}}{6} - 3x^{4} + \frac{2x^{3}}{3} + \frac{27x^{2}}{2} + 18x -30 $$

- $$ f'(x) = x^{5} - 12x^{3} -2x^{2} + 27x + 18 $$

<br>

- 

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>