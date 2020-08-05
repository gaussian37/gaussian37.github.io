---
layout: post
title: Taylor series 와 linearisation
date: 2019-09-30 00:00:00
img: math/mfml/mfml.jpg
categories: [math-mfml] 
tags: [calculus, multivariate chain rule, application] # add tag
---

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>

- 이번 글에서 다룰 `테일러 급수`는 함수를 `다항식 급수`로 다시 표현하는 방법입니다. 
- 테일러 급수는 간단한 선형 근사법을 복잡한 함수에 사용합니다. 이 글에서는 먼저 단일변수 테일러 급수의 공식 표현을 도출하고 기계 학습과 관련된 이 결과의 몇 가지 중요한 결과에 대해 논의해 보겠습니다. 더 나아가 다변수 사례로 확장한 후 `Jacobian`과 `Hessian`이 어떻게 적용되는 지 살펴보겠습니다. 마지막에 다루는 다변수 테일러 급수에서는 앞에서 다룬 모든 내용을 총 집합 해서 설명해 보도록 하겠습니다.

<br>

## **목차**
- ### Taylor series for approximations
    - #### Building approximate functions
    - #### Power series
    - #### Power series derivation
- ### Multivariable Taylor Series    

<br>

## **Taylor series for approximations**

<br>

- 이번 글에서는 복잡한 함수를 좀 더 간단한 함수를 이용하여 어떻게 근사화 할 수 있는 지 배워보려고 합니다.
- 근사화 하는 방법으로는 테일러 급수(`taylor series`)를 사용하려고 합니다.

<br>

#### **Building approximate functions**

<br>

- 테일러 급수는 어떤 함수를 근사화 하기 위해 사용하는 방법 중 하나입니다.
- 테일러 급수에 대하여 구체적으로 알아보기 이전에 그래프 상에서 근사화 하는 것이 어떤 의미를 가지는 지 한번 살펴보도록 하겠습니다.
- 예를 들어 닭을 오븐에 구울 때, 얼마나 오래 구울 지 시간을 구하는 함수가 있다고 가정해 보겠습니다.
- 이런 함수는 단순히 선형적이지 않을 뿐 아니라 많은 요소가 함수에 추가됭야 정확하게 추정할 수 있습니다.
- 예를 들어 다음과 같이 함수를 만들어 보겠습니다.

<br>

$$ \text{t(m, OvenFactor, ChickenShapeFactor)} = 7.33m^{5} - 72.3m^{4} + 253m^{3} - 368m^{2} + 250m + 0.02 + \text{OvenFactor} + \text{ChickenShapeFactor} $$

<br>

- 위 식은 닭의 질량 $$ m $$과 Oven과 Chicken의 특성에 따른 OvenFactor와 ChickenShapeFactor를 입력으로 받습니다.
- 만약 위 식이 닭을 얼만큼 구워야 할 지 잘 추정할 수 있는 식이라도 이 식은 사람들이 사용하기에 너무 복잡합니다.
- 따라서 추정 성능이 조금 떨어지더라도 식을 단순화 해서 사람들이 사용하기 쉽게 만들어 보려고 합니다.
- 먼저 사람들이 Oven과 Chicken은 비슷한 것을 사용한다고 가정하고 OvenFactor와 ChickenShapeFactor는 소거하겠습니다. 실제로 식을 단순화 시킬 때 큰 차이가 없을 것으로 생각하는 변수들은 동일하다고 가정하고 소거하는 방법을 사용하기도 합니다.
- 따라서 다음 식과 같이 두 Factor를 소거합니다.

<br>

$$ \text{t(m)} 7.33m^{5} - 72.3m^{4} + 253m^{3} - 368m^{2} + 250m + 0.02 $$

<br>

- 위 식은 닭의 질량이 입력되었을 때, 오븐에 구울 최적의 시간을 산출하는 함수입니다.

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞의 식을 좌표 평면에 그리와 위 그래프와 같이 그릴 수 있습니다.
- 만약 슈퍼마켓에 파는 일반적인 닭의 무게가 1.5kg이라고 가정해 보겠습니다.

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 1.5kg의 무게에 해당하는 시간은 그래프에 표시된 점의 y값에 해당합니다.

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프에서는 원래 그래프 $$ t(m) $$을 1.5kg 지점에서 1차 미분하여 접선 $$ t'(m) $$을 구하였습니다.
- 이 접선은 미분을 한 1.5kg 근방에서는 꽤나 근사가 잘 되었지만 1.5kg에서 멀어질수록 그 오차가 커지는 것을 알 수 있습니다.
- 하지만 예제에서 다루는 입력은 닭의 질량이므로 1.5kg에서 크게 벗어나지 않기 때문에 의미가 있습니다.

<br>

#### **Power series**

<br>

- 테일러 급수에 대하여 본격적으로 알아보기 이전에 간단한 컨셉에 대하여 먼저 다루어 보겠습니다.
- 먼저 테일러 급수는 멱급수(`Power series`)의 형태입니다. 테일러 급수의 형태를 보면 $$ x^{n} $$ 이고 $$ n $$은 점점 증가합니다. 그리고 $$ x^{n} $$ 앞의 계수가 붙어있습니다. 예를 들어 다음과 같습니다.

<br>

$$ g(x) = a + bx + cx^{2} + dx^{3} + \cdots $$

<br>

- 앞으로 살펴 볼 테일러 급수는 위 식과 같은 멱급수의 형태를 가지며 복잡한 식을 근사화 하는 목적으로 사용됩니다.
- 급수에 사용되는 $$ n $$이 단순히 1차식을 사용할 수도 있는 반면 큰 차수도 사용할 수 있습니다. 큰 차수를 사용할수록 더 정교하게 근사화 가능합니다.
- 물론 많은 application 들은 매우 큰 차수 보다는 앞의 몇 차수 (1차, 2차, 3차 ... 등)만을 사용하는 데, 낮은 차수 들의 합을 이용해서도 꽤나 근사화가 잘되기 때문입니다.
- 아래 예제에서는 ① 지수함수와 ② 6차 다항 함수에 대하여 0, 1, 2, 3 차로 각각 근사화 하는 예제를 보여줍니다.

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/4.gif" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/5.gif" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 두 그림을 보면 0차 즉, x축에 평행한 직선 부터 3차 곡선 까지 지수함수와 다항함수의 각 x점에 접하는 곡선으로 근사화 하였습니다.
- 근사화 하는 함수의 **차수가 증가할수록 더 정교하게 근사화** 할 수 있음을 확인할 수 있습니다. 물론 차수가 늘어날수록 계산량도 늘어납니다.

<br>

- 정리해 보겠습니다. 앞에서 다룬 바와 같이 Taylor 계열 근사법은 power series로 볼 수도 있습니다. 이 근사법은 특히 수치적 방법을 사용할 때 종종 더 간단하고 평가하기 쉬운 함수를 만드는 데 사용됩니다. 다음 예제들을 보면 점점 증가하는 power serires를 통해 함수에 대한 추가 정보를 어떻게 얻어 나아갈 수 있는 지 이해할 수 있습니다.
- 아래 그림에서 0차, 2차, 4차 3개의 근사식은 $$ x = 0 $$ 일 때 어떤 함수를 근사한 식이라고 하겠습니다. 이 경우 원래 함수는 $$ \cos{(x)} $$ 일 수 있습니다. (물론 가능한 다른 함수도 많습니다.)

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 보면 $$ f(x) = \cos{(x)} $$는 근사식의 일부를 모두 포함합니다. 즉 접하는 것을 알 수 있습니다.

<br>
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 위 그림에서 1차 함수가 지수 함수를 근사한 것이라고 하였을 때, 이 근사는 잘못되었다고 판단할 수 있습니다. 왜냐하면 지수 함수와 접하지 않기 때문입니다.

<br>

#### **Power series derivation**

<br>



<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>