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
<center><img src="../assets/img/math/mfml/taylor_series_and_linearisation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프에서는 원래 그래프 $$ t(m) $$을 1.5kg 지점에서 1차 미분하여 접선을 구하였습니다.
- 이 접선은 미분을 한 1.5kg 근방에서는 꽤나 근사가 잘 되었지만 1.5kg에서 멀어질수록 그 오차가 커지는 것을 알 수 있습니다.
- 하지만 예제에서 다루는 입력은 닭의 질량이므로 1.5kg에서 크게 벗어나지 않기 때문에 의미가 있습니다.



<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>