---
layout: post
title: Probability Model (확률 모형) 개념 학습
date: 2020-08-08 00:00:00
img: ml/concept/probability_model/0.png
categories: [ml-concept] 
tags: [machine learning, probability model, 확률 모형, MLE, Maximum Likelihood Estimation] # add tag
---

<br>

[머신 러닝 관련 글 목록](https://gaussian37.github.io/ml-concept-table/)

<br>

- 이번 글에서는 `Probability Model`이란 개념을 `Gaussian Distribution`을 통해 이해해 보도록 하겠습니다.

<br>

## **확률 모형 (Probability Model) 이란?**

<br>

- `확률 모형`이란 **수집 및 관측된 데이터의 발생 확률 (또는 분포)을 잘 근사하는 모형**으로 일반적으로 $$ p(x \vert \theta) $$ 로 표기합니다. `확률 모형 (Probability Model)`, `통계 모형 (Statistical Model)`, `확률 분포 (Probability Distribution)` 모두 같은 뜻으로 사용됩니다.
- 이때, $$ \theta $$ 는 확률 모형을 정의하는 데 중요한 역할을 하는 값으로 모수 `parameter` 또는 요약 통계량 (Descriptive measure)라고 부릅니다.
- 확률 모형은 상황에 따라 $$ p(x; \theta), p_{\theta}(x) $$ 와 같이 쓰이며 경우에 따라서 $$ p(x) $$ 와 같이 parameter $$ \theta $$ 를 생략하고 표기하기도 합니다.

<br>
<center><img src="../assets/img/ml/concept/probability_model/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 가우시안 분포와 관련된 상세 설명은 아래 링크를 참조하시기 바랍니다.
    - 링크 : [가우시안 분포 공식 유도](https://gaussian37.github.io/math-pb-about_gaussian/#%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88-%EB%B6%84%ED%8F%AC-%EA%B3%B5%EC%8B%9D-%EC%9C%A0%EB%8F%84-1)
    - 링크 : [정규 분포](https://gaussian37.github.io/math-pb-normal_distribution)

<br>

- `가우시안 분포`를 확률 모형을 설명하는 데 사용하는 이유는 기초적인 확률 모형 (또는 확률 분포)로 관찰된 전체 데이터 집합이 평균을 중심으로 하여 뭉쳐져 있는 형태를 표현하는 데 가장 적합하기 때문입니다.

<br>

## **모수 추정의 의미**

<br>  

- 데이터들이 어떤 확률 분포 $$ p(x \vert \theta^{*}) $$ 에 따라 샘플링 되어 구해졌다고 생각해 보도록 하겠습니다.

<br>

- $$ X = {x_{1}, x_{2}, \cdots , x_{n}}, \ \ x_{i} \sim p(x \vert \theta^{*}) $$

<br>

- 위 식에서 $$ p(x \vert \theta^{*}) $$ 이 의미하는 바는 `이상적인 실제 확률 분포`를 뜻합니다. 이러한 이상적인 확률 분포에서 $$ X $$ 를 수집한다고 이해하면 됩니다.
- 일반적으로 모수 $$ \theta $$ 추정의 목적은 관측된 데이터의 실제 확률 분포 $$ p(X \vert \theta^{*}) $$  를 **최대한 잘 근사 하는 수학적 모형을 찾는 것**입니다. 이와 같이 근사화한 모델을 사용하는 이유는 실제 데이터 확률 분포 또는 실제 파라미터 $$ \theta^{*} $$ 를 정확히 할 수 없기 때문입니다.
- 따라서 임의의 확률 모형 $$ p(x \vert \cdot) $$ 을 가정한 뒤, 적어도 그 모형이 데이터를 가장 잘 설명하는 파라미터 $$ \theta (\approx \theta^{*}) $$ 를 찾는 과정을 `모수 추정`이라고 합니다. 예를 들어 $$ p(x \vert \theta) = N(x \vert \mu, \sigma^{2}), \theta = [\mu, \sigma] $$ 와 같은 식에서 구체적인 값 $$ \theta $$ 를 찾는 과정이라고 말할 수 있습니다.

<br>

## **MLE(Maximum Likelihood Estimation)**

<br>



<br>

[머신 러닝 관련 글 목록](https://gaussian37.github.io/ml-concept-table/)

<br>