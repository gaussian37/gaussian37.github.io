---
layout: post
title: 가우시안 프로세스 (Gaussian Process) 내용 정리
date: 2021-07-17 00:00:00
img: math/pb/gaussian_process/0.png
categories: [math-pb]
tags: [가우시안 프로세스, gaussian process, gaussian] # add tag
---

<br>

[통계학 관련 글 목차](https://gaussian37.github.io/math-pb-table/)

<br>

- 참조 : https://www.edwith.org/bayesiandeeplearning/
- 참조 : https://students.brown.edu/seeing-theory/

<br>

- 이 글은 edwith의 베이지안 딥러닝 강의를 요약한 내용입니다. 
- 이 글의 학습 목적은 [베이지안 딥러닝](https://gaussian37.github.io/dl-concept-bayesian_neural_network/)과 [딥러닝에서의 Uncertainty](https://gaussian37.github.io/dl-concept-uncertainty_in_deep_learning/)를 이해하기 위하여 배경 지식을 쌓기 위함입니다.

<br>

## **목차**

<br>

- ### 배경지식 설명
- ### Gaussian Process
- ### Weight Space View
- ### Function Space View
- ### Gaussian process latent variable model (GPLVM)
- ### Gaussian process Application

<br>

## **배경지식 설명**

<br>

- 아래는 이 글에서 설명할 `Gaussian Process`를 이해하기 위한 배경지식으로 스킵하셔도 됩니다.
- 먼저 gaussian process에 대한 개념을 알아보기 이전에 `random variable`, `random process`등과 관련한 내용에 대하여 알아보도록 하겠습니다.

<br>

#### **random variable**

<br>

- 먼저 `random variable`에 대하여 알아보겠습니다. 많은 분들이 기존 통계학을 공부하실 때 이미 학습하신 내용입니다.

<br>
<center><img src="../assets/img/math/pb/gaussian_process/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 먼저 `Random`의 의미를 살펴보면 `sample space`에서 한 개의 원소를 뽑는 것을 의미합니다.
- 이 때, `random variable`은 임의의 프로그래밍 언어에서 랜덤값을 추출할 때, 그 랜덤 값이 어떤 방식으로 추출되는 지 모델링 되는 것을 `random variable`이라고 말할 수 있습니다.
- 위 그림에 빗대어 보면 sample space로 부터 실제 관측한 값을 뽑아 내는 `함수`가 random variable 입니다. 확률과 비교해 보면 확률은 sample space에서 영역(면적)을 만들어 낼 수 있는 `subset`인 반면에 random variable은 **sample space에서 어떻게 원소를 추출**할 지 또는 **어떤 subset으로 원소를 배정할 지** 정의되어 있는 `함수`라고 이해하면 됩니다.
- 추가적으로 `Random Experiment`라는 용어는 `random variable`을 이용하여 어떤 값을 추출하였을 때, 그 값을 실제 확인한 결과를 뜻합니다.
- 이와 같이 **가상의 sample space에서 random variable을 통하여 원소를 뽑고 확인하는 이 전체 과정**을 `realization` 또는 `sampling` 이라고 합니다.

<br>

- $$ \sum P(X = x_{i}) = 1 (x_{i} : i = 1, 2, ...) $$

<br>

- 위 식과 같은 discrete random variable을 해석하면 $$ X $$ 라는 random variable이 있는데 그 결과 값이 $$ x_{i} $$가 나올 면적의 크기라고 말할 수 있습니다.

<br>

- $$ E[X] = \begin{cases} \sum_{x} x p(x) \ \ \ \ \text{discrete} X \\ \int_{-\infty}^{\infty} x f(x) dx \ \ \ \ \text{continuous} X \end{cases} $$

<br>

- random variable $$ X $$ 를 이용하여 위 식과 같이 정의하였을 때, 잘 알려진 바와 같이 `expectation` 이라고 합니다.
- 위 식의 직관적인 의미는 random variable $$ X $$ 를 여러번 sampling 하였을 때, 평균적인 기댓값 (expectation)을 뜻합니다. random variable의 의미와 연관지어 생각하면 의미를 다시 느껴볼 수 있습니다.

<br>

- 또 다른 식으로 $$ E(X \vert Y) $$ 즉, `conditional expectaion`에 대하여 간단히 알아보겠습니다.
- 이 식에서 전제 조건은 random variable $$ X $$ 의 expectation인 $$ E(X) $$ 가 $$ E(X) = \int x f(x) dx $$ 로 정의되고 이 값이 특정 분포(ex. 가우시안 분포)로 정해져 있다고 하면 $$ X $$는 deterministic variable 이 된다라는 것입니다.
- 이 때, $$ Y $$ 가 random variable이라고 하면 $$ E(X \vert Y ) $$ 또한 random variable이 된다는 점입니다.

<br>
<center><img src="../assets/img/math/pb/gaussian_process/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그래프에서 $$ X $$는 위 그림과 같은 고정된 분포를 가지므로 deterministic variable이 됩니다.
- 이 때, random variable $$ Y $$ 는 4개의 값 $$ y_{1}, y_{2}, y_{3}, y_{4} $$ 로 추출될 수 있습니다. 즉, 확률 관점으로 보았을 때, sample space에 4개의 subset에 있고 4개의 subset의 확률 값을 모두 합하면 1이 됩니다.
- 이 각각의 $$ Y $$ 의 subset 내부에서 $$ X $$ 가 가지는 exectation을 계산하는 것이 `conditional expectation`이 됩니다. $$ X $$ 는 continuous 하여 많은 값을 가지게 되지만 $$ Y $$ 라는 random variable이 sample space를 굉장히 크게 나누어서 subset을 가졌기 때문에 $$ X $$ 도 영향을 받게 됩니다. (즉, conditional 하게 됩니다.)

<br>

#### **moment**

<br>

- 어떤 확률 분포를 나타낼 때 대표적인 값으로 기댓값 즉 평균값을 많이 사용합니다. 물론 평균값에 의해 왜곡되는 것이 많다는 것을 알고 있음에도 불구하고 대표적으로 사용하고 있습니다.

<br>
<center><img src="../assets/img/math/pb/gaussian_process/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 평균이라는 값은 `n-th moment` 중 1차에 해당하며 `2-th moment`는 흔히 아는 `variance` 이며 3차는 `skewness`, 4차는 `kurtosis` 라고 합니다.
- 어떤 두 분포가 비슷한 지 확인 할 때, 쉽게 비교할 수 있는 방법으로 두 분포의 `n-th moment`가 같은 지 확인하는 방법을 사용하곤 합니다. 이와 같은 방법을 이용하면 단순히 평균값을 통해 왜곡되는 문제를 보정하여 비교할 수 있습니다.

<br>

[통계학 관련 글 목차](https://gaussian37.github.io/math-pb-table/)

<br>
