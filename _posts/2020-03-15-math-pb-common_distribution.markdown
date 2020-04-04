---
layout: post
title: 기본 확률 분포들 (Common probability distribution)
date: 2020-03-15 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [기본 확률 분포, uniform, Bernoulli, Binomial, Poisson, Geometric, Negative Binomial, Multinomial, Beta, Exponential, Double Exponential, Gamma, Inverse-Gamma, Normal, t, Dirichlet] # add tag
---

<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

## **목차**

<br>

- ### Discrete distributions
    - ### Uniform
    - ### Bernoulli
    - ### Binomial
    - ### Poisson
    - ### Geometric
    - ### Negative Binomial
    - ### Multinomial
- ### Continuous distributions
    - ### Uniform
    - ### Beta
    - ### Exponential
    - ### Double Exponential
    - ### Gamma
    - ### Inverse-Gamma
    - ### Normal
    - ### t
    - ### Dirichlet

<br>

- 이 글에서는 기본적으로 많이 사용하는 다양한 분포들에 대해서 간략하게 다루어 보도록 하겠습니다.
- 먼저 Discrete distribution에 대하여 다루어 보도록 하고 그 다음 Continuous distribution에 대하여 다루어 보도록 하겠습니다.

<br>

## **Discrete : Uniform**

<br>

- Uniform distribution은 가장 간단한 discrete probablity distribution 입니다. 
- 일반적으로 숫자 1, 2 등으로 표현되는 N 개의 서로 다른 결과에 동일한 확률을 할당합니다.

<br>

$$ X \ \sim \ \text{Uniform}(N) $$

$$ P(X = x \vert N) =  \frac{1}{N} \ for x = 1, 2, ... , N $$

$$ E[X] = \frac{N+1}{2} $$

$$ \text{Var}[X] = \frac{N^{2} - 1}{12} $$

<br>

- Uniform distrtibution의 대표적인 예가 주사위 이며 이 때, $$ N = 6 $$이 됩니다.

<br>

## **Discrete : Bernoulli**

<br>

- Bernoulli distribution은 0과 1의 이진 결과에 사용됩니다.이 매개 변수에는 "성공"또는 1의 `확률` 인 `p`가 있습니다.

<br>

$$ X \ \sim \ Bern(p) $$

$$ P(X = x \vert p) = p^{x}(1 - p)^{1-x} \ \ \ \ for x = 0, 1 $$

$$ E[X] = p $$

$$ \text{Var}[X] = p(1-p) $$

<br>

- Bernoulli distribution의 대표적인 예는 동전 던지기 ($$ p =  0.5 $$) 입니다.

<br>

## **Discrete : Binomial**

<br>

- Binomial distribution(이항 분포)는 n번의 독립적인 Bernoulli 시행 (각각 동일한 성공 확률을 가집니다.)에서 "성공"의 `수`를 계산합니다.
- 따라서 $$ X_{1}, ... , X_{n} $$이 Bernoulli 분포를 가지는 확률 변수일 때, $$ Y = \sum_{i = 1}^{n} X_{i} $$가 됩니다.

<br>

$$ Y \ \sim \ Binom(n, p) $$

$$ P(Y = y \vert n, p) = \begin{pmatirx} n \\ y \end{pmatrix} p^{y}(1 - p)^{n - y} \ \ \ \ for y = 0, 1, ... , n $$

$$ E[Y] = np $$

$$ \text{Var}[Y] = np(1-p) $$

<br>

## **Discrete : Poisson**

<br>

- Poisson distribution은 다양한 상황에서의 어떤 event 발생에 대한 counting을 하기 위한 목적으로 사용됩니다.
- 파라미터 $$ \lambda > 0 $$ 는 우리가 어떤 event의 발생을 관찰할 때, 그 event 발생의 빈도를 나타냅니다.

<br>

$$ X \ \sim \ Pois(\lambda) $$

$$ P(X = x \vert \lambda) = \frac{\lambda^{x}exp(-\lambda)}{x!} $$

$$ E[X] = \lambda $$

$$ \text{Var}[X] = \lambda $$

<br>

- `Poisson process`는 평균적으로 **특정 시간 동안 어떤 event가 $$ \lambda $$ 만큼 발생**하고, 각각의 event가 발생하는 것은 서로 독립적이라는 전제하에 정의됩니다.
- `Poisson`과 관련된 내용은 이해하기 어려울 수 있으니 예를 한가지 들어보겠습니다.
- 어떤 지역에 `1주일`에 `평균 2번`의 지진이 발생한다고 가정해 보겠습니다. 다음 2주 동안 최소 3번 이상의 지진이 발생할 확률은 어떻게 될까요?
- 이 문제를 Poisson process로 접근하기 위해서 현재 가지고 있는 데이터인 평균 2회/1주일 → 평균 4회/2주일로 확장시켜 보겠습니다.
- 그러면 $$ X \ \sim \ Pois(4) $$를 따르게 됩니다. 즉, $$ \lambda = 4 $$가 됩니다.

<br>

$$ P(X \ge 3) = 1 - P(X \le 2) = 1 - P(X = 0) - P(X = 1) - P(X = 2) = 1 - e^{-4} - 4e^{-4} - \frac{4^{2}e^{-4}}{2} =  1 - 13e^{-4} = 0.762 $$

<br>


<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

