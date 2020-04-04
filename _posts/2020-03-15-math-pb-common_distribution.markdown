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



<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

