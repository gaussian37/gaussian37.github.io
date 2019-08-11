---
layout: post
title: 베이즈 방법
date: 2019-08-11 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [수리통계학, 베이즈 방법, 베이지안] # add tag
---

- 이번 글에서는 베이즈 방법(베이지안 방법 : Bayesian Methods Of Estimation)에 대하여 알아보도록 하겠습니다.

- 전통적인 방법 : 확률표본의 정보만 이용
- 예를 들어 정규 분포의 경우, 표본 평균 $$ \bar{X} $$에 대해, 구간 $$ \Bigl( \bar{X} - 1.96\frac{\sigma}{\sqrt{n}}, \bar{X} + 1.96\frac{\sigma}{\sqrt{n}} \Bigr) $$에 실제 평균값 $$ \mu $$가 들어있다고 95% 확신

<br>

- 베이즈 방법 : **모수를 확률 변수**로 다룸
- 　$$ \theta $$ : 모수(값)
- 　$$ \Theta $$ : 모수(확률변수)
- 　$$ \pi (\theta) $$ (사전분포 : **prior distribution**) : $$ \Theta $$의 확률분포로 $$ \theta $$의 값이 어느 정도 되는지를 알고 있는 상황

<br>

- 크기 n인 확률 표본을 $$ x = (x_{1}, x_{2}, \cdots , x_{n}) $$과 같이 나타내고, 모수 $$ \theta $$에 대해 표본의 표본분포를 $$ f(x \vert \theta) $$로 나타냅니다.

<br>

- 베이즈 정리 : $$ P(A \vert B) = \frac{ P(B \vert A)P(A)) }{P(B)} $$

<br>

- `정의` : 자료 $$ x $$가 주어질 경우의 $$ \theta $$의 분포(사후 분포 : **posterior distribution**)은 $$ \pi(\theta \vert x) = \frac{f(x \vert \theta)\pi(\theta)}{g(x)} $$로 주어집니다.
    - 여기서 $$ g(x) $$는 $$ x $$의 주변분포 입니다.
- 주변분포 $$ g(x) $$ 는
    - 이산형 : $$ \sum_{\theta} f(x \vert \theta)\pi(\theta) $$
    - 연속형 : $$ \int_{-\infty}^{\infty} f(x \vert \theta) \pi(\theta) d \theta $$

<br>

- 예제를 한번 살펴보겠습니다.
- 어느 생산 기계의 불량률의 사전 분포가 다음과 같습니다.

|       P      	| 0.1 	| 0.2 	|
|:------------:	|:---:	|:---:	|
| $$ \pi(p) $$ 	| 0.6 	| 0.4 	|

<br>

- 이 때, $$ x $$를 크기 2인 확률 표본 중 불량품의 수라고 가정하겠습니다. $$ x $$가 관측된 후 $$ p $$의 사후 분포를 구해 보겠습니다.
- 확률 분포 $$ x $$는 이항 분포를 따릅니다. 

