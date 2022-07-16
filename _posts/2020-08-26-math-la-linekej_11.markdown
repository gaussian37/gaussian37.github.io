---
layout: post
title: 기저와 차원
date: 2020-08-26 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Linear algebra, 기저, 차원, basis, dimension] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 `기저(basis)`와 `차원(dimension)`에 대하여 알아보도록 하겠습니다.
- 제 블로그의 선형대수학 이전 글들을 살펴보면 `생성` 또는 `Span` 이라는 용어로 설명한 내용이 있습니다. `Span`은 $$ v_{1}, v_{2}, \cdots , v_{k} $$ 의 $$ k $$ 개의 일차 결합을 모두 모은 것을 `Span` 이라고 하며 `Span`은 주어진 벡터 공간 $$ V $$ 의 부분공간이 됩니다.
- 이 때, $$ v_{1}, v_{2}, \cdots , v_{k} $$ 의 $$ k $$ 중에서 몇개를 제외하더라도 똑같은 부분공간을 만들 수 있는 경우들이 발생합니다. 이 경우는 굳이 $$ k $$ 개 벡터를 다 사용하지 않고 필요한 벡터만 사용하여 동일한 부분공간을 만들 수 있는데 이와 관련된 내용이 아래 `정리 10` 입니다.

<br>

- **(정리 10) 벡터 공간 $$ V $$ 의 원소 $$ v_{1}, v_{2}, \cdots , v_{k} $$ 에 대하여 $$ v_{j} = a_{1}v_{1} + a_{2}v_{2} + \cdots + a_{j-1}v_{j-1} + a_{j+1}v_{j+1} + \cdots + a_{k}v_{k}, \quad 1 \le j \le k $$ 을 만족하면 $$ \text{Span}(v_{1}, v_{2}, \cdots , v_{k}) = \text{Span}(v_{1}, v_{2}, \cdots , v_{j-1}, v_{k+1}, \cdots , v_{k}) $$ 이 된다.**

<br>

- 위 정리 10은  $$ v_{j} $$ 나머지 원소들의 일차 결합으로 만들 수 있는 일차 종속인 벡터이며 이러한 일차 종속인 벡터를 제외하더라고 `Span`을 만족한다는 뜻입니다. 정리 10에 대한 내용을 증명해 보면 아래와 같습니다.

<br>

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>
