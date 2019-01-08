---
layout: post
title: 선형 부분공간 
date: 2017-01-12 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra] # add tag
---

출처 : 칸 아카데미 선형대수학 (https://ko.khanacademy.org/math/linear-algebra/vectors-and-spaces)

이번 시간에는 `선형 부분공간(linear subspace)`에 대하여 알아보도록 하겠습니다.

![1](../assets/img/math/la/linear vector space/1.png)

<br>

+ subspace of $$ \mathbb{R}^{2} $$ 은 n차원의 실수 전체 공간을 뜻합니다.
+ $$ \mathbb{R}^{2} $$ 안에는 부분공간(subset) V가 있다고 가정합니다.
+ $$ \vec{x} $$ 가 V안에 속하는 벡터라면 $$ \vec{x} $$ 에 어떤 상수배를 하여 $$ c\vec{x} $$ 를 하더라도 V안에 속해야 `subspace` 라고 할 수 있습니다.
    + 즉, `subspace` 내에서는 스칼라 곱에 대한 연산이 닫혀 있습니다.
+ 또한 V 안의 어떤 벡터들을 더한다고 하더라도 V안에 속해야   `subspace` 라고 할 수 있습니다.
    + 즉, `subspace` 내에서는 덧셈 연산에 대하여 닫혀 있습니다.
    
<br>

![2](../assets/img/math/la/linear vector space/2.png)

<br>

+ 아주 간단한 예로 $$ V = \{ 0 \} $$ 는 덧셈과 스칼라 곱에 대한 연산이 닫혀 있습니다.

![3](../assets/img/math/la/linear vector space/3.png)

<br>

+ 위 예제에서는 subset에 조건이 걸려 있습니다. $$ x_{1} \ge 0 $$ 이란 조건이 추가 되었습니다.
+ 이 때에는 앞의 예제 처럼 모든 벡터에 대하여 덧셈과 스칼라곱이 닫혀 있지 않습니다.
+ 예를 들어 초록색 식처럼 -1을 곱하게 되면 subset을 벗어나게 됩니다. 이 때에는 닫혀 있지 않다고 합니다.

<br>

![4](../assets/img/math/la/linear vector space/4.png)

<br>

+ 어떤 집합의 선형생성을 알고 싶다면, `span` 형태로 정의할 수 있습니다.
    + 위의 예제처럼 $$ Span(\vec{1}, \vec{2}, \vec{3}) $$ `subspace`를 정의해 볼 수 있습니다.    

<br>

![5](../assets/img/math/la/linear vector space/5.png)

<br>

+ 스칼라 곱이 닫혀있는 subset에서는 스칼라 곱에 다른 스칼라 곱을 곱하더라도 치환하면 단순한 스칼라 곱이 됩니다.
    + 따라서 위의 주황색 글씨 처럼 $$ ac_{1} $$은 $$ c_{4} $$ 처럼 치환 됩니다.

<br>    

![6](../assets/img/math/la/linear vector space/6.png)

<br>

+ 자주색 식을 보면 $$ \vec{x} + \vec{y} $$ 식을 전개 하여 같은 항 끼리 묶으면 `선형결합`이 됩니다.
+ 즉, `덧셈`과 `스칼라 곱`에 대하여 닫혀있는 subset 상태에서의 연산은 또 다른 선형 결합을 만들어 냅니다. 

<br>

![7](../assets/img/math/la/linear vector space/7.png)

<br>

마지막 예제를 끝으로 `subspace`을 마무리 지어보겠습니다.

+ 만약 $$ V = Span(\[1, 1\]^{T}) 이라면 V의 `subspace` 는 직선에 머물게 되곘습니다.
+ 이 때, 비록 `subspace`는 직선에 불과하지만 그것 또한 공간을 만들어 낼 수 있습니다.
    + 스칼라 곱을 할 때 0을 곱하면 `subspace`에 속하게 됩니다.