---
layout: post
title: 외적과 값의 사인값 사이의 관계
date: 2017-02-01 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, 벡터, 외적] # add tag
---

<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.3%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>

<br><br>

출처 : 칸 아카데미 선형대수학 (https://ko.khanacademy.org/math/linear-algebra/vectors-and-spaces)

이번 글에서는 `외적(cross product)`와 sin 값의 관계에 대하여 알아보도록 하겠습니다.

+ 결과적으로 $$ \Vert \vec{a} \times \vec{b} \Vert = \Vert \vec{a} \Vert \ \Vert \vec{b} \Vert sin \theta $$ 을 만족합니다. 

<br>

<img src="../assets/img/math/la/cross product and sin of angle/1.png" alt="Drawing" style="width: 600px;"/>

+ 식을 유도하기 위해 먼저 위의 슬라이드 처럼 두 벡터를 정의합니다. 외적을 다루기 때문에 $$ \mathbb R^{3} $$ 의 벡터 $$ \vec{a}, \vec{b} $$를 정의 합니다.
    + 즉, $$ \vec{a} = \[a_{1}, a_{2}, a_{3} \]^{T}, \vec{b} = \[b_{1}, b_{2}, b_{3} \]^{T} $$ 입니다. 
+ 다음으로 $$ \Vert \vec{a} \times \vec{b} \Vert^{2} $$ 를 각각의 원소를 이용하여 `전개` 합니다.
    + 위 슬라이드에서 $$ \Vert \vec{a} \times \vec{b} \Vert^{2} $$ 의 전개 결과를 볼 수 있습니다.

<br>

<img src="../assets/img/math/la/cross product and sin of angle/2.png" alt="Drawing" style="width: 600px;"/>

+ 다음으로 벡터의 내적과 cosine 간의 관계를 이용해야 합니다.
+ 위 슬라이드 처럼 $$ (\Vert \vec{a} \Vert \ \Vert \vec{b} \Vert cos\theta)^{2} $$ 을 각 원소를 이용하여 전개 합니다.
+ 앞의 슬라이드의 `외적을 제곱한 결과`와 현재 슬라이드의 `내적을 제곱한 결과`를 더해보겠습니다.
    + 앞의 슬라이드의 `외적을 제곱한 결과`와 현재 슬라이드의 `내적을 제곱한 결과`를 비교해 보면 마지막 term이 `서로 상쇄` 될 수 있음을 보입니다. 즉 마지막 term 합은 0입니다.

<br>

<img src="../assets/img/math/la/cross product and sin of angle/3.png" alt="Drawing" style="width: 600px;"/>

+ 최종적으로 $$ \Vert \vec{a} \times \vec{b} \Vert^{2} + (\Vert \vec{a} \Vert \ \Vert \vec{b} \Vert^{2} cos\theta)^{2} $$ 을 정리해 보면
    + 다음과 같습니다. $$ \Vert \vec{a} \times \vec{b} \Vert^{2} = \Vert \vec{a} \Vert^{2} \ \Vert \vec{b} \Vert^{2}(1 - cos^{2}\theta $$ 가 됩니다.
    + 그리고 $$ sin^{2}\theta  = 1 - cos^{2}\theta $$ 를 이용하면
    + 처음에 정의한 식인 $$ \Vert \vec{a} \times \vec{b} \Vert = \Vert \vec{a} \Vert \ \Vert \vec{b} \Vert sin \theta $$ 를 얻을 수 있습니다.