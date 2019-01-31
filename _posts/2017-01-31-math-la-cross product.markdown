---
layout: post
title: 벡터의 외적이란?
date: 2017-01-31 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, 벡터, 내적] # add tag
---

<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.3%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>

<br><br>

출처 : 칸 아카데미 선형대수학 (https://ko.khanacademy.org/math/linear-algebra/vectors-and-spaces)

이번 글에서는 벡터의 외적 즉, cross product에 대하여 알아보도록 하겠습니다.

<br>

<img src="../assets/img/math/la/cross product/1.png" alt="Drawing" style="width: 600px;"/>

+ 먼저 `내적(dot product)`은 모든 차원에서 정의되는 반면 `외적(cross product)`는 오직 $$ \mathbb R^{3} $$ 에서만 정의 됩니다.
+ 위의 슬라이드에서 외적의 정의인 $$ \vec{a} \times \vec{b} $$에 대하여 살펴보도록 하겠습니다.
    + 외적의 1행은 $$ a_{2}b_{3} - a_{3}b_{2} $$ 입니다. 마치 두 벡터의 1행을 제외하고 determinant를 구한 것과 같습니다.
    + 외적의 2행은 $$ a_{3}b_{1} - a_{1}b_{3} = -(a_{1}b_{3} - a_{3}b_{1}) $$ 입니다. 2행을 제외하고 determinant를 구한 것에 음수를 취한것과 같습니다.
    + 외적의 3행은 $$ a_{1}b_{2} - a_{2}b_{1} $$ 입니다. 두 벡터의 3행을 제외하고 determinant를 구한 것과 같습니다.
+ 외적의 정의에 따라 예제를 살펴보면 쉽게 구할 수 있습니다. 

<br>

<img src="../assets/img/math/la/cross product/2.png" alt="Drawing" style="width: 600px;"/>

+ 외적의 성질을 살펴보면 두 벡터 $$ \vec{a}, \vec{b} $$ 에 대하여 모두 `orthogonal` 합니다.
+ <img src="../assets/img/math/la/cross product/righthand.png" alt="Drawing" style="width: 600px;"/>
    + 오른손의 법칙을 이용하면 두 벡터가 있을 때, 어느 방향으로 orthogonal 하는 지 알 수 있습니다.
+ `orthogonal` 하다는 것은 orthogonal 한 두 벡터를 내적하였을 때 값이 0이라는 뜻입니다.    
 
<br>

<img src="../assets/img/math/la/cross product/3.png" alt="Drawing" style="width: 600px;"/>

+ 먼저 위의 예제를 보면 $$ (\vec{a} \times \vec{b}) \cdot \vec{a} = 0 $$ 임을 확인할 수 있습니다.

<br> 

<img src="../assets/img/math/la/cross product/4.png" alt="Drawing" style="width: 600px;"/>

+ 위의 예제를 보면 $$ (\vec{a} \times \vec{b}) \cdot \vec{b} = 0 $$ 임을 확인할 수 있습니다.
+ 다음 강의에서는 외적에 대한 다른 성질을 알아보겠습니다.
