---
layout: post
title: 평면 사이의 거리
date: 2017-02-12 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학, 벡터, 내적, 외적, 점 평면 거리] # add tag
---

<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.3%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>

<br><br>

출처 : 칸 아카데미 선형대수학 (https://ko.khanacademy.org/math/linear-algebra/vectors-and-spaces)

+ [선형대수학 전체 글 목록](https://gaussian37.github.io/math-la-Linear-Algebra-Table/) 

<img src="../assets/img/math/la/distance-between-planes/0.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 3개의 점을 이용하여 2개의 벡터를 만들어 보겠습니다.
+ ·$$ \frac{x-1}{2} = \frac{y-2}{2} = \frac{z-3}{4} $$ 식을 만족하는 두 개의 점을 구하면
    + (1, 2, 3)과 (3, 5, 7) 이 있습니다.
    + 이 두 점을 이용하여 벡터를 구하면 $$ \vec{a} = 2\hat{i} + 3\hat{j} + 4\hat{k} $$가 됩니다.
        + (3, 5, 7) - (1, 2, 3) 
+ ·$$ \frac{x-2}{3} = \frac{y-3}{4} = \frac{z-4}{5} $$ 식을 만족하는 점 하나와 첫 번째 식에서 사용한 점 (1, 2, 3)을 이용해서 벡터를 만들겠습니다.
    + 즉 첫 번째 식을 통해 도출한 점 (1, 2, 3)과 두 번째 식을 통해 도출한 점 (2, 3, 4)가 있습니다.
    + 이 두 점을 이용하여 벡터를 구하면 $$ \vec{b} = \hat{i} + \hat{j} + \hat{k} $$ 입니다.
        + (2, 3, 4) - (1, 2, 3)
+ 즉, (1, 2, 3)을 벡터의 시작점으로 공유하는 두 벡터 $$ \vec{a}, \vec{b} $$가 만들어 집니다.
+ 이 때, 두 벡터 $$ \vec{a}, \vec{b} $$ 모두와 직교인 벡터를 외적을 통해서 구하면
    + 위 슬라이드의 식을 참조하면 $$ \vec{a} \times \vec{b} = -\hat{i} + 2\hat{j} - \hat{k} = \vec{n} $$이 됩니다.

<img src="../assets/img/math/la/distance-between-planes/1.PNG" alt="Drawing" style="width: 600px;"/>

+ 임의의 점 (x, y, z)가 두 선을 이용하여 생성한 평면(파란색 평면)에 있다고 가정하겠습니다.
+ (x, y, z)와 (3, 5, 7)을 연결한 벡터는 앞에서 정의한 $$ \vec{n} $$과 직교한 관계를 가집니다.
+ 따라서 $$ \vec{n} \dot ((x - 3)\hat{i} + (y-5)\hat{j} + (z - 7)\hat{k} ) = 0 $$ 입니다.
    + ·$$ (-\hat{i} + 2\hat{j} -\hat{k}) \dot ((x - 3)\hat{i} + (y-5)\hat{j} + (z - 7)\hat{k}) = 0 $$
        + i의 계수 : $$ -(x-3) $$
        + j의 계수 : $$ 2(y-5) $$
        + k의 계수 : $$ -(z-7) $$
    + ·$$ 3-x + 2y-10 + 7-z = 0 $$
    + ·$$ x - 2y + z = 0 $$ : `파란색 면`의 식을 정의하였습니다.
+ 우리가 구해야 할 빨간색 면의 식은 $$ Ax -2y + z = d $$ 입니다.
+ 빨간색 면과 파란색 면은 평행하기 때문에 A = 1이 되어 평면의 식을 정의하면
    + ·$$ x -2y + z = d $$가 됩니다.
+ [앞에 글](https://gaussian37.github.io/math-la-point-distance-to-plane/)을 참조하여 점과 평면사이의 거리를 구해보겠습니다.
    + 문제의 조건에서 점과 평면사이의 거리가 $$ sqrt{6} $$ 이라고 하였으므로
    + ·$$ x -2y + z = d $$ 와 (1, 2, 3) 점 사이의 거리를 $$ sqrt{6} $$ 라고 정의하면 됩니다. 
    + distance = $$ \frac{1 -4 + 3 -d}{\sqrt{1 + 4 + 1}} = \frac{-d}{\sqrt{6}} = \sqrt{6} $$
    + 따라서 d = -6 입니다.
+ 문제의 정답 $$ \vert d \vert = 6 $$ 입니다.