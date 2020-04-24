---
layout: post
title: (CLRS) 알고리즘의 역할
date: 2020-04-10 00:00:00
img: math/algorithm/algorithm.png
categories: [math-algorithm] 
tags: [algorithm, 알고리즘] # add tag
---

<br>

- [알고리즘 글 목록](https://gaussian37.github.io/math-algorithm-table/)

<br>

- 이 글은 CLRS(Introduction to algorithm) 책을 요약한 것이므로 자세한 내용은 CLRS 책을 참조하시기 바랍니다.
- CLRS 내용 외에 따로 정리한 선분의 성질 관련 블로그 글은 아래 링크를 참조하시기 바랍니다.
    - `CCW` : https://gaussian37.github.io/math-algorithm-ccw/
    - `선분의 교차` :  https://gaussian37.github.io/math-algorithm-line_intersection/

<br>

## **목차**

<br>

- ### 선분의 정의
- ### 선분 관련 문제
- ### 벡터 곱

<br>

## **선분의 정의**

<br>

- 점 $$ p_{1} = (x_{1}, y_{1}), p_{2} = (x_{2}, y_{2}) $$이 있을 때, 선분(`line segment`) $$ \bar{p_{1}p_{2}} $$는 양 끝점을 $$ p_{1}, p_{2} $$로 합니다.
- 이 때, 선분 위의 점들을 $$ (x_{3}, y_{3}) $$ 라고 하면, $$ (x_{1}, y_{1}), (x_{2}, y_{2}) $$와 다음 관계를 가집니다.
    - $$ x_{3} = \alpha x_{1} + (1 - \alpha)x_{2} $$
    - $$ y_{3} = \alpha y_{1} + (1 - \alpha)y_{2} $$
    - $$ \text{where, } 0 \ge \alpha \le 1
- 만약 선분이 아니라 방향 성분이 필요한 방향 선분(`directed segment`)라면 $$ \vec{p_{1}p_{2}} $$라고 정의해야 합니다.
- 만약 $$ p_{1} $$이 원점이라면 방향 선분 $$ \vec{p_{1}p_{2}} $$ 를 벡터 $$ \vec{p_{2}} $$로 나타낼 수 있습니다.

<br>

## **선분 관련 문제**

<br>

- 이 책에서 다루는 선분과 관련된 대표적인 문제는 아래 3가지 입니다.
- ① $$ \vec{p_{1}p_{2}}, \vec{p_{0}p_{2}} $$에 대해 $$ \vec{p_{0}p_{2}} $$가 $$ \vec{p_{0}p_{1}} $$의 시계 방향에 있는가? (두 선분의 방향성)
    - 이 문제의 경우 두 선분의 시작점이 둘 다 $$ p_{0} $$ 인 것에 유의하시면 됩니다.
- ② $$ \vec{p_{1}p_{2}}, \vec{p_{0}p_{2}} $$에 대해 $$ \vec{p_{0}p_{1}} $$을 순회한 다음 $$ \vec{p_{1}p_{2}} $$를 순회할 때, 점 $$ p_{1} $$에서 좌회전 하는가?
    - 이 문제는 ①과 유사하지만 첫번째 방향 선분의 끝점이 두번쨰 방향 선분의 시작점이 된다는 차이점이 있습니다.
- ③ $$ \bar{p_{1}p_{2}} $$와 $$ \bar{p_{3}p_{4}} $$가 서로 교차하는가?

<br>

- 위 3가지 문제를 어떻게 해결하는 지 이 글에서 자세하게 다루어 보도록 하겠습니다.

<br>

- [알고리즘 글 목록](https://gaussian37.github.io/math-algorithm-table/)

<br>