---
layout: post
title: (CLRS) 선분의 교차성 결정
date: 2020-04-25 00:00:00
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

- ### 선분의 순서화
- ### 검사선의 이동
- ### 선분 교차 알고리즘
- ### 선분 교차 알고리즘의 예
- ### 선분 교차 알고리즘의 정확성
- ### 선분 교차 알고리즘의 수행 시간

<br>

- [이전 글]()에서 살펴본 내용은 선분이 2개가 있을 때, 2개의 선분의 관계에 대하여 알아보았습니다.
- 이번 글에서 살펴볼 내용은 **임의의 선분 n개**가 존재할 때, **n개의 선분들 사이에 교차되는 점**이 있는 지, 선분들 간의 관계를 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/algorithm/clrs_determining_whether_any_pair_of_segments_intersects/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>



<br>

- [알고리즘 글 목록](https://gaussian37.github.io/math-algorithm-table/)

<br>