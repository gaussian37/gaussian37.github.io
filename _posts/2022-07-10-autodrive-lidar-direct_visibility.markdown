---
layout: post
title: Direct Visibility of Point Sets 
date: 2022-07-10 00:00:00
img: autodrive/lidar/direct_visibility/0.png
categories: [autodrive-lidar] 
tags: [라이다, 포인트 클라우드, direct visibility of point set, hidden points removal] # add tag
---

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/y77VcOot_Aw" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

- 논문 : http://www.weizmann.ac.il/math/ronen/sites/math.ronen/files/uploads/katz_tal_basri_-_direct_visibility_of_point_sets.pdf

<br>

- 이번 글은 `Direct Visibility of Point Sets` 라는 논문에 대하여 리뷰하도록 하겠습니다.
- 라이다를 통해 취득한 포인트 클라우드를 다룰 때, 현재 시점에서 볼 수 있는 포인트 클라우드의 정보를 얻고 싶을 때 이 논문에서 다루는 개념을 사용할 수 있습니다.
- 그래픽스에서는 가상의 공간에 포인트 클라우드를 미리 만들어 놓고 사용자의 시점에 따라 화면에서 보이는 점들만을 이용하여 렌더링을 해야하는 문제점이 있었고 이와 같은 문제를 해결하는 Task를 `Visibility` 라고 정의하고 이와 관련된 연구들을 진행하였습니다.
- 본 글에서 다루는 논문 `Direct Visibility of Point Sets`는 `Visibility`와 관련된 다양한 논문 중 컨셉이 간단하고 계산 비용도 효율적이면서 일정 성능을 보장하기 때문에 많이 사용하는 방법으로 알려져 있고 `HPR(Hidden Point Removal)` 연산으로 불립니다.

<br>

