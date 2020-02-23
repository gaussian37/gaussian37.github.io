---
layout: post
title: 헤딩 각(heading angle) 계산
date: 2020-02-01 00:00:00
img: autodrive/concept/heading_angle/0.jpg
categories: [autodrive-concept] 
tags: [헤딩 각, heading angle] # add tag
---

<br>

[Autonomous Driving 관련 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>

- 이번 글에서는 이미지에서의 자동차 헤딩 각을 알아보는 방법에 대하여 다루어 보도록 하겠습니다.
- 이미지의 자동차 헤딩 각이므로 `2차원 평면`임을 가정하고 다룹니다.

<br>

## **목차**

<br>

- ### 헤딩각의 정의
- ### 벡터가 주어질 때, 그 벡터의 헤딩각
- ### 두 점이 주어질 때, 두 점의 헤딩각
    - 2) 두 점이 주어질 때, 두 점의 헤딩각
    - 3) 뒷 바퀴 두 점이 주어질 때, 

<br>

## **헤딩각의 정의**

<br>
<center><img src="../assets/img/autodrive/concept/heading_angle/0.jpg" alt="Drawing" style="width: 400px;"/></center>
<br>

- 헤딩각은 북쪽을 기준점으로 하였을 때, 어떤 물체의 방향이 기준점인 북쪽과 얼만큼의 회전 각도를 가지고 있는 지를 의미합니다.
- 헤딩각은 `0 ~ 180도`의 영역과 `-180 ~ 0도` 까지의 영역으로 나뉩니다. 위 그림과 같이 0도는 정 북쪽 방향을 뜻하고 북쪽에서 반시계방향으로 이동하면서 + 각도가 되고 시계방향으로 이동하면서 - 방향이 됩니다.

<br>
<center><img src="../assets/img/autodrive/concept/heading_angle/1.jpg" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이 방향은 자동차에서 바퀴의 방향과 일치하므로 헤딩각은 스티어링각과 동일한 값을 갖습니다.
- 이 때, 정 북쪽이 실제로 북쪽을 나타내는 각도인 경우 내 자동차의 헤딩 각도 또한 위 기준으로 표현할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/heading_angle/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 반면 나의 자동차를 절대 기준으로 잡고 내 자동차의 방향이 정 북쪽이라고 가정하면 나의 자동차를 기준으로 한 주위 자동차의 헤딩 각도를 표현할 수 있습니다.
- 위 그림과 같이 나의 자동차의 방향을 북쪽 이라고 하면 주위 자동차의 방향을 통하여 주위 자동차 들의 헤딩 각도를 구할 수 있습니다.

<br>

## **벡터가 주어질 때, 그 벡터의 헤딩각**

<br>

- 어떤 벡터가 주어지면 벡터에는 방향이 있기 때문에 헤딩각을 쉽게 구할 수 있습니다.



<br>

[Autonomous Driving 관련 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>