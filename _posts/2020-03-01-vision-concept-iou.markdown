---
layout: post
title: 다양한 IOU(Intersection over Union) 구하는 법
date: 2020-03-01 00:00:00
img: vision/concept/iou/0.png
categories: [vision-concept] 
tags: [IoU, Intersection over Union] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

**목차**

<br>

- ### IoU의 정의
- ### 두 영역이 직사각형이고 각 축과 수평할 때 IoU
- ### 두 영역이 임의의 볼록 다각형일 때 IoU

<br>

## **IoU의 정의**

<br>

- 이번 글에서는 `IoU`(Intersection Over Union)을 구하는 방법에 대하여 알아보도록 하겠습니다.
- `IoU`의 개념에 대해서는 많은 영상 및 블로그에서 다루고 있으니 간단하게만 설명하도록 하겠습니다. 아래 참조글을 참조하셔도 됩니다.
    - 참조 : https://inspace4u.github.io/dllab/lecture/2017/09/28/IoU.html

<br>
<center><img src="../assets/img/vision/concept/iou/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 두 영역이 위 처럼 겹칠 때, 얼만큼 겹친다고 정량적으로 나타낼 수 있는 방법이 `IoU`가 됩니다.

<br>
<center><img src="../assets/img/vision/concept/iou/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 여기서 `I`에 해당하는 Intersection은 두 영역의 교집합이 되고 `U`에 해당하는 Union은 두 영역의 합집합이 됩니다.
- 이 값을 구하기 위해서는 두 영역에 대한 정보를 이용하여 Intersection을 먼저 구하고 $$ A \cup B = A + B - A \cap B $$를 이용하여 Union의 값을 구하면 됩니다.

<br>
<center><img src="../assets/img/vision/concept/iou/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉 위와 같이 구하면 됩니다. 그러면 어떻게 위의 그림과 같이 구할 수 있는지 2가지 경우로 나누어서 살펴보도록 하겠습니다.
- 첫번째로 살펴 볼 경우는 **두 영역이 직사각형이고 각 축과 수평할 때 IoU** 입니다. 일반적인 bounding box가 있을 때 사용하는 방법입니다.
- 두번째 방법은 **두 영역이 임의의 볼록 다각형일 때 IoU** 입니다. 좀 더 유연하게 적용할 수 있는 방법인 반면에 `ccw`, `두 선분의 교차`의 개념이 필요합니다.

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

