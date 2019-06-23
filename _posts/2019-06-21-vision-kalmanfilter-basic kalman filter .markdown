---
layout: post
title: 칼만 필터 기초
date: 2019-06-23 00:00:00
img: vision/kalmanfilter/kalman.PNG
categories: [vision-kalmanfilter] 
tags: [컴퓨터 비전, 칼만 필터, kalman filter] # add tag
---

- 출처 : 칼만필터는 어렵지 않아
- 칼만필터 관련 코드는 python, c++로 바꿔서 글 올립니다.

- 이번 글에서는 칼만필터의 전체적인 구조 및 이론에 대하여 알아보도록 하겠습니다.
- 이 글에서는 칼만 필터의 이론적인 배경이나 증명 보다는 사용적인 측면에서의 이론을 다룹니다.
- 아래 스테이트 다이어그램은 칼만 필터의 전체 플로우를 나타냅니다.
- 이번 글의 목적은 아래 플로우 전체를 이해하기 위한 것이니 천천히 따라가시면 이해가 될 거라고 생각됩니다.

<br>

<center><img src="../assets/img/vision/kalmanfilter/basic/kalman.png" alt="Drawing" style="width: 600px;"/></center>

