---
layout: post
title: 선형 칼만 필터
date: 2019-11-07 00:00:00
img: ad/kalmanfilter/kalman_filter.jpg
categories: [ad-kalmanfilter] 
tags: [칼만 필터, kalman filter, 선형 칼만 필터] # add tag
---

<br>

- 참조: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits

<br>

- 이번 글에서는 선형 칼만 필터를 통하여 칼만 필터의 기본 컨셉에 대하여 자세하게 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 1. 칼만 필터란 무엇일까?
- ### 2. 칼만 필터로 무엇을 할 수 있을까?
- ### 3. 어떻게 칼만필터가 문제를 다루는 지 살펴보자
- ### 4. 행렬을 통하여 문제 다루어 보기
- ### 5. measurement로 estimate 에 반영해 보기
- ### 6. gaussian 결합
- ### 7. 앞에서 다룬 내용 종합

<br>

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/robot_ohnoes.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>
