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

## **1. 칼만 필터란 무엇일까?**

<br>

- 칼만 필터는 일부 동적 시스템에 대한 정보가 확실하지 않은 곳에서 사용할 수 있으며 시스템이 다음에 수행 할 작업에 대한 정확한 추측을 할 수 있습니다.
- 칼만 필터는 센서를 통해 추측한 움직임에 노이즈가 들어오더라도 노이즈 제거에 좋은 역할을 합니다.
- 칼만 필터는 지속적으로 변화하는 시스템에 이상적입니다. 왜냐하면 어떤 연산 환경에서는 메모리가 부족할 수 있는데 칼만 필터에서는 이전 상태 이외의 기록을 유지할 필요가 없기 때문입니다. 
- 또한 연산 과정 또한 빠르기 때문에 `실시간 문제` 및 `임베디드 시스템`에 적합합니다.
- 이 글에서 살펴볼 `Linear` 칼만 필터는 기본적인 확률과 행렬에 대한 지식만 있으면 이해 가능합니다. 여기까지 읽고 관심이 있으시면 아래 내용을 한번 살펴보시기 바랍니다.

<br>

## **2. 칼만 필터로 무엇을 할 수 있을까?**

<br>



<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/robot_ohnoes.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>
