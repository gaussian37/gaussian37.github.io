---
layout: post
title: 칼만필터 인트로덕션
date: 2019-10-27 00:00:00
img: vision/kalmanfilter/kalman.PNG
categories: [vision-kalmanfilter] 
tags: [컴퓨터 비전, 칼만 필터, kalman filter] # add tag
---

- 출처: Understanding Kalman Filters (Mathworks), Bayesian Inference : Kalman filter에서Optimization까지

<br>
 
- 이번 글에서는 칼만 필터와 관련하여 전체적으로 한번 살펴보려고 합니다. 전체 내용을 한번 쓱 살펴보겠습니다.
- 칼만 필터는 루돌프 칼만에 의해서 개발된 `Optimal estimation algorithm`입니다.
- 현재 내비게이션, 컴퓨터 비전, 신호 처리 등등 다양한 분야에서 사용중에 있고 최초로 사용된 것은 아폴로 프로젝트 였다고 전해집니다.

<br>

- 그러면 칼만 필터는 언제 사용될까요?
- 주로 사용 되는 것은 알고 싶은 변수를 직접적으로 확인할 수는 없고 `변수를 간접적인 방법으로 유도` 해서 알아내야 할 때 사용할 수 있습니다.
    - 예를 들면 직접적으로 확인하고 싶은 곳이 너무 온도가 높거나 아니면 외부 환경이라서 센서를 설치할 수 없는 경우가 있습니다.
- 또는 다양한 센서들을 통하여 값을 측정할 수는 있지만 노이즈가 발생할 때 `노이즈 문제를 개선`하기 위한 경우에 사용할 수 있습니다.    