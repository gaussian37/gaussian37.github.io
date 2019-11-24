---
layout: post
title: 선형 칼만 필터로 human tracking (w/ opencv)
date: 2019-11-19 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [칼만 필터, kalman filter, tracking, 트래킹] # add tag
---

<br>

- 이번 글에서는 opencv를 이용하여 칼만필터를 사용하는 방법에 대하여 알아보겠습니다.
- 전체적으로는 2D 이미지에서 컴퓨터 비전 알고리즘으로 Detection한 포인트를 선형 칼만 필터 알고리즘으로 tracking 할 예정입니다.
- 선형 칼만 필터의 자세한 원리를 이해하고 싶으면 제 블로그의 [이 링크](https://gaussian37.github.io/ad-ose-lkf_basic/)를 참조하시기 바랍니다. 이 글에서는 간략하게만 이해하고 넘어갈 예정입니다.
- 이 글에서는 opencv를 이용하여 어떻게 칼만 필터를 사용하는 지 관점으로 접근해 보도록 하겠습니다.

<br>

## **목차**
    - ### 칼만 필터의 역할
    - ### State Transition Equation

<br>

## **칼만 필터의 역할**

<br>

- 칼만 필터는 2가지 전략을 따릅니다.
- 1) **predict** : 시스템의 `내부 상태`에 대하여 예측을 합니다. (e.g. 드론의 위치와 속도) 이 예측은 `이전 내부 상태`와 `센서 등으로 부터 입력`된 값 (e.g. 드론의 프로펠러 force)을 기반으로 만들어집니다.
- 2) **update** : 내부 상태에 관련된 새로운 계측값 (e.g. GPS 정보)이 들어오면 이 값을 사용하여 **predict**에 반영합니다.

<br>

## **State Transition Equation**

<br>

- 먼저 칼만 필터에서 **predict**에 관련된 과정부터 이해해 보도록 하겠습니다.
- 