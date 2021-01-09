---
layout: post
title: 베이즈 필터 (Bayes Filter)
date: 2020-02-01 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [Optimal State Estimation, 최정 상태 이론, 베이즈 필터, Bayes filter] # add tag
---

<br>

- 선수 지식 1 : [상태 방정식 (state equation)](https://gaussian37.github.io/autodrive-ose-state_equation/)
- 선수 지식 2 : [자율주행에서의 Localization과 Tracking](https://gaussian37.github.io/autodrive-ose-localization_and_tracking/)

<br>

- 이번 글에서는 Bayes Filter에 대하여 다루어 보도록 하겠습니다. Bayes Filter는 Kalman Filter와 Particle Filter의 개념의 기초가 되는 간단하지만 매우 중요한 Filter입니다. 따라서 Kalman Filter나 Particle Filter를 배우기 전 단계라면 베이즈 필터를 통하여 어떻게 동작하는 지 그 메커니즘을 먼저 배우기를 추천합니다.
- 만약 Kalman Filter와 Particle Filter에 대하여 잘 아신다면 아래 Bayes Filter의 개념과 비교해 보는 것을 추천 드립니다.

<br>

## **목차**

<br>



<br>

