---
layout: post
title: gymnasium 학습 정리
date: 2025-09-02 00:00:00
img: ml/rl/gymnasium/0.png
categories: [ml-rl] 
tags: [강화 학습, reinforcement learning] # add tag
---

<br>

[RL 관련 글 목차](https://gaussian37.github.io/ml-rl-table/)

<br>

- 본 글은 `gymnasium`의 강화 학습 내용을 공부하면서 실험해본 내용을 정리한 글입니다.
- 각 실험들은 다음 조건들을 이용하여 실험하였습니다.
    - ① 같은 문제를 `continuous` 입력 문제로 풀 수 있으면 `discrete` 입력 대신에 `continuous` 문제로 풀었습니다. `continuous` 문제가 좀 더 현실적이기 때문입니다.
    - ② 난이도를 높일 수 있으면 더 높인 상태로 문제를 풀었습니다.
    - ③ `stable baseline3`를 이용하여 문제를 해결하였습니다. 가능한 코드를 간단하게 작성하기 위함과 학습에 필요한 방법을 빠르게 확인하기 위함입니다. 만약 pytorch scratch 코드가 필요하면 해결한 코드를 `GPT`에 요청하면 pytorch scratch 버전으로 변환 요청하면 변환해 줍니다.
    - ④ `discrete` 입력 문제에는 `PPO` 알고리즘을 이용하였고 `continous` 입력 문제에는 `SAC` 알고리즘을 이용하였습니다.

<br>

- ## **목차**
    - ### Acrobot
    - ### Cart Pole
    - ### Mountain Car Continuous
    - ### Pendulum
    - ### Bipedal Walker (Continuous + Hardcore)
    - ### Car Racing
    - ### Lunar Lander Continuous

<br>

[RL 관련 글 목차](https://gaussian37.github.io/ml-rl-table/)

<br>