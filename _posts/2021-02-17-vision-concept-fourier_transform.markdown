---
layout: post
title: Fourier transform (퓨리에 변환)
date: 2021-02-17 00:00:00
img: vision/concept/fourier_transform/0.png
categories: [vision-concept] 
tags: [fourier transform, 퓨리에 변환] # add tag
---

<br>

- 참조 : Introduction to Computer Vision
- 참조 : https://www.youtube.com/c/AngeloYeo (공돌이의 수학정리노트)
- 참조 : https://youtu.be/TB1A2-Db67s

<br>

- 이번 글에서는 기본적인 퓨리에 변환 (Fourier transform)에 대하여 다루어 보도록 하겠습니다.
- 보다 자세한 내용의 퓨리에 변환은 아래 링크를 참조해 주시기 바랍니다. 이 글은 신호와 시스템 전반적인 내용을 다루며 그 중 퓨리에 변환에 대한 자세한 내용을 확인하실 수 있습니다.
    - 링크 : [https://gaussian37.github.io/vision/signal/](https://gaussian37.github.io/vision/signal/)

<br>

## **목차**

<br>

- ### **푸리에 시리즈를 배우는 이유**
- ### **푸리에 급수의 의미와 주파수 분석에서의 활용**
- ### **연속 시간 푸리에 급수 유도**
- ### **이산 시간 푸리에 급수 유도**
- ### **연속 시간 푸리에 변환 유도**
- ### **이산 시간 푸리에 변환 유도**
- ### **푸리에 변환에서 음의 주파수**
- ### **라플라스 변환과 푸리에 변환**