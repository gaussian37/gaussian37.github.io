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

<br>

## **푸리에 시리즈를 배우는 이유**


## **푸리에 급수의 의미와 주파수 분석에서의 활용**

<br>

- 아래 두가지 식은 푸리에 급수에 관련된 동일한 식을 다른 관점으로 표현한 식입니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 식의 관점은 **연속 신호(함수)는 무한 차원 벡터이고, 이것은 기저 벡터의 선형 결합으로 재구성할 수 있음**입니다.

<br>
<center><img src="../assets/img/vision/concept/fourier_transform/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 위 식의 관점은 **신호를 구성하는 각 기저 벡터는 얼마 만큼의 기여도를 가지고 있는지를 뜻**하며 이는 `주파수 분석에 활용 가능`합니다.

<br>

- 푸리에 급수에 대하여 알아보기 이전에 `벡터`와 `직교`에 대하여 간략하게 알아보도록 하겠습니다.

<br>

- 벡터 $$ \overrightarrow{i}, \overrightarrow{j} $$가 직교한다면 내적 $$ \overrightarrow{i} \cdot \overrightarrow{j} = 0 $$을 만족합니다.
- 이렇게 직교하는 벡터들을 이용하여 단위 벡터를 만들 수 있는데, 2차원에서 2개의 단위 벡터가 있으면 2차원 공간의 모든 벡터를 표현할 수 있습니다. 같은 개념으로 **N차원 공간의 모든 벡터를 표현하기 위해서는 N개의 직교하는 벡터가 필요**합니다.

<br>

- 앞으로 다룰 내용은 함수이기 때문에 벡터가 아닌 함수를 다루어야 합니다. 함수에서도 벡터의 직교와 같은 개념이 있을까요?
- 먼저 함수는 일반화된 벡터로 취급할 수 있습니다. N차원 벡터는 N개의 숫자 나열이듯이 실수 함수는 실수 값을 무한 개 나열한 것으로 본다면 **실수 함수는 무한 차원 벡터**라고 볼 수 있습니다.