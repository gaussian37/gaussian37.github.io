---
layout: post
title: HarDNet(A Low Memory Traffic Network)
date: 2020-07-06 00:00:00
img: dl/concept/hardnet/0.png
categories: [dl-concept] 
tags: [딥러닝, HarDNet, harmonious densenet, densenet, densely connected convolution networks] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 논문 : https://arxiv.org/abs/1909.00948
- 이 네트워크는 DenseNet을 기반으로 만들어 졌기 때문에 아래 링크의 DenseNet을 반드시 읽은 후 보시길 추천드립니다.
- DenseNet : https://gaussian37.github.io/dl-concept-densenet/

<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [Introduction](#introduction-1)
- ### [Related works](#related-works-1)
- ### [Proposed Harmonic DenseNet](#proposed-harmonic-densenet-1)
- ### [Experiments](#Experiments-1)
- ### [Discussion](#Discussion-1)
- ### [Concolusion](#Concolusion-1)
- ### [Pytorch Code](#pytorch-code-1)

<br>

## **Abstract**

<br>

- 많은 네트워크들이 `MAC(number of multiply-accumulate operations or floating point operations)`를 낮추면서 좋은 accuracy 성능을 내기 위하여 발전되어 왔습니다.
- 하지만 단순히 MAC 수치를 이용하여 inference 시간을 예상하는 것에는 한계가 있습니다. 이 논문에서는 `memory traffic` 계산하는 것을 제안합니다. 이 방법은 feature map의 크기를 이용하여 수치화 합니다. 왜냐하면 중간 단계에 있는 feature map의 크기가 inference 시간과 밀접한 관계가 있기때문입니다.
- 논문에서 제안하는 `Harmonic Densely Connected Network`는 MAC와 Memory traffic을 낮추어 효율성을 높입니다. 또한 NVIDIA profiler와 ARM Scale-Sim이라는 tool을 이용하여 Memory traffic을 낮추었을 때, inference 시간도 이에 비례하여 줄어든다는 것을 확인하였습니다.
- 따라서 논문에서는 high resolution 영상을 edge computer에서 이용할 때 memory traffic을 고려하여 네트워크를 설계하는 것을 권장합니다.

<br>

## **Proposed Harmonic DenseNet**

<br>

- `HarDNet`에서는 DenseNet에 기반한 새로운 architecture를 제안합니다. 
- 먼저 `LogDenseNet`에서 제안한 방법은 layer $$ k $$를 $$ k - 2^{n} (n \ge 0, k - 2^{n} \ge 0) $$ 번째 layer와 연결합니다. 
- `HarDNet`에서는 `LogDenseNet`의 방법을 조금 더 sparse하게 만듭니다. layer $$ k $$를 $$ k - 2^{n} (2^{n} divides k , n \ge 0, k - 2^{n} \ge 0) $$ 조건에 맞는 layer에 연결합니다. 즉, $$ k $$ 가 $$ 2^{n} $$에 나뉘어 지는 경우의 layer에만 연결합니다.
- 참고로 이 논문에서 layer 0은 input을 뜻합니다.

<br>
<center><img src="../assets/img/dl/concept/hardnet/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림의 아래쪽을 보면 HarDNet Block의 구조를 볼 수 있습니다. DenseNet, LogDenseNet에 비해 connection 수가 줄어든 것을 볼 수 있습니다.
- 위와 같은 방법을 통하여 concatenation 에 필요한 메모리, 계산 비용등을 줄일 수 있는 것이 HarDNet의 핵심 내용입니다.


<br>

- 이 논문에서 `harmonic` 이라는 용어를 사용한 이유는 논문에서 제시하는 block에서 layer 간 concatenation 하는 양상이 harmonic 한 형태로 이루어지기 떄문입니다.
- 하모닉(Harmonic)이란 우리말로는 '고조파'라고 부릅니다. 참고로 고주파는 주파수가 높은 전자기파를 말하며, 고조파는 바로 harmonic을 말하는 것입니다.
- Harmonic의 정의는 원천주파수(Fundamental Frequency)의 배수 주파수 성분을 말합니다. 예를 들어 1.2GHz의 Harmonic 주파수는 2.4GHz, 3.6GHz, 4.8GHz... 등이 됩니다. 아래 그림을 참조하시기 바랍니다.

<br>
<center><img src="../assets/img/dl/concept/hardnet/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>