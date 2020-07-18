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

- ### Abstract
- ### Introduction
- ### Related works
- ### Proposed Harmonic DenseNet
- ### Experiments
- ### Discussion
- ### Concolusion
- ### Pytorch Code

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

## **Proposed Harmonic DenseNet**

<br>

- `HarDNet`에서는 DenseNet에 기반한 새로운 architecture를 제안합니다. 
- 먼저 `LogDenseNet`에서 제안한 방법은 layer $$ k $$를 $$ k - 2^{n} (\text{where  } n \ge 0, k - 2^{n} \ge 0) $$ 번째 layer와 연결합니다. 
- `HarDNet`에서는 `LogDenseNet`의 방법을 조금 더 sparse하게 만듭니다. layer $$ k $$를 $$ k - 2^{n} (\text{where  } 2^{n} divides k , n \ge 0, k - 2^{n} \ge 0) $$ 조건에 맞는 layer에 연결합니다. 즉, $$ k $$ 가 $$ 2^{n} $$에 나뉘어 지는 경우의 layer에만 연결합니다.