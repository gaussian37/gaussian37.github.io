---
layout: post
title: Basic Probability and Information Theory
date: 2019-08-21 00:00:00
img: gan/gan.png
categories: [gan-concept] 
tags: [probability, information theory, 정보이론, kl divergence] # add tag
---

- 이번 글에서는 `GAN`을 다루기 전에 필요한 기본적인 **확률과 정보이론**을 다루어 보려고 합니다.

<br>

- **1) Probability Model**
- **2) Discriminative Model**
- **3) Bayesian Theory**
- **4) Basic Information Theory**

<br>

<br>

### **4) Basic Information Theory**

<br>

- 정보이론

<br>

- 마지막으로 정보 이론에 대하여 정리해 보도록 하겠습니다.
- `Entropy`는 확률 분포 $$ p(x) $$에서 일어날 수 있는 모든 사건들의 **정보량의 기댓값**으로 $$ p(x) $$의 불확실성 정도를 나타냅니다.
- `Cross Entropy`는 실제 데이터 $$ P $$의 분포로부터 생성되지만, **분포 Q를 사용하여 정보량을 측정**해서 나타낸 평균적 bit수를 의미합니다.
- `KL divergence`는 두 확률 분포 $$ P $$와 $$ Q $$의 차이를 측정합니다.
- `Mutual Information`은 두 확률 변수들이 얼마나 서로 dependent한 지 측정합니다.