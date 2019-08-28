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

- 이번에 다루어 볼 주제는 `Entropy`, `KL divergence`, `Mutual Information`입니다.
- 먼저 정보이론을 확률이론 및 결정이론과 비교하여 간단하게 식으로 알아보겠습니다.
- **Probability Theory**
    - 불확실성(Event, 변수등)에 대한 일어날 **가능성을 모델링** 하는 것입니다.
    - 　$$ P(Y \vert X) = \frac{ P(X \vert Y)P(Y) }{ P(X) } $$
- **Decision Theory**
    - 불확실한 상황에서 **추론에 근거해 결정**을 내리는 것입니다.
    - 　$$ Y = 1 $$, if $$ \frac{ P(x \vert y = 1)P(y=1) }{ P(x \vert y = 0)P(y=0) } > 1 $$ 
- **Information Theory**
    - 확률 분포 $$ P(x) $$의 불확실성 정도를 평가하는 방법
    - 　$$ H(X) = -\sum_{x}P(x)log_{2}P(x) $$
    
<br>

- 그러면 먼저 간단하게 `정보량(Information)`이 무엇인지 알아보도록 하겠습니다.
- 예를 들어 1 ~ 16까지 16개 숫자 중에서 1개의 숫자를 하나 생각한 다음 다른 사람이 내가 생각한 숫자를 맞추도록 하려고 합니다. 이 때 최소 몇 번 물어야 할까요?
    - 질문을 들으면 아시겠지만 이진 탐색으로 풀 수 있는 쉬운 문제입니다.
- 이 문제의 정답은 4번 입니다. 이진탐색으로 정답을 찾게 되면 $$ n = log_{2}16 = 4 $$(단위: bit)가 됩니다.
- 여기서 `정보량`을 정의할 수 있는데, **불확실함을 해소하기 위해 필요한 질문(정보)의 수(불확실한 정도)라고** 할 수 있습니다.
- 특히, 정보량의 단위는 `bit`로 표현하게 됩니다. (Yes / No 대답에 해당합니다.)

<br>

- 여기서 **정보량과 확률의 관계**를 살펴보겠습니다.
- 카드 한장을 선택할 확률 $$ p = 1/16 $$으로 모두 동일합니다.
- 선택한 카드를 맞추기 위한 정보량을 확률 $$ p $$를 이용해 표현해 보면
    - 　$$ n = -log_{2}(p) = log_{2}(1/p) = log_{2}(16) = 4 $$ 가 됩니다.
- 일반적으로 어떤 사상의 확률이 $$ p $$라고 하였을 때, 그 사상에 대한 정보량 $$ I $$는 다음과 같습니다.
    - 　$$ I = log_{2}(\frac{1}{p}) = -log_{2}(p) $$

<br>

- 마지막으로 정보 이론에 대하여 정리해 보도록 하겠습니다.
- `Entropy`는 확률 분포 $$ p(x) $$에서 일어날 수 있는 모든 사건들의 **정보량의 기댓값**으로 $$ p(x) $$의 불확실성 정도를 나타냅니다.
- `Cross Entropy`는 실제 데이터 $$ P $$의 분포로부터 생성되지만, **분포 Q를 사용하여 정보량을 측정**해서 나타낸 평균적 bit수를 의미합니다.
- `KL divergence`는 두 확률 분포 $$ P $$와 $$ Q $$의 차이를 측정합니다.
- `Mutual Information`은 두 확률 변수들이 **얼마나 서로 dependent한 지 측정**합니다.