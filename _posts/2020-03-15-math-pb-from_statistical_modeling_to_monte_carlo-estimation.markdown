---
layout: post
title: Statistical modeling, Bayesian modeling, Monte carlo estimation, Markov chain.
date: 2020-03-15 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [Statistical modeling, Bayesian modeling, Monte carlo estimation, Markov chain.] # add tag
---

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

- 이번 글에서는 `Statistical modeling`, `Bayesian modeling`, `Monte carlo estimation` 그리고 `Markov chain` 까지 연계해서 다루어보도록 하겠습니다.

<br>

## **Statistical modeling**

<br>

### **Statistical model이란?**

- `statistical model`이 무엇일까요?  statistical model은 **데이터 생성 과정을 모델링** 한 것입니다. statistical model은 다양한 변수들의 관계에 대하여 설명하는데 이 변수들은 데이터 속에서 매우 다양하게 나타나고 불확실하게 나타나는 특성을 가집니다. 
- 예를 들어 변수들이 너무 다양하거나 변수들의 관계가 너무 복잡하면 변수들 간의 관계를 알아차리기가 어려워 random behavior 형태로 데이터가 나타날 수 있습니다. 
- 이러한 random behavior들의 불확실성과 다양성을 나타내기 위해서는 `확률 이론`을 도입해서 나타낼 수 있습니다. 이 때 사용하는 것이 바로 `statistical model`이 됩니다.
- `statistical model`의 첫번째 목적은 **불확실성의 정량화(quanrify uncerntainty)**입니다.
    - 예를 들어 투표를 하였을 때, 투표율이 57% 라고 한다면 이 데이터를 정확히 믿을 수 있을까요? 어떤 집단에서 어떻게 뽑았느냐에 따라서 데이터의 정합성이 달라질 수 있습니다. 만약 99%의 신뢰도로 (51%, 63%) 범위의 투표율을 가진다고 표현한다면 좀 더 적합해 보입니다. 즉, 불확실성에 대한 수치를 정량화 함으로써 막연한 데이터를 좀 더 수치적으로 표현할 수 있고 좀 더 많은 정보를 얻을 수 있습니다.
- `statistical model`의 두번째 목적은 **어떤 가정에 대한 근거를 마련**하기 위함입니다.
    - 예를 들어 어떤 후보에 대한 지지율이 여성들은 55%, 남성들은 59%이므로 남성들이 더 선호한다 와 같이 근거를 마련하는 데 사용됩니다. 
- `statistical model`의 세번째 목적은 **예측(prediction)**입니다. 사실 이 목적으로 저희는 statistical model을 많이 사용합니다. 
    - 다양한 변수들을 조합하였을 때, 어떤 후보의 지지율이 가장 높을 지 예측 하거나 어떤 변수들을 조합하였을 때, 이런 패턴의 투표자들은 어떤 후보를 뽑을 지 등등 다양한 예측을 하는데 statistical model이 사용됩니다. 특히 이러한 방법들을 machine learning 으로 사용되는데 machine learning의 주 목적인 prediction을 하는데 model이 사용됩니다.

<br>

### **Modeling process**

<br>

- 그러면 `statistical model`을 어떻게 모델링 하는 지 스텝 별로 차례대로 알아보도록 하겠습니다.

<br>

## **Bayesian modeling**

<br>

- 그러면 앞에서 배운 

<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>