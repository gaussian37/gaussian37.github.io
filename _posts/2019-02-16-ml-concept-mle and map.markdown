---
layout: post
title: MLE(Maximum Likelihood Estimation)와 MAP(maximum a posteriori)
date: 2019-02-16 00:00:00
img: ml/concept/mle-and-mlp/mle.jpg
categories: [ml-concept] 
tags: [ml, machine learning, 머신 러닝, mle, map, 우도, 사전확률, 사후확률] # add tag
---

+ 출처 : 패턴인식(오일석)
+ 이번 글에서는 MLE(Maximum Likelihood Estimation)와 MAP(maximum a posteriori)에 대하여 알아보도록 하겠습니다.


## MLE와 MAP의 개념적 설명

+ 먼저 MLE는 개념적으로 어떠한 형태의 분포에도 적용 가능합니다.
    + 현실적으로는 정규 분포와 같이 매개 변수로 표현되는 경우에만 적용 가능한 데 매개 변수로 표시 된 경우만 계산이 가능하기 때문입니다.
+ 이 매개 변수 집합을 $$ \theta $$ 라고 하겠습니다.
+ 이 때, 문제를 다음과 같이 정의 할 수 있습니다.
    + **주어진 X를 발생시켰을 가능성이 가장 높은 $$\theta$$를 찾아라**
    + **X에 대하여 가장 큰 likelihood를 갖는 $$\theta$$를 찾아라**  
+ <img src="../assets/img/ml/concept/mle-and-mlp/mleGraph.PNG" alt="Drawing" style="width: 300px;"/>
    + 위 그래프에서 X는 6개의 샘플을 갖습니다.
    + 이 X를 발생시킬 가능성은 $$ \theta_{1} $$이 $$ \theta_{2} $$ 보다 높습니다.
    + likelihood를 이용하여 다시 표현하면 $$ p(X \vert \theta_{1}) > p(X \vert \theta_{2}) $$ 라고 할 수 있습니다.  
    + 이 때 우리가 풀어야 할 문제는 **어떤 $$ \theta $$가 최대 우도**를 가질까 입니다.
    