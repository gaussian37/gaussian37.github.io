---
layout: post
title: AutoEncoder의 모든것 (1)
date: 2019-02-24 00:00:00
img: dl/concept/autoencoder1/autoencoder.png
categories: [dl-concept] 
tags: [deep learning, autoencoder] # add tag
---

+ 출처 : https://www.youtube.com/watch?v=o_peo6U7IRM

+ 이 글은 오토인코더의 모든것 강의를 보고 요약한 글입니다.

<img src="../assets/img/dl/concept/autoencoder1/1-1.jpg" alt="Drawing" style="width: 800px;"/>

+ AutoEncoder와 관련된 키워드는 다음과 같습니다.
    + Unsupervised Learning
    + Representation learning = Efficient coding learning
    + Dimensionality reduction
    + Generative model learning
+ 여기서 가장 많이 알려진 키워드가 `Dimensionality reduction` 입니다.
    + 많은 사람들이 이 용도로 AutoEncoder를 사용하고 있습니다. 

<img src="../assets/img/dl/concept/autoencoder1/1-2.jpg" alt="Drawing" style="width: 800px;"/>

+ Nonlinear Dimensionality Reduction과 같은 용도로 사용되는 키워드는 위와 같습니다.
+ 이 때, `Feature Extraction`과 `Manifold learning`은 자주 사용되는 용어입니다.

<img src="../assets/img/dl/concept/autoencoder1/1-3.jpg" alt="Drawing" style="width: 800px;"/>

+ 벤지오의 기술분류표를 보면 AutoEncoder는 Representation Learning에 해당합니다.

<img src="../assets/img/dl/concept/autoencoder1/1-4.jpg" alt="Drawing" style="width: 800px;"/>

+ 이안 굿펠로우의 기술분류표를 보면 `Variational Autoencoder`는 `Maximum Likelihood`와 연관이 되어 있습니다.
+ 따라서 Maximum Likelihood에 대해서도 알아보도록 하겠습니다. 

<img src="../assets/img/dl/concept/autoencoder1/1-5.jpg" alt="Drawing" style="width: 800px;"/>

+ AE(AutoEncoder)는 입력과 출력이 같은 동일한 구조를 가지고 있으면 AE라고 부릅니다.
+ 이 때 관계된 키워드가 앞에서 언급한 바와 같이 크게 4가지가 있습니다.
+ **오토인코더를 학습**할 때,
    + `Unsupervised learning` : 학습 방식에 해당합니다.
    + `ML density estimation` : `Loss function`이 `Negative Maximum Likelihood`로 해석을 해서 Loss를 Minimize 하면 `Maximum Likelihood`가 됩니다.
+ **학습된 오토인코더**에서
    + `Manifold learning` : 인코더는 차원 축소 역할을 수행하고 (일반적으로 가운데 차원은 입력 보다 작음) 이것을 Manifold learning 이라고 합니다.
    + `Generative model learning` : manifold learning을 한 결과를 다시 입력 차원과 똑같이 복원을 하는 데 이것을 Generative model 이라고 합니다.   

<img src="../assets/img/dl/concept/autoencoder1/1-6.jpg" alt="Drawing" style="width: 800px;"/>

<img src="../assets/img/dl/concept/autoencoder1/1-7.jpg" alt="Drawing" style="width: 800px;"/>

<img src="../assets/img/dl/concept/autoencoder1/1-8.jpg" alt="Drawing" style="width: 800px;"/>

<img src="../assets/img/dl/concept/autoencoder1/1-9.jpg" alt="Drawing" style="width: 800px;"/>

<img src="../assets/img/dl/concept/autoencoder1/1-10.jpg" alt="Drawing" style="width: 800px;"/>




