---
layout: post
title: Luis Serrano의 PCA(Principal Component Analysis) 강의
date: 2020-07-19 00:00:00
img: ml/concept/serrano_pca/0.png
categories: [ml-concept] 
tags: [machine learning, PCA, Principal Component Analysis, 주성분 분석, Dimensional Reduction, 차원 축소] # add tag
---

<br>

- 참조 : https://youtu.be/g-Hb26agBFg?list=WL
- 이번 글에서는 주성분 분석 (Principal Component Analysis)에 대하여 다루어 보도록 하겠습니다.
- 제 블로그에 `PCA`와 관련된 글들이 더 있으니 아래 링크를 참조하셔도 됩니다.
    - Andrew Ng PCA lecture : https://gaussian37.github.io/ml-concept-andrewng-pca/
    - Mathematics for Machine Learning : https://gaussian37.github.io/math-mfml-table/

<br>

## **목차**

<br>

- ### Dimensionality Reduction


<br>

## **Dimensionality Reduction**

<br>

- PCA의 목적인 Dimensionality Reduction이란 feature의 갯수를 줄여서 나타내는 방법입니다. 
- feature의 갯수가 많으면 시각화 해서 보기 어려운 문제가 있으므로 임시로 2, 3 개의 feature만 선택하여 시각화 하는 방법을 사용하거나 feature가 너무 많아서 성능이 나오지 않는 경우 일부로 feature의 갯수를 줄일 수 있습니다.
- 그러면 PCA의 목적인 Dimensionality Reduction에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/ml/concept/serrano_pca/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림의 가운데에 있는 점들을 빨간색, 파란색 체크한 두 선에 정사영(projection) 하려고 합니다.
- 그러면 각 선에 정사영 된 점들이 위 그림 처럼 배치됩니다. 어떤 선에 정사영 한 것이 더 좋을까요?
- 느낌적으로 생각하였을 때, 파란색 체크한 선에 정사영 한 것이 더 나아 보입니다. 왜냐하면 점들이 기존 데이터 성향에 맞게 잘 분포되어 있기 떄문입니다.
- 두 선분에 따라 정사영한 결과에 차이가 발생하는 이유는 각 선이 실제 데이터를 얼마나 잘 나타내는 지가 다르기 때문입니다.
- 위 예시에서는 파란색 선이 실제 데이터를 더 잘 표현하기 때문에 원본 데이터에서 나타난 분포의 차이가 파란색 선에 정사영하였을 때에도 잘 나타나 집니다.
- 먼저 예제를 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/ml/concept/serrano_pca/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 2 축인 방의 갯수와 사이즈를 기준으로 2차원 평면위에 5개의 집 데이터를 표시해 보겠습니다. 그러면 위 그림처럼 나타나집니다.

<br>
<center><img src="../assets/img/ml/concept/serrano_pca/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이 때, 위 그림의 검은 선과 같이 데이터를 잘 표현할 수 있을 것 같은 기준을 정해야 합니다. 위 검은 선은 상당히 데이터 들을 잘 반영하도록 정해져 있습니다.

<br>
<center><img src="../assets/img/ml/concept/serrano_pca/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이 때, 검은 선에 데이터 들을 정사영 하면 위 그림 처럼 나타낼 수 있습니다.

<br>
<center><img src="../assets/img/ml/concept/serrano_pca/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서는 사이즈와 방의 갯수 즉, 2가지 feature를 이용하여 데이터를 표현하였습니다.
- 사실 집의 사이즈나 방의 갯수는 상관 관계가 있습니다. 넓은 집이 많은 방을 가질 수 있기 때문입니다.
- 위 그림에서도 2개의 차원을 1개의 차원으로 줄이고 이 때 사용된 1개의 차원을 ize feature 라고 하겠습니다.

<br>
<center><img src="../assets/img/ml/concept/serrano_pca/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 기존의 데이터를 어떤 feature에 정사영 시키면 차원을 줄일 수 있고 이 때 사용된 feature가 원래 데이터를 잘 나타내는 좋은 feature라면 정사영 하였을 떄에도 데이터가 기존의 데이터를 잘 나타낼 수 있습니다. 이 예시를 위 그림의 size feature를 통해 확인 하실 수 있습니다.