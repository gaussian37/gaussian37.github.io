---
layout: post
title: Evaluating a Learning Algorithm (Andrew Ng)
date: 2019-04-12 03:49:00
img: ml/concept/machineLearning.jpg
categories: [ml-concept] 
tags: [machine learning, pca] # add tag
---

출처 : Andrew Ng 강의

+ 이번 글에서는 학습 알고리즘을 평가하는 방법에 대하여 간략하게 알아보도록 하겠습니다.

<img src="../assets/img/ml/concept/evaluate-algorithm/1.PNG" alt="Drawing" style="width: 600px;"/>

+ 위 슬라이드와 같이 `cost function`이 이와 같이 정의가 되어있다고 가정합시다.
+ 위 cost function은 linear regression에서의 기본적인 cost function이고 `regularization`이 적용된 형태입니다.
+ 만약 cost가 너무 크다면 step-by-step으로 접근하여 알고리즘을 수정할 필요가 있습니다.
    + 먼저 가장 효과적인 방법은 `training data`를 늘리는 방법입니다.
    + 그리고 학습데이터의 `feature`의 갯수를 작게 시작해서 더 늘려가는 방법이 좋습니다.
    + 또한 `feature`의 의미를 잘 이해하고 있다면 feature를 더 의미있게 활용하거나 가중치를 주기 위하여 `polynomial feature`를 사용하는 것도 추천합니다.
    + 그리고 `regularization`을 사용한 형태라면 $$ \lambda $$의 크기를 변형하는 것도 학습에 도움이 됩니다.

+ 아래 슬라이드는 대표적인 불안정한 학습 상태인 `overfitting` 입니다.

<img src="../assets/img/ml/concept/evaluate-algorithm/2.PNG" alt="Drawing" style="width: 600px;"/>

+ 만약 위 슬라이드 처럼 fitting한 결과가 같다면 학습 데이터에 너무 맞춰져서 새로운 데이터가 들어왔을 때 좋은 성능을 내기 어려울 수 있습니다.
+ 특히 위와 같이 `feature`의 갯수가 너무 많다면 overfitting될 가능성이 더 커집니다.

<img src="../assets/img/ml/concept/evaluate-algorithm/3.PNG" alt="Drawing" style="width: 600px;"/>

+ 학습 알고리즘의 성능을 평가하기 위해서는 데이터 셋을 나누어야 합니다. 학습할 데이터와 테스트할 데이터가 필요합니다.
+ 당연히, 학습 데이터의 양이 훨씬 더 많아야 합니다. 위 슬라이드의 학습 데이터의 비율을 70%를 한 것은 한가지 예이고 일반적으로 훨씬 더 많은 데이터를 학습에 사용합니다.
+ 학습 데이터와 테스트 데이터의 비율이 정해지면, 순서에 맞게 분할할 수도 있지만 랜덤으로 선택하여 데이터를 분할하는 방법을 많이 사용합니다.
    + 랜덤으로 데이터를 선택하면 혹시나 있을 수 있는 데이터 편향 문제를 방지할 수도 있습니다.
    
<img src="../assets/img/ml/concept/evaluate-algorithm/4.PNG" alt="Drawing" style="width: 600px;"/>

+ training data를 이용하여 파라미터 $$ \theta $$의 학습을 하여 training data의 error를 먼저 구합니다.
+ 그리고 학습한 결과를 이용하여 test data를 이용하여 error를 구하면 성능을 평가할 수 있습니다.
+ 가장 전형적인 `overfitting`의 예시는 training data에서의 error는 낮은 반면 test data에서의 error는 높은 경우 입니다.

<img src="../assets/img/ml/concept/evaluate-algorithm/5.PNG" alt="Drawing" style="width: 600px;"/>

+ 테스트를 할 때 모델의 성능을 cost function으로도 가늠할 수 있지만 좀 더 직접적인 성능 지표로 확인할 수 있습니다.
+ `classification` 문제를 예를 들면 잘못 분류한 경우를 카운트 하여 `Error`율을 계산할 수 있습니다.
    + 위 슬라이드의 logistic regression 예제를 보면 모델이 $$ h_{\theta}(x) \ge 0.5 $$인 경우를 0으로 판별하면 error 라고 판단합니다.
    + 반대로  $$ h_{\theta}(x) \lt 0.5 $$인 경우를 1으로 판별하면 error 라고 판단합니다.
    

