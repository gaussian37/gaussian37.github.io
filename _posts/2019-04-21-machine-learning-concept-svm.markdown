---
layout: post
title: Support Vector Machine (Andrew Ng)
date: 2019-04-21 00:00:00
img: ml/concept/machineLearning.jpg
categories: [ml-concept] 
tags: [machine learning, svm, suport vector machine] # add tag
---

출처 : Andrew Ng 강의

+ 이 글에서는 SVM(Support Vector Machine)에 대하여 알아보겠습니다.

### Optimization objective

+ SVM은 기본적으로 Supervised Learning의 한 방법입니다.
+ SVM은 성능도 좋을 뿐만 아니라 복잡한 non-linear function에서 학습할 때에도 좋습니다.

<img src="../assets/img/ml/concept/svm/1.PNG" alt="Drawing" style="width: 600px;"/>

+ SVM을 시작하기에 앞서서 logistic regression부터 접근해서 좀 더 확장하는 방법으로 SVM에 대하여 배워보도록 하겠습니다.
+ 위 슬라이드는 기초적인 logistic regression 식을 나타냅니다. 
    + 일반적으로 logistic regression에서의 threshold 값은 0.5로 설정하게 됩니다.
    + 따라서 $$ z = \theta^{T}x $$의 값이 0보다 크게 되면 prediction은 1이 되고 0보다 작게 되면 prediction은 0이 되게 됩니다.
       
<img src="../assets/img/ml/concept/svm/2.PNG" alt="Drawing" style="width: 600px;"/>

+ logistic regression의 cost function을 살펴보면 위 슬라이드와 같은 식이 도출됩니다.
+ logistic regression은 label(y)의 값이 0 또는 1인데 label의 값이 0 또는 1임에 따라서 cost function의 term 하나가 0이 되게 됩니다.
+ 먼저 y = 1일 때를 살펴보겠습니다. 왼쪽 아래 슬라이드와 같이 cost function을 표현할 수 있습니다.
    + 이 때 cost function을 접하는 직선을 2개 그어서 근사한 것 처럼 표현해 보겠습니다.
    + 예를 들어 임계값이 $$ z = 1 $$로 잡으면 $$ z $$ 값이 점점 증가하다가 $$ z = 1 $$인 순간부터 cost 값은 0이 되는 구조입니다.
+ 반대로 y = 0일 때를 살펴보겠습니다. 오른쪽 아래 슬라이드와 같이 cost function을 나타낼 수 있습니다.
    + 이 때에도 직선으로 cost function에 접하는 근사한 그래프를 그려볼 수 있습니다.

<img src="../assets/img/ml/concept/svm/3.PNG" alt="Drawing" style="width: 600px;"/>

+ 앞에서 정의한 각 데이터의 cost function을 모든 데이터에 대하여 정의해 보도록 하겠습니다.
+ 위 슬라이드의 식을 보면 $$ \sum $$ 이 추가 된것을 볼 수 있는데, cost를 모두 합하여 전체 cost를 구하는 것입니다.
+ 위 식이 일반적으로 사용하는 식과 다른점은 식을 간편하게 치환하기 위하여 가장 바깥쪽으로 빼두었던 - 부호를 안쪽으로 옮긴 것이 있습니다.
+ 이전 슬라이드에서 정의한 바와 같이 직선 2개로 근사한 cost 함수의 그래프를 아래와 같이 간략하게 표현할 수 있습니다.
    + 　$$ cost_{1}(\theta^{T}x^{(i)}) $$ if $$ y = 1 $$
    + 　$$ cost_{0}(\theta^{T}x^{(i)}) $$ if $$ y = 0 $$
+ 또 간략화 할수 있는 것은 식에 곱해진 $$ \frac{1}{m} $$ 식입니다. 이 식은 cost function과 regularization에 모두 곱해져 있습니다.
+ 이 식을 생략할 수 있는 이유는 위의 슬라이드에서 설명을 하고 있는데, 최소화 하는 optimization 문제에서 곱해진 값은 optimal 값을 찾는 데 영향을 끼치지 않기 때문입니다.
+ 마지막으로 regularization에 곱해진 $$ \lambda $$ 또한 없애주기 위해 두 term에  $$ C = \frac{1}{\lambda} $$를 곱해줍니다.
    + 이 또한 optimization 문제를 푸는 데 영향을 주지 않습니다.

<img src="../assets/img/ml/concept/svm/4.PNG" alt="Drawing" style="width: 600px;"/>

+ 식을 정리하면 위 슬라이드 처럼 간략하게 표현할 수 있습니다.
+ 그러면 다시 이 식을 통하여 prediction을 할 때 $$ \theta^{T}x^{(i)} $$ 의 값이 0 보다 크면 1로 판단하고 0보다 작으면 0으로 판단하도록 합니다.  

### Large Margin Intuition

<img src="../assets/img/ml/concept/svm/5.PNG" alt="Drawing" style="width: 600px;"/>

+ 앞에서 정리한 SVM의 cost function은 위 슬라이드의 식과 같습니다.
+ 이 식과 logistic regression의 식과의 차이점은 Support Vector Machine의 식이 logistic regression의 곡선을 직선으로 근사한 것이라는 점입니다.
    + 여기서 발생하는 차이점은 logistic regression에서는 $$ z = \theta^{T}x $$의 값 비교 기준이 0이었으나(0 이상 또는 0 이하) SVM에서는 1과 -1로 바뀌었다는 점입니다.
+ 그리고 앞에서 식을 정리하면서 새로 생긴 term인 $$ C $$의 크기는 어떤 영향을 미칠까요?

<img src="../assets/img/ml/concept/svm/6.PNG" alt="Drawing" style="width: 600px;"/>

+ 위 슬라이드 식의 목적은 cost를 최소한으로 만드는 것 입니다. 즉, 위 식의 값을 최대한 0에 가깝게 줄이는 것입니다.
+ 만약 C의 값이 굉장히 크다면 $$ \sum $$에 해당하는 값이 0에 가까워 져야 optimization에 가까워집니다. 
    + 즉 C의 값이 커질수록 학습할 때 $$ \sum $$에 해당하는 값이 더욱 0에 가까워지도록 학습하게 됩니다.

<img src="../assets/img/ml/concept/svm/7.PNG" alt="Drawing" style="width: 600px;"/>

+ 이번에는 공간 상에서 SVM이 하는 역할에 대하여 알아보려고 합니다.
+ 만약 위 슬라이드의 2차원 공간과 같이 데이터가 분포되어 있습니다. 두 데이터를 나누려고 하는데 어떻게 나누는 것이 좋을까요?
+ 두 데이터 부류를 나누는 데에는 상당히 많은 방법이 있습니다. 
    + 위 예시를 봤을 때, 자주색 또는 연두색 보다 검은색 선이 더 잘 나눈거 같다고 생각은 드는데요..
    
<img src="../assets/img/ml/concept/svm/8.PNG" alt="Drawing" style="width: 600px;"/>

+ 그 이유는 검은색 선을 보면 검은색 선에 가장 가까운 데이터와의 거리를 `margin` 이라고 하는데 검은색 선의 margin이 다른색 선의 margin보다 더 크기 때문입니다.
   
<img src="../assets/img/ml/concept/svm/9.PNG" alt="Drawing" style="width: 600px;"/>

+ 앞에서 설명한 C 상수의 크기는 decision boundary에 영향을 미치게 됩니다.
+ $$ C $$는 정확히 `regularization` 이라고 볼 수 있습니다. 왜냐하면 $$ C = \frac{1}{\lambda} $$ 이기 때문입니다.
    + 즉 $$ \lambda $$의 값이 너무 커지게 되면($$ C $$의 값이 너무 작아지게 되면) `high bais` 문제로 인하여 underfitting이 발생할 수 있습니다.
    + 반대로 $$ \lambda $$의 값이 너무 작아지게 되면($$ C $$의 값이 커지게 되면) `high variance` 문제로 overfitting 문제가 발생할 수 있습니다.
+      

<div style="text-align: center;">
<iframe src="https://www.youtube.com/embed/0-DL_nSa_ew" frameborder="0" allowfullscreen="true" width="600px" height="400px" align="middle"> </iframe>
</div>
    