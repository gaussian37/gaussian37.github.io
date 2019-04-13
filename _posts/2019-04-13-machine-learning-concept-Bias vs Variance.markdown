---
layout: post
title: Bias Vs. Variance (Andrew Ng)
date: 2019-04-13 00:00:00
img: ml/concept/machineLearning.jpg
categories: [ml-concept] 
tags: [machine learning, pca] # add tag
---

출처 : Andrew Ng 강의

+ 이번 글에서는 `Bias`와 `Variance` 문제의 정의에 대하여 다루어 보도록 하겠습니다.

### Bias Vs. Variance

<img src="../assets/img/ml/concept/bias-variance/1.PNG" alt="Drawing" style="width: 600px;"/>

+ Bias와 Variance 문제의 정의를 살펴보면 bias 문제는 데이터의 분포에 비하여 모델이 너무 간단한 경우 underfit이 발생한 경우에 발생합니다.
+ Variance 문제는 모델의 복잡도가 데이터 분포보다 커서 데이터를 overfitting 시키는 문제를 말합니다.
 
<img src="../assets/img/ml/concept/bias-variance/2.PNG" alt="Drawing" style="width: 600px;"/>

+ 모델의 degree를 높이면 식의 표현력이 더 증가해서 성능이 증가하지만 데이터 분포에 비하여 너무 표현력이 증가하면 overfitting이 발생할 수 있음을 확인하였습니다.
+ 위 슬라이드의 그래프는 degree의 증가에 따라서 error가 어떻게 바뀌는 지 보여줍니다.
+ 예를 들어 $$ d = 1 $$ 일 때, 식이 너무 단순해서 데이터를 잘 표현하지 못하는 underfitting 문제가 발생합니다.
    + 이 때에는 training error와 validation error 가 둘 다 높습니다.
    + 앞에서 설명한 바와 같이 이 문제가 `bias problem`입니다.
+ degree를 높이다 보면 training error와 validation error가 둘 다 감소하는 구간이 있습니다. 
    + error가 감소하는 마지막 구간이 가장 적합한 degree 입니다.
+ degree를 계속 증가시키다 보면 training error는 감소하지만 validation error는 다시 증가하는 현상이 발생하는데 이 때가 overfitting이 발생한 구간 입니다.
    + 이 문제가 `variation problem`입니다. 

<img src="../assets/img/ml/concept/bias-variance/3.PNG" alt="Drawing" style="width: 600px;"/>

+ 앞에서 설명한 내용을 다시 정리하면 위와 같습니다.
+ 다시 한번 확인해볼 점은 Bias와 Variance 문제가 언제 발생하고 이 때의 train error와 validation error의 관계를 파악하는 것입니다.

<br><br>

### Regularization and Bias/Variance

+ 이번에는 앞에서 알아본 `bias`와 `variance`를 `regularization`과 연관하여 알아보도록 하겠습니다.

<img src="../assets/img/ml/concept/bias-variance/4.PNG" alt="Drawing" style="width: 600px;"/>

+ 만약 위 슬라이드와 같이 `polynomial` 모델이 있고 `regularization`을 적용했다고 가정해 보겠습니다.
+ 이 때, $$ \lambda $$ 값에 따라서 bias/variance 문제는 어떻게 되는지 살펴보겠습니다.
+ 먼저 `regularization` 텀을 보면 $$ j = 1, ..., m $$의 범위를 가집니다.
+ 즉, $$ \lambda $$ 가 아주 큰 값을 가지게 되면 $$ \theta_{1}, ..., \theta_{m} $$에 대해서는 학습이 안되게 됩니다.
    + 반면 $$ \theta_{0} $$의 값만 남게 되어 `High bias`의 그래프 처럼 수평선 그래프가 그려지게 됩니다.
+ 반대로 High variance의 상황을 보면 $$ \lambda = 0 $$ 으로 극단적으로 생각할 수 있습니다.
    + 이 상황은 regularization을 사용하지 않은 것으로 overfitting의 문제가 나타날 수 있습니다.
+ 따라서 우리의 목적은 적당한 bias와 variance를 가지도록 $$ \lambda $$ 값을 설정할 필요가 있습니다.
   
<img src="../assets/img/ml/concept/bias-variance/5.PNG" alt="Drawing" style="width: 600px;"/>

+ 따라서 위 슬라이드와 같이 $$ \lambda $$를 정할 때, 0부터 시작해서 점점 더 크기를 올리면서 적용하는 것이 좋습니다.
+ 위 슬라이드에서 보면 각각의 $$ \lambda $$ 후보값들을 이용하여 validation error를 구하고 그 error가 최소인 것을 선택하는 방법을 사용하였습니다.
    + 예를 들면 위 슬라이드에서는  $$ \lambda =0.08 $$에서 validation error가 최소가 되므로 선택되었습니다.
     
<img src="../assets/img/ml/concept/bias-variance/6.PNG" alt="Drawing" style="width: 600px;"/>

+ regularization parameter의 변화에 따라서 train error와 validation error에 대하여 확인해 보면 위 슬라이드의 그래프와 같습니다.
+ train error의 경우 regularization의 크기가 커질수록 error의 크기가 더 커지게 됩니다.
    + cost function을 보면 알 수 있듯이 regularization term은 항상 양수 값이 더해지기 때문입니다.
+ 반면 validation error의 경우 regularization이 점점 커질수록 error가 줄다가 다시 커지게 됨을 알 수 있습니다.
    + regularization이 매우 작은 값에서 적당한 값으로 커지게 되면 overfitting 문제가 조금씩 해결되면서 validation error가 줄어들게 됩니다.
    + 하지만 적정 크기의 regularization parameter 보다 값이 커지게 되면 cost function 의 형태와 같이 항상 error에 양의 값이 더해지게 되므로 error 값이 증가하게 됩니다.
    + 즉, regularization parameter가 적당한 값까지 증가할 때에는 variance 문제가 해결되는 것이 error에 값이 더해지는 것 보다 효과가 있어서 error가 줄어듭니다.
    + 반면 최적점을 지날 만큼 parameter 값이 커지게 되면 bias 문제에 빠지게 되고 error값에 더해지는 regularization값도 커지게 되어 error가 증가하게 됩니다.
    
      
