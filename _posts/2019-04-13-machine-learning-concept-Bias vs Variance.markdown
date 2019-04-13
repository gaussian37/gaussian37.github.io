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

<br><br>

### Learning Curves

+ 이번에는 `training set`의 크기에 따른 bias와 variance의 변화를 살펴보도록 하겠습니다.

<img src="../assets/img/ml/concept/bias-variance/7.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 위 슬라이드를 보면 `training set`의 크기에 따라서 모델의 `generalization` 성능이 상승되는 것을 알 수 있습니다.
+ 슬라이드의 오른쪽을 보면 데이터가 1개 있을 때부터 점점 증가하여 데이터가 6개 있을 때 까지 그래프의 모양이 변형되는 것을 볼 수 있습니다.
    + 즉, 데이터가 많아질수록 파라미터가 데이터에 맞춰 학습이 되기 때문에 점점 더 데이터의 데이터 모집단의 분포에 가까워 지게 됩니다.
+ 슬라이드 왼쪽 하단의 training set과 error의 그래프를 보면 학습 데이터 셋의 크기가 매우 작은 경우는 error가 상당히 작은 것을 알 수 있습니다.
    + 왜냐하면 데이터가 너무 작기 때문에 error가 발생할 데이터의 수가 작기 때문입니다.
+ 데이터가 점점 증가할수록 training error는 점점 증가하다가 정체됩니다. 이 경향이 일반적인 training error의 변화 과정입니다.
+ 반면 validation error는 학습 데이터가 매우 작을 때에는 상당히 큽니다. 왜냐하면 아주 조금의 데이터로 모델을 학습하였기 때문에 `generalization` 성능이 매우 떨어지기 때문입니다.
+ 그러다가 학습 데이터의 갯수가 늘어날수록 `generalization`성능이 커지게 되어 training error보단 크지만 유사한 수준으로 validation error가 감소하는 것을 볼 수 있습니다.

<br>

+ 그러면 이러한 training set의 크기와 bias/variance 의 관계에 대해서 알아보도록 하겠습니다.

<img src="../assets/img/ml/concept/bias-variance/8.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 `bias`문제에 대하여 살펴보겠습니다. `bias`문제는 기본적으로 모델의 복잡도가 낮아서 표현력이 안좋기 때문에 발생합니다.
+ 위 슬라이드처럼 모델이 단순 선형이라고 가정하면 데이터 분포를 적합하게 표현할 수가 없습니다.
+ 이런 경우에 training data 크기를 늘리더라도 bias 문제를 해결하기는 어렵습니다.

<img src="../assets/img/ml/concept/bias-variance/9.PNG" alt="Drawing" style="width: 600px;"/>

+ 반면, `variance`문제는 기본적으로 학습 데이터에 비하여 너무 모델의 복잡도가 높아서 모델이 너무 과하게 학습한 문제로 인해 발생합니다.
+ 즉, 학습 데이터의 크기를 늘려서 validation/test error를 줄인다는 것은 `variance`문제에 상당히 적합합니다.
+ 표현력이 너무 좋아서 문제가 된 모델이 더 많은 학습데이터를 통하여 데이터 분포를 정확하게 표현하게 되므로 `generalization` 성능도 올라가게 됩니다.
    + 즉 validation/test error가 줄어들게 됩니다.
    
<br><br>

## bias/variance 문제 정리

+ 앞에서 배운 `bias`와 `variance`관련 내용을 정리해 보겠습니다.

<img src="../assets/img/ml/concept/bias-variance/10.PNG" alt="Drawing" style="width: 600px;"/>

+ 위 슬라이드 내용을 정리하면
    + `high variance` 문제를 해결하기 위해서
        + training data의 갯수를 늘린다.
        + feature의 갯수를 줄인다.
        + regularization parameter $$ \lambda $$의 크기를 증가시킨다.
    + `high bias` 문제를 해결하기 위해서
        + feature의 갯수를 늘인다.
        + polynomial feature를 추가해 본다. 좀 더 복잡한 모델을 사용해 본다.
        + regularization parameter $$ \lambda $$의 크기를 줄여본다.

<img src="../assets/img/ml/concept/bias-variance/11.PNG" alt="Drawing" style="width: 600px;"/>

+ Neural Network에서 layer의 갯수와 parameter의 갯수는 비례합니다. 
+ layer의 갯수가 작으면 계산 비용은 작지만 `high bias 문제`에 빠질 가능성이 있습니다.
    + 이 떄에는 layer를 추가하는 것이 좋습니다.
+ 반면 layer의 갯수가 너무 많으면 계산 비용도 많이 들고 `high variance 문제`에 빠질 가능성이 있습니다.
    + 이 때에는 데이터의 갯수를 늘리거나 regularization을 추가하거나 layer의 갯수를 줄여보면 좋습니다.