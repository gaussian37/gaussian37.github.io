---
layout: post
title: Recommender System
date: 2019-03-12 00:00:00
img: ml/concept/machineLearning.jpg
categories: [ml-concept] 
tags: [ml, machine learning, recommender system, 추천 시스템] # add tag
---

+ 출처 : Andrew Ng 머신 러닝
+ 이번 글에서는 Recommender System에 대하여 알아보도록 하겠습니다.

+ <img src="../assets/img/ml/concept/Recommender-systems/movierating1.PNG" alt="Drawing" style="width: 400px;"/>
+ 위와 같이 영화 평점을 매긴 현황이 있다고 가정하겠습니다.
+ 이 때 사용할 notation은 다음과 같습니다.
    + 　$$ n_{u} $$ : 사람 수
    + 　$$ n_{m} $$ : 영화의 갯수
    + 　$$ r(i, j) $$ : 만약 $$ j $$번째 사람이 $$ i $$번째 영화를 평가 하였으면 1
    + 　$$ y^{(i,j)} $$ : $$ j $$번째 사람이 $$ i $$번째 영화에 매긴 평점 (오직 $$ r(i,j) = 1$$ 인 경우만 값이 존재함)
+ 위 도에서 ?는 아직 영화를 보지 않았거나 평점이 없는 경우입니다. 이 값을 `missing value`라고 하고 자동적으로 값을 채워줘야 합니다.
    + 자동으로 채우는 방법에 대해서는 아래 글에서 다루겠습니다.
    
<br>

### Content-based recommender systems

+ 먼저 다음과 같은 상황을 가정해 보도록 하겠습니다.

+ <img src="../assets/img/ml/concept/Recommender-systems/movierating2.PNG" alt="Drawing" style="width: 400px;"/>
+ 추가된 가장 오른쪽 2개의 열 $$ x_{1}, x_{2} $$는 `feature vector`입니다.
    + `feature vector`는 영화의 성격을 나타냅니다.
    + 일반적으로 `bias`에 해당하는 $$ x_{0} $$은 따로 표시 않았고 여기에서도 연산할 때에만 추가됩니다.
        + 　$$ x^{i} = [x_{0}^{i}, x_{1}^{i}, x_{2}^{i}]^{T} $$ 에서
        + 예를 들어 $$ x_{0} = 1 $$로 두면,        
        + $$ x^{1} = [1, 0.9, 0]^{T} $$이 됩니다.
    + 일반적으로 `bias`는 feature의 갯수에서 제외합니다.

<br>

+ <img src="../assets/img/ml/concept/Recommender-systems/1.PNG" alt="Drawing" style="width: 600px;"/>
+ 위 슬라이드에서 $$ j $$ 번째 사람은 파라미터 $$ \theta^{j} $$를 학습해야 합니다.
    + 이 때 파라미터의 차수는 $$ n + 1 $$이 됩니다.
+ 예를 들어 1번째 사람의 학습된 파라미터의 결과가 $$ [0, 5, 0] $$ 이라고 가정하겠습니다.
+ 1번째 사람이 아직 정하지 못한 점수의 영화는 3번째 영화이므로 3번째 영화의 feature를 가져오겠습니다.
    + 즉, $$ [1, 0.99, 0] $$이 됩니다.
+ 따라서 $$ (\theta^{1})^{T}x^{3} = 4.95 $$가 됩니다. 

<br>

+ <img src="../assets/img/ml/concept/Recommender-systems/2.PNG" alt="Drawing" style="width: 600px;"/>
+ 상세 용어에 대한 설명은 위 슬라이드와 같습니다.
    + 　$$ r(i,j), y^{(i,j)} $$는 각각 평점을 매겼는지 체크하는 값과 평점값에 해당합니다.
    + 　$$ \theta{j} $$는 $$ j $$번째 사람의 영화 취향에 관한 벡터에 해당합니다.
    + 　$$ x^{i} $$ 는 i번째 영화의 성격에 관한 **feature** 벡터에 해당합니다.
    + 따라서 $$ j $$번째 사람의 벡터와 $$ i $$ 번째 영화의 feature 벡터의 벡터곱을 하면 $$ j $$ 번째 사람의 $$ i $$ 번째 영화에 대한 평점을 예상할 수 있습니다.
    + 　$$ m^{j} $$는 $$ j $$ 번째 사람이 평점을 매긴 영화의 갯수 입니다.
+ 위 용어를 참조하여 $$ j $$ 번째 사람의 $$ i $$ 번째 영화에 대한 평점을 예측하는 것을 살펴보겠습니다.
    + 위 슬라이드에 있는 식은 `linear regression` 방법 입니다. `linear regression`의 방법을 이용하여 예측값과 정답의 오차를 줄여나가는 학습 방법을 취합니다.
    + 그리고 오차함수의 뒤쪽에 `regularization`도 붙어서 `overfitting`을 줄이는 방법도 적용해 보았습니다.
    + 위의 오차 함수 식에서 $$ m_{j} $$를 지운 것은 상수항으로 학습에 영향이 없기 때문입니다.
    +  　$$ \sum_{i:r(i,j) = 1} $$의 뜻은 $$ r(i,j) = 1 $$ 즉, 평점을 매겼을 때에만 계산 과정에 포함시킨다는 의미입니다.
    
<br>

+ <img src="../assets/img/ml/concept/Recommender-systems/3.PNG" alt="Drawing" style="width: 600px;"/>
+ 위 슬라이드의 첫번째 식은 특정 사람($$ j $$ 번째 사람)에 관하여 학습하는 것을 보여줍니다.
+ 모든 사람들을 대상으로 학습을 하려면 두번째 식을 이용하면 됩니다. 이때 내부 식을 보면 첫번째 식을 좀 더 확장한 것임을 알 수 있습니다.

<br>

+ <img src="../assets/img/ml/concept/Recommender-systems/4.png" alt="Drawing" style="width: 600px;"/>
+ 앞의 슬라이드에서 정의한 오차 함수를 `gradient descent` 방법으로 최적화 시킵니다.
    + 위 슬라이드의 식과 같이 $$ \frac{\partial}{\partial \theta_{k}^{(j)} } J(\theta^{(1)}, \cdots , \theta^{(n_{u})} )
    
<br><br>

### Collaborative filtering 
    
    
   
    


