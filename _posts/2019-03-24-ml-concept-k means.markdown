---
layout: post
title: Clustering with K-Means Algorithm
date: 2019-03-024 00:00:00
img: ml/concept/machineLearning.jpg
categories: [ml-concept] 
tags: [ml, machine learning, k means, k-means, clustering] # add tag
---

+ 출처 : Andrew Ng 머신러닝
+ 이번 글에서는 `K-means` 알고리즘을 이용한 `Clustering` 방법론에 대하여 알아보도록 하겠습니다.

+ 먼저 `Unsupervised Learning`에 대한 개념을 알아본 뒤 `K-means` 알고리즘에 대하여 알아보도록 하겠습니다.

### Unsupervised Learning

<img src="../assets/img/ml/concept/k-means/unsupervised-learning.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 `unsupervised learning`에 대하여 간략하게 알아보도록 하겠습니다.
+ `unsupervised learning`에서는 Training Set에 `label` 데이터가 없습니다. 
+ 여기서 우리가 할 일은 라벨이 없는 트레이닝 데이터를 알고리즘을 이용하여 데이터가 가지고 있는 어떤 구조를 찾는 것입니다.
    + 위 슬라이드에서 보면 두 개의 분리된 클러스터 구조를 가지고 있음을 알 수 있습니다.
    + 위와 같이 두 개의 클러스터를 만들어 내는 것을 `Clustering algorithm` 이라고 합니다. 
    
<br><br>

### K-means Algorithm

+ `K means` 알고리즘은 주어진 `unlabeled` 데이터셋에서 유사한 데이터 성격을 가지는 데이터 부분 집합들을 자동적으로 만들어 내는 데 사용됩니다.
    + `K means` 알고리즘은 가장 많이 그리고 자주 사용되는 clustering algorithm 중의 하나입니다.

+ K-means 알고리즘에서는 2가지 스텝을 가집니다.
    + 첫번째, `cluster assignment step` 입니다. 이 단계에서는 K 개의 클러스터를 할당하는 작업을 합니다.
    + 두번째, `move centroid step` 입니다. 이 단계에서는 각 클러스터의 중심을 이동합니다.

<img src="../assets/img/ml/concept/k-means/1.PNG" alt="Drawing" style="width: 600px;"/>
    
+ 첫번째 작업으로는 K개의 클러스터 중심점을 할당해야 합니다. 위 슬라이드에서는 K=2라고 가정해 보겠습니다.
+ 즉, cluster centroid는 2개가 되고 그 위치는 random으로 할당합니다.

<img src="../assets/img/ml/concept/k-means/2.PNG" alt="Drawing" style="width: 600px;"/>

+ centroid를 정하였다면 centroid와 가까운 점을 기준으로 clustering을 합니다.
+ 위의 슬라이드를 보면 centroid와 가까운 점을 기준으로 빨간색 또는 파란색으로 clustering을 하였습니다.
    + 이 작업을 `cluster assignment step` 이라고 합니다.

<img src="../assets/img/ml/concept/k-means/3.PNG" alt="Drawing" style="width: 600px;"/>

+ 그 다음으로 같은 cluster 내의 데이터들의 평균 위치를 구합니다. 그리고 이 평균 위치 값을 새로운 centroid로 지정합니다.
+ 그러면 위의 슬라이드와 같이 centroid가 움직이는 것을 알 수 있습니다. 

<img src="../assets/img/ml/concept/k-means/4.PNG" alt="Drawing" style="width: 600px;"/>

+ 이제는 앞에서 한 두 가지 스텝인 `cluster assignment`와 `move centroid`를 다시 반복해보겠습니다.
+ 먼저 `cluster assignment`를 해보겠습니다.
+ 위 슬라이드를 이전 슬라이드와 비교해 보면 데이터들이 재 배치되었습니다. 즉 centroid가 새로운 값으로 변경됨에 따라 데이터 중 일부가 cluster가 바뀌게 된 것입니다. 

<img src="../assets/img/ml/concept/k-means/5.PNG" alt="Drawing" style="width: 600px;"/>

+ 그 다음으로 `move centroid`를 해보면 centorid가 변경된 것을 볼 수 있습니다.

<img src="../assets/img/ml/concept/k-means/6.PNG" alt="Drawing" style="width: 600px;"/>

+ 이 작업을 계속 반복해 보면, 위 슬라이드와 같은 clustering을 얻을 수 있습니다.
+ 위와 같은 clustering을 얻기 위해서는 알고리즘 종료 조건이 필요합니다. 그 종료 조건은 클러스터를 구성하는 데이터가 더 이상 바뀌지 않고 수렴하게 되는 상태를 뜻합니다.
    + 위 알고리즘 전체를 보면 K-means 알고리즘은 iterative 알고리즘에 속하게 되고 알고리즘을 종료하기 위한 조건은 더 이상 상태 변화가 없을 때가 됩니다.
    
<br>

+ K-means 알고리즘을 정리해 보도록 하겠습니다.

<img src="../assets/img/ml/concept/k-means/7.PNG" alt="Drawing" style="width: 600px;"/>

+ 먼저 입력 값은 두 개 입니다.
    + 몇 개의 cluster로 나눌 지에 해당하는 `K` 입니다.
    + 그리고 Training data set이 필요합니다.
        + 이 때, 각각의 데이터는 n차원으로 표현할 수 있습니다.
    

