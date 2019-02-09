---
layout: post
title: t-SNE 개념과 사용법  
date: 2019-02-09 08:42:00
img: ml/concept/t-SNE/t-sne.png
categories: [ml-concept] 
tags: [ML, machine learning, 머신 러닝, t-SNE, PCA] # add tag
---

출처 : https://www.datacamp.com/community/tutorials/introduction-t-sne

+ 이번 글에서는 `t-SNE`에 대하여 알아보도록 하겠습니다.
    + Dimensionality Reduction
    + Principal Component Analysis (PCA)와 파이썬에서의 사용법
    + t-Distributed Stochastic Neighbor Embedding (t-SNE)와 파이썬에서의 사용법
    + PCA와 t-SNE 의 visualization 차이점
    + PCA와 t-SNE의 차이점 비교

<br><br>

## Dimensionality Reduction

+ 수많은 feature들을 가지고 있는 데이터셋을 이용하여 작업을 해보았다면, feature들 간의 관계를 파악하기가 어려운 것을 느꼈을 것입니다.
    + feature의 수가 너무 많으면 머신 러닝 모델의 성능을 저하시키곤 합니다.
    + feature의 갯수가 너무 많으면 overfit이 될 가능성이 있습니다. (PRML 책 참조)
+ 머신 러닝에서 dimensionality reduction(차원 축소)는 중요한 feature의 갯수는 남기고 불필요한 feature의 갯수를 줄이는 데 사용되곤 합니다.
+ 불필요한 feature의 갯수를 줄이는 방법으로 
    + 복잡한 feature들 간의 관계를 줄일 수도 있습니다.
    + 2D, 3D로 시각화 할 수도 있습니다.
    + overfit을 방지할 수도 있습니다.

+ Dimensionality Reduction은 다음과 같은 방법으로 할 수 있습니다.
    + `Feature Elimination`
        + Feature를 단순히 삭제하는 방법입니다. 이 방법은 간단하나 삭제된 feature들로 부터 어떠한 정보를 얻지는 못합니다.
    + `Feature Selection`
        + 통계적인 방법을 이용하여 feature들의 중요도에 rank를 정합니다.
        + 이 방법 또한 information loss가 발생할 수 있으며 동일한 문제를 푸는 다른 데이터셋에서는 다른 rank를 매길 수 있다는 문제가 있을 수 있습니다.
    + `Feature Extraction`
        + 새로운 독립적인 feature를 만드는 방법이 있습니다.
        + 새로 만들어진 feature는 기존에 존재하였던 독립적인 feature들의 결합으로 만들어 집니다.
        + 이 방법에는 linear한 방법과 non-linear한 방법들로 나뉘어 집니다.

<br><br>
    
## Principal Component Analysis (PCA)

+ PCA는 `Feature Extraction`의 방법이고 linear한 방법을 사용합니다.
+ PCA는 원본 데이터를 저차원으로 linear mapping 합니다. 이 방법으로 저차원에 표현되는 데이터의 variance가 최대화 됩니다.
+ 기본적인 방법은 **공분산 행렬에서 고유벡터를 계산**하는 것 입니다.
+ 가장 큰 고유값을 가지는 고유벡터를 principal component로 생각하고 새로운 feature를 생성하는 데 사용합니다.
+ 위 방법을 이용하여 PCA는 입력 받은 데이터 들의 feature를 결합합니다.
    + feature들을 결합할 때, 가장 중요하지 않은 feature들은 제거해가고 가장 중요한 feature들은 남깁니다.
    + 새로 생성된 feature들은 기존의 feature들과 독립적입니다. 즉 기존 feature들의 단순 선형 결합으로 만들어진 것은 아닙니다.  
+ PCA에 대한 자세한 방법은 [선형대수학 관련 기본 내용](https://gaussian37.github.io/math-la-linear-algebra-basic/) 또는 [PCA 개념 설명 글](https://gaussian37.github.io/machine-learning-concept-pca/)을 참조하시기 바랍니다.

<br><br>

## t-Distributed Stochastic Neighbor Embedding (t-SNE)

+ t-SNE는 **비선형적인 방법**의 차원 축소 방법이고 특히 고차원의 데이터 셋을 시각화하는 것에 성능이 좋습니다.
+ t-SNE는 다양한 분야에서 시각화 하는 데 사용되고 있습니다.
+ t-SNE 알고리즘은 고차원 공간에서의 점들의 유사성과 그에 해당하는 저차원 공간에서의 점들의 유사성을 계산합니다.
    + 점들의 유사도는 A를 중심으로 한 정규 분포에서 확률 밀도에 비례하여 이웃을 선택하면 포인트 A가 포인트 B를 이웃으로 선택한다는 조건부 확률로 계산됩니다.
+ 그리고 저 차원 공간에서 데이터 요소를 완벽하게 표현하기 위해 고차원 및 저 차원 공간에서 이러한 조건부 확률 (또는 유사점) 간의 차이를 최소화하려고 시도합니다.
+ 조건부 확률의 차이의 합을 최소화하기 위해 t-SNE는 gradient descent 방식을 사용하여 전체 데이터 포인트의 KL-divergence 합계를 최소화합니다.
    + `Kullback-Leibler divergence` 또는 `KL divergence`는 한 확률 분포가 두 번째 예상 확률 분포와 어떻게 다른지 측정하는 척도입니다.
+ 정리하면 `t-SNE`는 두가지 분포의 `KL divergence`를 최소화 합니다.
    + 입력 객체(고차원)들의 쌍으로 이루어진 유사성을 측정하는 분포
    + 저차원 점들의 쌍으로 유사성을 측정하는 분포  
+ 이러한 방식으로, t-SNE는 다차원 데이터를보다 낮은 차원 공간으로 매핑하고, 다수의 특징을 갖는 데이터 포인트의 유사성을 기반으로 점들의 클러스터를 식별함으로써 데이터에서 패턴을 발견합니다.
+ 하지만 `t-SNE` 과정이 끝나면 input feature를 확인하기가 어렵습니다. 그리고 t-SNE 결과만 가지고 무언가를 추론 하기는 어려움도 있습니다.
    + 따라서 `t-SNE`는 주로 시각화 툴로 사용 됩니다.

<br><br>

## t-SNE의 자세한 설명


## 파이썬을 이용한 t-SNE 구현 방법

+ Fashion MNIST 데이터에 `t-SNE`를 적용하고 결과를 시각화 해보겠습니다.
    + [Fashion MNIST 데이터](https://github.com/zalandoresearch/fashion-mnist)
    + Fashion-MNIST 데이터 세트는 카테고리 당 7,000 개 이미지, 10 개 카테고리, 70,000 개 패션 제품에 대한 28x28 grayscale 이미지 입니다.
    + training 세트에는 60,000 개의 이미지가 있고 test 세트에는 10,000 개의 이미지가 있습니다. 
    + MNIST와 마찬가지로 Fashion-MNIST는 10 개의 레이블로 이루어져 있지만, 숫자 대신 샌들, 셔츠, 바지 등과 같은 패션 액세서리의 10 가지 레이블이 있습니다.
        + 0 T-shirt/top
        + 1 Trouser
        + 2 Pullover
        + 3 Dress
        + 4 Coat
        + 5 Sandal
        + 6 Shirt
        + 7 Sneaker
        + 8 Bag
        + 9 Ankle boot
    
+ 또한 동일한 데이터 세트에서 `PCA`의 출력을 시각화하고 `t-SNE`와 비교해 보도록 하겠습니다.
+ 먼저 fashion MNIST 데이터를 다운 받아 보도록 하겠습니다.

<br>

```python
import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

<br>

+ 입력 받은 `X_train` 데이터의 shape을 확인해 보겠습니다.

<br>

```python
>> print(X_train.shape)

(60000, 28, 28)
```

<br>

+ 현재 각 이미지가 (28, 28)의 크기로 되어 있습니다. 이미지들을 28 x 28 = 784 크기의 벡터로 만들겠습니다.
    + 각 이미지가 열벡터로 된 행렬을 만들어야 처리하기 용이합니다.
    
```python
X_train = X_train.reshape(60000, 784)
```

 
    
    


  


