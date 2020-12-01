---
layout: post
title: sklearn 코드 snippets
date: 2019-01-11 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [python, machine learning, ml, sklearn, scikit learn] # add tag
---

<br>

- `sklearn`은 가장 유명한 머신 러닝 용도의 파이썬 패키지 입니다. 이 글을 통하여 `sklearn`을 사용하면 자주 사용하거나 필요한 코드들을 정리하겠습니다.

<br>

## **목차**

<br>

- ### Dictionary에서 Feature 가져오기
- ### sklearn의 Toy Data 불러오기
- ### sklearn의 학습한 모델 저장하기
- ### sklearn으로 Perceptron 실습하기
- ### Pandas Categorical 데이터 전처리
- ### sklearn을 이용한 데이터 분할
- ### Outlier 검출 후 제거하기

<br>

## **Dictionary에서 Feature 가져오기**

<br>

<br>

## **sklearn의 Toy Data 불러오기**

<br>

<br>

## **sklearn의 학습한 모델 저장하기**

<br>

<br>

## **sklearn으로 Perceptron 실습하기**

<br>

<br>

## **Pandas Categorical 데이터 전처리**

<br>

<br>

## **sklearn을 이용한 데이터 분할**

<br>






<br>

## **Outlier 검출 후 제거하기**

<br>

- 데이터를 분석할 때 `Outlier`들은 먼저 제거하고 분석이 필요할 때가 있습니다. 아웃라이어를 제거할 때에 사용할 수 있는 방법이 `Anomaly Detection`을 한 다음 detection된 데이터들은 제거하는 것입니다.
- 간단하게 설명하면, 정규 분포를 이용하여 데이터의 확률을 구하고 너무 확률이 낮은 것은 outlier로 간주하여 삭제하는 것입니다.
- 이번 글에서는 랜덤 데이터를 만들어 보고 지우는 실습을 해보겠습니다.

<br>

```python
import numpy as np

# 정규 분포를 이용하여 데이터 분포에 타원을 그립니다. 타원에서 벗어날수록 outlier입니다.
from sklearn.covariance import EllipticEnvelope
# 랜덤 데이터를 생성하는데 사용됩니다.
from sklearn.datasets import make_blobs

# 랜덤 데이터를 생성합니다.
X, _ = make_blobs(n_samples = 10,
                  n_features = 2,
                  centers = 1,
                  random_state = 1)

# 이상치를 만듭니다.
X[0,0] = 10000
X[0,1] = 10000

# EllipticEnvelope 을 이용하여 outlier를 검출하기 위한 객체를 생성합니다.
# contamination의 비율을 기준으로 비율보다 낮은 값을 검출합니다.
outlier_detector = EllipticEnvelope(contamination=.1)

# EllipticEnvelope 객체를 생성한 데이터에 맞게 학습합니다.
outlier_detector.fit(X)

# outlier를 검출 합니다.
# +1 이면 boundary 안에 들어온 값으로 정상 데이터 입니다.
# -1 이면 outlier로 간주 합니다.
outlier_detector.predict(X)
# array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
```

<br>