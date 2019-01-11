---
layout: post
title: Outlier 검출 후 제거하기 
date: 2019-01-11 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [python, machine learning, ml, sklearn, scikit learn, outlier] # add tag
---

데이터를 분석할 때 `Outlier`들은 먼저 제거하고 분석이 필요할 때가 있습니다.
아웃라이어를 제거할 때에 사용할 수 있는 방법이 `Anomaly Detection`을 한 다음 detection된 데이터들은 제거하는 것입니다.

간단하게 설명하면, 정규 분포를 이용하여 데이터의 확률을 구하고 너무 확률이 낮은 것은 outlier로 간주하여 삭제하는 것입니다.

랜덤 데이터를 만들어 보고 지우는 실습을 해보겠습니다.

```python
import numpy as np

# 정규 분포를 이용하여 데이터 분포에 타원을 그립니다. 타원에서 벗어날수록 outlier입니다.
from sklearn.covariance import EllipticEnvelope
# 랜덤 데이터를 생성하는데 사용됩니다.
from sklearn.datasets import make_blobs
```

<br>

+ 랜덤 데이터를 생성합니다.

```python
X, _ = make_blobs(n_samples = 10,
                  n_features = 2,
                  centers = 1,
                  random_state = 1)
```

<br>

+ 이상치를 만듭니다.

```python
X[0,0] = 10000
X[0,1] = 10000
```

<br>

+ EllipticEnvelope 을 이용하여 outlier를 검출하기 위한 객체를 생성합니다.
+ contamination의 비율을 기준으로 비율보다 낮은 값을 검출합니다.    

```python
outlier_detector = EllipticEnvelope(contamination=.1)
```

<br>

+ EllipticEnvelope 객체를 생성한 데이터에 맞게 학습합니다.

```python
outlier_detector.fit(X)
```

<br>

+ outlier를 검출 합니다.
+ +1 이면 boundary 안에 들어온 값으로 정상 데이터 입니다.
+ -1 이면 outlier로 간주 합니다.

```python
>> outlier_detector.predict(X)

array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])

```

<br>

도움이 되셨다면 광고 클릭 한번이 저에게 큰 도움이 되겠습니다!