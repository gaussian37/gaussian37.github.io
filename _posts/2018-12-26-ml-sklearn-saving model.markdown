---
layout: post
title: 사이킷런(scikit-learn)으로 학습한 모델 저장하기 
date: 2018-12-26 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [python, machine learning, ml, sklearn, scikit learn, save, model] # add tag
---

sklearn을 이용하여 model을 학습한 후 학습한 결과를 저장하는 방법에 대하여 알아보겠습니다.

`pickle` 형태로 모델을 저장할 것이고 저장할 때에는 sklearn의 `joblib`을 사용할 것입니다.
`pickle`은 파이썬에서 지원하는 serializer 형태의 저장 방식입니다.
참고로 JSON 같은 경우는 언어에 상관없이 범용적으로 사용할 수 있는 seriazlier 형태이지만 `pickle`은 파이썬에서만 사용가능 하되
지원되는 데이터 타입이 JSON 보다 많이 있습니다.

자 그러면 코드를 통하여 알아보겠습니다. 예제는 iris 데이터를 사용해 보겠습니다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import pickle
from sklearn.externals import joblib
```

<br>

다음으로 데이터를 로드해 보겠습니다.

```python
# Load the iris data
iris = datasets.load_iris()

# Create a matrix, X, of features and a vector, y.
X, y = iris.data, iris.target
```

<br>

다음으로 간단한 Logistric Regression을 적용해 보겠습니다.

```python
clf = LogisticRegression(random_state=0)
clf.fit(X, y)  
```

<br>

여기서, 모델을 저장해 보겠습니다. 여기서는 변수에 먼저 저장하는 방법을 소개하고, 아래에서 파일에 저장하는 방법을 소개해 드리겠습니다.

```python
saved_model = pickle.dumps(clf)
```

<br>

`saved_model` 을 실행해 보면 이상한 문자열이 나오는데 그것이 serializer 형태로 저장된 것이라고 볼 수 있습니다.

pickle로 저장한 모델을 불러와 보겠습니다.

```python
# Load the pickled model
clf_from_pickle = pickle.loads(saved_model)

# Use the loaded pickled model to make predictions
clf_from_pickle.predict(X)
>> array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

<br>

위에서 해 본 내용은 학습한 모델을 pickle 형태로 변수에 저장한 것이고, 이제 파일에 저장을 해보겠습니다.

```python
joblib.dump(clf, 'filename.pkl') 

>>

['filename.pkl',
 'filename.pkl_01.npy',
 'filename.pkl_02.npy',
 'filename.pkl_03.npy',
 'filename.pkl_04.npy']

```

<br>

이제 저장한 파일을 불러와서 predict 해보겠습니다.

```python
clf_from_joblib = joblib.load('filename.pkl') 
clf_from_joblib.predict(X)

>>> array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

```

<br>

도움이 되셨다면 광고 클릭 한번이 제게 큰 도움이 됩니다. 꾸벅.