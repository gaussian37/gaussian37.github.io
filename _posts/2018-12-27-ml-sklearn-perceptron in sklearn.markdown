---
layout: post
title: 사이킷런(scikit-learn)으로 Perceptron 실습하기
date: 2018-12-26 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [python, machine learning, ml, sklearn, scikit learn, perceptron] # add tag
---

Perceptron은 그냥 아주 간단한 신경망이라고 할 수 있습니다. 보통 신경망을 사용하는 머신러닝을 딥러닝이라고 하니,
사실상 사이킷 런을 쓸 필요는 없습니다. 그냥 한번 실습해보는 정도에 의의를 두면 좋을 것 같습니다.

사이킷 런에도 Multi-layer Perceptron을 적용할 수 있는 함수가 있습니다.
하지만 GPU를 사용하지 않고 CPU를 사용하다보니, 속도가 상당히 느리고 궂이 딥러닝을 쓰려면 케라스를 쓰면 좋겠지요?

이번시간에는 간단하게 Perceptron을 구현하는 정도로 해보려고 합니다.


필요한 라이브러리를 불러옵니다.

```python
# Load required libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
```

<br>

iris 데이터를 불러와서 실습해 보겠습니다.

```python
# Load the iris dataset
iris = datasets.load_iris()

# Create our X and y data
X = iris.data
y = iris.target
```

<br>

기본적으로 실시하는 data 분리를 먼저 하겠습니다. train 데이터와 test 데이터 두개로 분리하겠습니다.
이 때, 분리하는 비율은 train : test = 7 : 3 으로 하겠습니다.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

<br>

StandardScaler를 X_train 데이터가 mean = 0, variance = 1로 만들 수 있도록 학습을 합니다.
즉, 학습 데이터가 $$ (\mu, \sigma^{2}) = (0, 1^{2}) $$ 이 될 수 있도록 학습합니다.

```python
sc = StandardScaler()
sc.fit(X_train)
```

<br>

학습한 Scaler를 가지고 X_train과 X_test를 정규화 해줍니다.

```python
# Apply the scaler to the X training data
X_train_std = sc.transform(X_train)

# Apply the SAME scaler to the X test data
X_test_std = sc.transform(X_test)
```

<br>

자, 이제 Perceptron을 가지고 학습해 보겠습니다.

```python
# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)

# Train the perceptron
ppn.fit(X_train_std, y_train)
```

<br>

이제, 학습 완료한 데이터를 가지고 predict를 해보겠습니다.

```python
y_pred = ppn.predict(X_test_std)
```

<br>

`y_pred` 와 `y_test` 비교를 통하여 정답률을 알아보겠습니다.

```python
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
>> Accuracy: 0.87

``` 

<br>

아주 간단한 코드이지만 생각보다 정답률이 높게 나왔지요? 신경망을 이용하면 간단하지만 효과적인 코드를 짤 수 있습니다.

도움이 되셨으면 광고 클릭이 큰 도움이 될 것 같습니다. 꾸벅
