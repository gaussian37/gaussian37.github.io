---
layout: post
title: Dictionary에서 Feature 가져오기 
date: 2018-12-23 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [python, machine learning, ml, sklearn, scikit learn, feature, dictionary] # add tag
---

Dictionary의 형태의 데이터를 학습할 때 Table형태로 바꿀 필요가 있습니다.
이것을 알아서 해줄 수 있는 것이 있을까요?

물론 있습니다. sklearn을 이용하면 할 수 있습니다.

+ 필요한 라이브러리를 import 합니다.

```python
from sklearn.feature_extraction import DictVectorizer
```

<br>

+ Dictionary를 만들어 보겠습니다.

```python
men = [{'name': 'Kim', 'age': 20},
         {'name': 'Lee', 'age': 40},
         {'name': 'Park', 'age': 50}]
```

<br>

+ Dictionary to Vectorize 객체를 생성합니다.

```python
vec = DictVectorizer()
```

<br>

+ Dictionary를 Vector 타입으로 바꿔 보겠습니다.

```python
vec.fit_transform(men).toarray()
>>>array([[20.,  1.,  0.,  0.],
       [40.,  0.,  1.,  0.],
       [50.,  0.,  0.,  1.]])

```

<br>

오 뭔가 바뀐건 같지요? 나이관련 field도 있는 것 같고 one-hot vector 타입도 생긴것 같습니다.
그러면 각 field가 어떤 의미를 가지는지 알아보겠습니다.

```python
vec.get_feature_names()
>>> ['age', 'name=Kim', 'name=Lee', 'name=Park']

```

<br>

첫 field는 age를 나타내고 나머지는 각 사람의 이름을 one-hot으로 나타낸것을 확인할 수 있습니다.

전체 코드는 다음과 같습니다.

```python
from sklearn.feature_extraction import DictVectorizer

men = [{'name': 'Kim', 'age': 20},
         {'name': 'Lee', 'age': 40},
         {'name': 'Park', 'age': 50}]

vec = DictVectorizer()
vec.fit_transform(men).toarray()

vec.get_feature_names()
```

<br>

도움이 되셨다면 광고 한번 클릭 부탁 드립니다. 꾸벅.