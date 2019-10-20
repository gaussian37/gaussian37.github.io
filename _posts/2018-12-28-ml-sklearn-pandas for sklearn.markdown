---
layout: post
title: Pandas Categorical 데이터 전처리
date: 2018-12-26 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [python, machine learning, ml, sklearn, scikit learn, pandas, categorical, preprocessing] # add tag
---

이번 글에서는 Pandas 의 Categorical 타입의 데이터를 sklearn에서 사용하기 위해 전처리 하는 방법에 대해서 알아보겠습니다.

```python
from sklearn import preprocessing
import pandas as pd
```

<br>

먼저 아래와 같은 데이터가 있다고 가정합시다.

```python
raw_data = {'id': [1, 1, 1, 2, 2],
        'p1': [1, 2, 3, 1, 2],
        'p2': [0, 1, 0, 1, 0],
        'p3': ['strong', 'weak', 'normal', 'weak', 'strong']}
df = pd.DataFrame(raw_data, columns = ['id', 'p1', 'p2', 'p3'])
```

<br>

df를 살펴보면 다음과 같습니다.

```python
>> df.head()

   id  p1  p2      p3
0   1   1   0  strong
1   1   2   1    weak
2   1   3   0  normal
3   2   1   1    weak
4   2   2   0  strong

```

<br>

이제, `LabelEncoder`를 만들고 `LabelEncoder`의 내용을 학습하도록 하겠습니다.

```python
le = preprocessing.LabelEncoder()
le.fit(df['p3'])
```

<br>

`LabelEncoder`의 학습한 내용을 보면 p3 열에서 가지고 있는 항목들을 가지고 있습니다.

```python
>> list(le.classes_)

['normal', 'strong', 'weak']

```

<br>

`p3` 열에서 가지고 있는 unique한 값은 ** 'normal', 'strong', 'weak' ** 3개 이므로 각각 숫자로 0, 1, 2에 대응됩니다.

```python
>> le.transform(df['score'])

array([1, 2, 0, 2, 1])
 
```

<br>

따라서 categorical 데이터를 숫자로 변경시키면 위 처럼 변형될 수 있습니다.

```python
list(le.inverse_transform([2, 2, 1]))
>> ['weak', 'weak', 'strong']

```

반대로 숫자를 문자로 원복시킬 수도 있습니다.