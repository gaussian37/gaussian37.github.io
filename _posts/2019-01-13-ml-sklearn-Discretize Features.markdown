---
layout: post
title: sklearn을 이용한 데이터 분할
date: 2019-01-13 00:00:00
img: ml/sklearn/sklearn.png
categories: [ml-sklearn] 
tags: [split, 데이터 분할, python, machine learning, ml, sklearn, scikit learn,] # add tag
---

머신러닝을 적용하기 이전에 데이터 전처리를 할 필요가 있습니다.
데이터 전처리 중 데이터를 적절하게 변경 및 분할할 필요가 있는데, 그 방법에 대하여 알아보도록 하겠습니다.

<br><br>

## Binary 형태로 만들기

+ 먼저 필요한 라이브러리를 불러 옵니다.

```python
from sklearn.preprocessing import Binarizer
import numpy as np
```

+ 임시로 데이터를 만듭니다.

```python
age = np.array([[6], 
                [12], 
                [20], 
                [36], 
                [65]])
```

+ 18살을 기준으로 Binary로 데이터를 변경합니다.
+ Binarizer는 기준값 미만은 0으로 기준값 이상은 1로 만듭니다.

```python
# Create binarizer
binarizer = Binarizer(18)

# Transform feature
>> binarizer.fit_transform(age)

array([[0],
       [0],
       [1],
       [1],
       [1]])

```

## Bin 기준으로 나누기

+ 데이터를 Bin들을 기준으로 0, 1, 2, ... 로 나누고 싶을 땐 Numpy를 이용하면 됩니다.

```python
age = np.array([[6], 
                [12], 
                [20], 
                [36], 
                [65]])
                
>> np.digitize(age, bins=[20,30,64])

array([[0],
       [0],
       [1],
       [2],
       [3]])

```