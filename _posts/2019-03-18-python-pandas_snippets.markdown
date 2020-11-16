---
layout: post
title: Pandas 기본 문법 및 코드 snippets
date: 2019-03-18 00:00:00
img: python/basic/pandas.png
categories: [python-basic] 
tags: [pandas, python, python 기본] # add tag
---

- 이 글에서는 `Pandas`를 사용하면서 필요하다고 느끼는 `Pandas 기본 문법 및 코드`들을 정리해 보겠습니다.

<br>

## **목차**

<br>

- ### DataFrame에 column 추가
- ### DataFrame에 행 단위로 데이터 추가하기
- ### pd.read_csv(excel) 함수를 통하여 파일 읽을 때
- ### df.to_csv(excel) 함수를 통하여 파일 쓸 때
- ### column 명 확인
- ### category 데이터 → Ordinal 데이터로 변경


<br>

## **DataFrame에 column 추가**

<br>

- DataFrame을 생성할 때, column을 추가하려면 다음과 같이 DataFrame 생성 시 옵션을 넣을 수 있습니다.

<br>

```python
columns = ['name', 'age', 'address', 'phone']
data_frame = pd.DataFrame(columns = columns)
```

<br>

## **DataFrame에 행 단위로 데이터 추가하기**

<br>

- DataFrame에 한 행씩 데이터를 차곡 차곡 쌓아가고 싶으면 아래 코드와 같이 `.loc[i]`를 이용하여 한 행씩 접근하고 column의 갯수 만큼의 값을 가지는 리스트를 `df.loc[i]`에 입력해 주면 됩니다.

<br>

```python
import pandas as pd
from numpy.random import randint

df = pd.DataFrame(columns=['lib', 'qty1', 'qty2'])
for i in range(5):
    df.loc[i] = ['name' + str(i)] + list(randint(10, size=2))
```

<br>

## **pd.read_csv(excel) 함수를 통하여 파일 읽을 때**

<br>

- ① 파일을 읽을 때, 첫 열의 인덱스(0, 1, 2, ...)를 만들고 싶지 않다면 다음 옵션을 준다.
    - `index_col = False`
- ② 파일을 읽을 때, 첫 행에 헤더를 만들고 싶지 않다면 다음 옵션을 줍니다.
    - `header = None`
- ③ 파일을 읽을 때, 구분자의 기준을 주고 싶다면 다음 옵션을 줍니다.
    - `sep = ','`, 특정 문자를 입력하면 됩니다.

<br>

## **df.to_csv(excel) 함수를 통하여 파일 쓸 때**

<br>

- ① 파일을 쓸 때, 첫 열의 인덱스(0, 1, 2, ...)를 만들고 싶지 않다면 다음 옵션을 준다.
    - `index = False`
- ② 파일을 쓸 때, 첫 행에 헤더를 만들고 싶지 않다면 다음 옵션을 줍니다.
    - `header = None`
- ③ 파일을 쓸 때, 구분자의 기준을 주고 싶다면 다음 옵션을 줍니다.
    - `sep = ','`, 특정 문자를 입력하면 됩니다.

<br>

## **column 명 확인**

<br>

- DataFrame에서 column의 이름을 확인할 때, 다음과 같습니다.

<br>

```python
df = pd.read_csv("file.csv")
column_list = list(df.columns)
column_list = df.columns.values.tolist()
```

<br>

## **category 데이터 → Ordinal 데이터로 변경**

<br>

- 이번 글에서는 category 형태의 데이터들을 숫자 형태로 바꾸는 간단한 방법에 대하여 알아보도록 하겠습니다.
- 이 작업은 데이터 분석을 할 때 상당히 중요하기 때문에 자주 사용됩니다. 왜냐하면 데이터 분석 시 카테고리 형태의 데이터는 수치화 할 수 없기 때문입니다. 다음 예제를 살펴보도록 하겠습니다.

<br>

```python
import pandas as pd
df = pd.DataFrame({'Score': [
    'Low', 'Low', 'Medium', 'Medium', 'High'
    ]})

print(df)
#     Score
# 0     Low
# 1     Low
# 2  Medium
# 3  Medium
# 4    High
```

<br>

- 위 예제에서 Low, Medium, High과 같은 문자열을 그대로 데이터 분석을 하기는 어렵습니다. 따라서 다음과 같이 Low = 1, Medium = 2, High = 3으로 매핑시켜 보겠습니다.

<br>

```python
scale_mapper = {'Low':1, 'Medium':2, 'High':3}
df['Scale'] = df['Score'].replace(scale_mapper)

print(df)
#     Score  Scale
# 0     Low      1
# 1     Low      1
# 2  Medium      2
# 3  Medium      2
# 4    High      3
```

<br>

- 위 코드와 같이 새로운 축에 Scale이 생긴것을 확인할 수 있습니다. 물론 Low, Medium, High이 필요없기 때문에 Score에 그대로 대입할 수 있습니다.

<br>

```python
scale_mapper = {'Low':1, 'Medium':2, 'High':3}
df['Score'] = df['Score'].replace(scale_mapper)

print(df)
#    Score
# 0      1
# 1      1
# 2      2
# 3      2
# 4      3
```


Structured data에서 결측값(Missing Value)를 제거하는 방법에 대하여 알아보겠습니다.

<br>

- 먼저 라이브러리를 입력합니다.

<br>

```python
# Load libraries
import numpy as np
import pandas as pd
```

<br>



<br>

## Pandas에서 결측값 제거하기

<br>

+ 행렬을 만들어 줍니다.

```python
X = np.array([[1, 2], 
              [6, 3], 
              [8, 4], 
              [9, 5], 
              [np.nan, 4]])
```

+ Numpy로 만든 행렬을 Pandas 타입으로 바꿔 줍니다.
+ `df.dropna()`를 통하여 결측값을 제거합니다.

```python
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])

# 결측값을 제거해 줍니다.
df.dropna()
```

<br>

## Pandas에서 결측값 인덱스 받기

<br>

- 임의의 DataFrame을 `NaN` 값을 포함하여 만든 후 NaN의 값이 있는 좌표만 따로 반환 받습니다.

<br>

```python
X = np.array([[1, 2], 
              [6, 3], 
              [8, 4], 
              [9, 5], 
              [np.nan, 4]])
              
df = pd.DataFrame(X)

print( np.asarray(df.isnull()).nonzero() )

>> (array([4], dtype=int64), array([0], dtype=int64))
```

<br>
