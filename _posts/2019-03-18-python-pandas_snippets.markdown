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

