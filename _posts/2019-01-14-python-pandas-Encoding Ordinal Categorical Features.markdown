---
layout: post
title: Category 형태의 Feature를 Ordinal 형태로 변경
date: 2019-01-14 00:00:00
img: python/pandas/pandas.png
categories: [python-basic] 
tags: [python, pandas, categorical, ordinal, replace] # add tag
---

이번 글에서는 category 형태의 feature들을 숫자 형태로 바꾸는 간단한 방법에 대하여 알아보도록 하겠습니다.

+ 먼저 category 형태의 데이터는 주로 pandas를 많이 이용하여 처리를 많이 하므로 pandas를 불러오겠습니다.

```python
import pandas as pd
```

<br>

+ 실습해 볼 데이터 예제를 만들어 보겠습니다.

```python
df = pd.DataFrame({'Score': ['Low', 
                             'Low', 
                             'Medium', 
                             'Medium', 
                             'High']})
                             
 >> df
 
 	Score
0	Low
1	Low
2	Medium
3	Medium
4	High
```

<br>

+  각각의 category 데이터에 맞는 숫자 값을 만들어 줍니다.

```python
scale_mapper = {'Low':1, 
                'Medium':2,
                'High':3}
```

<br>

+ scale_mapper를 통하여 새로운 feature를 만들어 줍니다.

```python
df['Scale'] = df['Score'].replace(scale_mapper)


>> df

	Score	Scale
0	Low	1
1	Low	1
2	Medium	2
3	Medium	2
4	High	3
```