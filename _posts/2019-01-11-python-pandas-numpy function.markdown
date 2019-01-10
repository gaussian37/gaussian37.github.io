---
layout: post
title: Numpy에서 종종 사용하는 기능 모음
date: 2019-01-11 00:00:00
img: python/pandas/numpy.png
categories: [python-pandas] 
tags: [Numpy, 넘파이] # add tag
---

Numpy에서 종종 사용하는 기능들을 정리해 보려고 합니다.

<br><br>

## np.c_[]

np.c_[]를 이용하면 행으로 입력한 것이 열로 입력이 됩니다. 정확하게는 첫번째 축이 두번째 축이 되는 것입니다.
예제를 보면 쉽게 이해하실 수 있습니다.

```python
>>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
array([[1, 4],
       [2, 5],
       [3, 6]])
>>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
array([[1, 2, 3, 0, 0, 4, 5, 6]])
```

첫 번째 예제를 보면 행으로 입력한 것이 열 단뒤로 들어간 것을 볼 수 있습니다.

<br><br>

## np.meshgrid()

2차원 평면에서 grid를 만들어 주려면 x, y 축에 대하여 겹치는 좌표의 경우를 다 만들어 주어야합니다.
이 때, grid에 해당하는 격자를 쉽게 만들어 주는 함수가 `np.meshgrid()` 함수 입니다.

예를 들면 x = {0, 1, 2} 이고 y = {0, 1, 2, 3} 이라면 좌표로 총 12개의 경우의 수가 나옵니다.
이것을 간단하게 만들어 보겠습니다.

```python
x = np.arange(3)
y = np.arange(4)

xx, yy = np.meshgrid(x, y)

>> xx
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])
       
>> yy
array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])

```

이 때, (0, 0) ~ (4, 3)까지의 조합을 만들 수 있습니다.

<br><br>

## ravel(), flatten()

다차원 배열을 1차원으로 펼치기 위해서는 `np.flatten()` 또는  `np.ravel()` 메서드를 사용합니다.

```python
>> xx
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])

>> xx.flatten() # 또는 xx.ravel()

array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
```

<br><br>