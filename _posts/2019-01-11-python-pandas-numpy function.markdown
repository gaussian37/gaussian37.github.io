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

## **np.c_[]**

<br>

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

## **np.meshgrid()**

<br>

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

## **ravel(), flatten()**

<br>

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

<br>

## **np.polyfit으로 fitting**

<br>

- `np.polyfit`으로 `least square` 방법으로 fitting하는 방법을 간략하게 정리하겠습니다.
- 자세한 내용은 [링크](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html#numpy-polyfit)를 참조하시면 됩니다.
- 전체적인 방법은 `least square`를 이용하여 1차, 2차, n차 다항식을 fitting 합니다.
- 파라미터는 첫번째로 입력할 `x 값`, 두번째로 입력할 `y 값`, 그리고 fitting할 `degree`를 입력하면 됩니다.

<br>

```python
x = [1,2,3,4,5]
y = [2,4,6,8,10]

one_d_fit = np.polyfit(x, y, 1)
print(one_d_fit)
two_d_fit = np.polyfit(x, y, 2)
print(two_d_fit)

p1 = np.poly1d(one_d_fit)
p2 = np.poly1d(two_d_fit)

p1(1)
p2(1)
``` 

<br>

- 먼저 `np.polyfit(x, y, degree)` 순으로 입력하면 됩니다.
- 그러면 출력되는 값은 파라미터 값이 차례대로 출력되는데 1차로 fitting한 경우 $$ ax + b $$에서 (a, b) 순서대로 출력됩니다.
- 2차로 fitting한 경우 $$ ax^{2} + bx + c $$에서 (a, b, c) 순서로 출력됩니다.
- prediction할 때에 파라미터와 값을 대응시키기 불편할 수 있습니다. 이 때, `np.poly1d` 함수를 이용하여 함수를 생성할 수 있습니다.
- 위 코드와 같이 `p1 = np.poly1d(onw_d_fit)`을 선언하고 `y예측값 = p1(x값)`과 같이 이용할 수 있습니다. 즉, `np.poly1d`에 입력된 파라미터와 파라미터 갯수에 따라서 degree와 계수가 정해지게 되고 입력값만 넣으면 예측값이 나오는 함수가 만들어진 셈입니다. 
