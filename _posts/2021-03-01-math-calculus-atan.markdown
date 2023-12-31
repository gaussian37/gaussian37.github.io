---
layout: post
title: atan과 atan2 비교
date: 2021-03-01 00:00:00
img: math/calculus/atan/0.png
categories: [math-calculus] 
tags: [atan, atan2] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 이번글에서는 간단하게 `atan`과 `atan2` 함수의 차이점에 대하여 확인해 보고 `atan`을 이용하여 어떻게 `atan2`를 만드는 지 알아보도록 하겠습니다.

<br>

- `atan`과 `atan2`모두 `inverse tangent`를 구하는 데 사용하는 함수이지만 사용하는 데 큰 차이점이 있습니다.

<br>

- ## **atan**

<br>

- `atan`은 하나의 입력을 받아서 `arctangent` 값을 출력합니다. 입력값은 `tangent`의 각도를 의미하며 범위는 $$ (-\pi/2 , \pi/2) $$ 의 범위를 가집니다. 따라서 `atan`함수는 4사분면 전체를 모두 다루지 못하고 **1사분면과 4사분면을 다룰 수 있습니다.**

<br>

- $$ \theta = \arctan (x), \ \text{where } -\frac{\pi}{2} \lt \theta \lt \frac{\pi}{2} $$

<br>

- ## **atan2**

<br>

- 반면 `atan2` 함수는 $$ y, x $$ 2가지 값을 전달 받으며 일반적으로 순서 또한 $$ y, x $$ 순서로 받습니다. 이 값은 1사분면에서 `atan`의 입력값과 비교하면 $$ y/x $$ 에 해당합니다.
- `atan2`에서 값을 2개로 구분하여 받는 이유는 부호를 고려하기 위해서 입니다. 부호를 고려한다면 4사분면 전체를 고려하여 `arctangent`를 사용할 수 있기 때문입니다. 따라서 출력의 범위는 $$ (-\pi, \pi] $$ 가 됩니다.

<br>

- $$ \theta = \arctan2(y, x), \ \text{where } -\pi \lt \theta \le \pi $$

<br>

- ## **atan을 이용한 atan2 구현**

<br>

```python
import numpy as np

def custom_arctan2(y, x):
    if x > 0:
        return np.arctan(y / x)
    elif x < 0 and y >= 0:
        return np.arctan(y / x) + np.pi
    elif x < 0 and y < 0:
        return np.arctan(y / x) - np.pi
    elif x == 0 and y > 0:
        return np.pi / 2
    elif x == 0 and y < 0:
        return -np.pi / 2
    else:  # x == 0 and y == 0
        # The point is at the origin, so the angle is undefined.
        # You could return NaN or some specific value here.
        return np.nan

# Example usage
y = 1
x = -1
angle = custom_arctan2(y, x)
print("Angle in radians:", angle)
```

<br>

- 아래와 같이 테스트 해 보았습니다.

<br>

```python
import numpy as np

def custom_arctan2(y, x):
    if x > 0:
        return np.arctan(y / x)
    elif x < 0 and y >= 0:
        return np.arctan(y / x) + np.pi
    elif x < 0 and y < 0:
        return np.arctan(y / x) - np.pi
    elif x == 0 and y > 0:
        return np.pi / 2
    elif x == 0 and y < 0:
        return -np.pi / 2
    else:  # x == 0 and y == 0
        return np.nan

# Generating 10 random test cases
test_cases = np.random.randn(10, 2)

# Testing custom_arctan2 function and comparing with numpy's arctan2 function
results = []
for x, y in test_cases:
    custom_result = custom_arctan2(y, x)
    np_result = np.arctan2(y, x)
    results.append((x, y, custom_result, np_result))
```

<br>

- 위 코드를 통하여 `custom_arctan2`와 `np.arctan2`가 동일한 값을 얻을 수 있음을 확인하였습니다.

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>