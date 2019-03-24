---
layout: post
title: Python 기본 문법 모음
date: 2019-03-18 00:00:00
img: python/pandas/python.jpg
categories: [python-pandas] 
tags: [python, python 기본] # add tag
---

+ 이 글에서는 Python을 사용하면서 필요하다고 느끼는 `Python 기본 문법`에 대하여 알아보겠습니다.

### Comparator를 이용한 클래스 정렬

+ 리스트를 정렬할 때 리스트의 원소가 클래스라면 정렬하는 기준을 정해주어야 합니다.
+ 가장 유연하게 정할 수 있는 방법 중 하나인 `Comparator`를 사용하는 방법에 대하여 알아보겠습니다.

```python
sorted(iterable, key, reverse)
``` 

+ 위 함수를 이용하여 정렬을 합니다.
+ `iterable` 에는 탐색이 가능한 타입을 입력합니다. 대표적인 예로 정렬할 리스트를 넣으면 됩니다.
+ `key`에는 정렬에 사용할 `comparator`를 입력합니다.
+ `reverse`는 역으로 정렬할 것인지에 대한 파라미터로 필요시 사용하면 됩니다.

<br>

+ 여기서 중요한 `key`에 입력할 `comparator`에 대하여 알아보도록 하겠습니다.

```python
sorted(A, key=lambda x:foo(x))
```

+ A라는 리스트가 있다고 하겠습니다. A리스트의 원소는 클래스로 (name, age)의 값을 가지고 있다고 가정하겠습니다.
    + 예를들어 ``` A = [("JISU", 28), ("JINSOL", 30)] ```
+ 이 때, 나이 기준으로 정렬하고 싶으면 나이에 대한 기준을 주어야 합니다.

```python
def foo(x):
    return x.age
``` 

+ 위 코드와 같이 클래스의 어떤 값으로 기준을 주면 됩니다. 

```python
sorted(A, key=lambda x:foo(x))
```

+ 따라서 위와 같이 정의 하면 각 원소는 foo라는 함수를 통하여 어떤 값을 기준으로 가질 지 알게 됩니다.

