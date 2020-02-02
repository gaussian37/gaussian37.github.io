---
layout: post
title: 파이썬에서의 asterisk(*) 사용 방법
date: 2020-01-25 00:00:00
img: python/basic/python.jpg
categories: [python-basic] 
tags: [python, asterisk, 별] # add tag
---

<br>

- 참조 : https://medium.com/understand-the-python/understanding-the-asterisk-of-python-8b9daaa4a558
- C언어에서 주소값 변수의 값 참조를 할 떄 `*`를 사용하듯이 파이썬에서도 산술 연산 이외에 사용되는 

<br>

## **목차**

<br>

- ### 산술 연산
- ### 리스트의 반복 확장
- ### 가변 인수 packing
- ### 컨테이터 unpacking


<br>

## **산술 연산**

<br>

- 먼저 가장 기본적인 연산인 산술 연산으로 사용되는 `곱하기`와 `제곱`은 다음과 같습니다.
- `2 * 5` : 10
- `2 ** 5` : 32

<br>

## **리스트의 반복 확장**

<br>

- 리스트나 튜플에 `*` 연산자를 적용하면 내부 원소를 곱해진 원소만큼 붙여서 확장합니다.

```python
zero_list = [0] * 10
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

zero_tuple = (0,) * 10
# (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
```

<br>

## **가변 인수 packing**

<br>

- 종종 일부 함수에 대해 가변성 인수 (또는 변수)가 필요합니다. 예를 들어, 전달 해야 할 인수의 갯수를 모르거나 어떤 이유로 임의의 전달 인수로 무언가를 처리해야하는 경우가 필요하기 때문이지요.
- 파이썬에는 두 가지 종류의 인수가 있습니다. 하나는 `위치 인수`이고 다른 하나는 `키워드 인수`입니다. 전자는 위치에 따라 지정되고 후자는 인수의 이름 인 키워드가 포함 된 인수입니다.
- 가변성 위치 / 키워드 인수를 살펴보기 전에 `위치 인수` 및 `키워드 인수`에 대해 간단히 설명하겠습니다.

<br>

```python
def print_name(first, second, third=None, fourth=None):
    names = {}
    names[1], names[2] = first, second
    names[3] = third if third is not None else 'Nobody'
    names[4] = fourth if fourth is not None else 'Nobody'
    print(names)    

# 2개의 위치 인수(positional argument)를 전달합니다.
print_name('ming', 'alice')
# 2개의 위치 인수와 1개의 키워드 인수를 전달합니다.
print_name('alice', 'ming', third='mike')
# 2개의 위치 인수와 2개의 키워드 인수를 전달합니다. (2개의 키워드 인수 중 1개인 마치 위치 인수 처럼 전달하였습니다.)
print_name('alice', 'ming', 'mike', fourth='jim')
```

<br>

- 위 함수에는 두 개의 위치 인수가 있습니다. (첫번째, 두번째 인수) 또한 두 개의 키워드 인수가 있습니다. (세번째, 네번째 인수)
- 위치 인수의 경우 생략 할 수 없고 **선언 된 각 인수 수에 대한 모든 위치 인수를 올바른 위치로 전달**해야합니다. 
- 그러나 `키워드 인수`의 경우 함수를 선언 할 때 기본값을 설정할 수 있으며 인수를 생략하면 해당 기본값이 인수의 값으로 입력됩니다. 즉, **키워드 인수는 생략 할 수 있습니다.**
- 따라서 키워드 인수를 생략 할 수 있으므로 키워드 인수는 위치 인수 전에 선언 할 수 없다는 것입니다. 따라서 다음 코드는 잘못된 코드입니다.

<br>

```python
def print_name(first, second=None, third, fourth=None):
    ...
```

<br>

- 위 예에서 세 번째 경우에는 3개의 위치 인수와 1개의 키워드 인수가 있음을 알 수 있습니다. `키워드 인수`의 경우 **전달된 위치가 선언 된 위치와 동일**하면 키워드를 제외하고 **위치 인수로 전달**할 수 있습니다. 즉, 위에서 mike는 세 번째 키로 자동 전달됩니다.
- 위와 같은 방법의 단점은 변수의 갯수와 종류가 가변적인 경우를 다룰 수 없다는 것입니다.

<br>

- 다음과 같이 `*`를 이용하여 인수를 전달하면 가변 인수를 전달 할 수 있습니다.
- 다음 예제는 `위치 인수`만 존재하는 경우의 가변 인수 전달 방법입니다.

```python
def print_name(*args):
    print(args) 

print_name('ming', 'alice', 'tom', 'wilson', 'roy')
# ('ming', 'alice', 'tom', 'wilson', 'roy')
```

<br>

- 다음은 `**`를 이용하여 `키워드 인수`를 전달하는 방법입니다.
- `키워드 인수`를 이용하면 키워드 인수의 `Key`와 `Value`를 동시에 처리할 수 있습니다.

<br>

```python
def print_name(**kwargs):
    print(kwargs)
print_name(first='ming', second='alice', fourth='wilson', third='tom', fifth='roy')
# {'first': 'ming', 'second': 'alice', 'fourth': 'wilson', 'third': 'tom', 'fifth': 'roy'}
```

<br>

- 위의 `*args`는 임의의 수의 위치 인수를 허용 함을 의미하고 `**kwargs`는 임의의 수의 키워드 인수를 허용 함을 의미합니다. 
    - 여기서 `*args`, `**kwargs`는 패킹이라고 합니다.
- 위에서 볼 수 있듯이 임의의 수의 위치 또는 키워드 값을 보유 할 수있는 인수를 전달합니다. 위치로 전달 된 인수는 `args`라는 `튜플`에 저장되고 키워드로 전달 된 인수는 `kwargs`라는 `dict`에 저장됩니다.
- 앞에서 언급했듯이 키워드 인수는 위치 인수 앞에 선언 할 수 없으므로 다음 코드는 에러를 발생시킵니다.

<br>

```python
def print_name(**kwargs, *args):
    ...
```

<br>

- `가변 인수`는 매우 자주 사용되는 기능이며 많은 오픈 소스 프로젝트에서 볼 수 있습니다. 
- 일반적으로 많은 오픈 소스는 일반적으로 사용되는 인수 이름 (`*args` 또는 `**kwargs`)을 가변 인수 이름으로 사용합니다.
- 물론, `*required` 또는 `**optional` 처럼 고유 한 이름을 사용할 수도 있습니다. (단, 프로젝트가 오픈 소스이고 가변 인수에 특별한 의미가 없는 경우 `*args` 및 `**kwarg` 사용 규칙을 따르는 것이 좋습니다)

<br>

## **컨테이터 unpacking **

<br>

- `*`는 `컨테이너 unpacking`에도 사용할 수 있습니다. 그 원리는 위에서 다룬 가변 인수를 사용하는 것과 유사합니다. 가장 쉬운 예는 리스트, 튜플 또는 dict 형식의 데이터가 있고 함수가 변수 인수를 취하는 것입니다.

<br>

```python
from functools import reduce

primes = [2, 3, 5, 7, 11, 13]

def product(*numbers):
    print(numbers)
    print(*numbers)
    p = reduce(lambda x, y: x * y, numbers)
    return p

product(*primes)
# (2, 3, 5, 7, 11)
# 2 3 5 7 11 13
# Out : 30030

product(primes)
# ([2, 3, 5, 7, 11, 13],)
# [2, 3, 5, 7, 11, 13]
# Out : [2, 3, 5, 7, 11, 13]
```

<br>

- `product()`는 변수 인수를 취하므로 리스트 데이터를 unpacking하고 해당 함수에 전달해야합니다. 
- 이 경우, 소수를 `*primes`로 전달하면 소수 목록의 모든 요소가 unpacking된 다음 numbers라는 목록에 저장됩니다. 압축을 풀지 않고 해당 리스트의 소수들을 함수에 전달하면 numbers에 소수의 모든 요소가 아닌 하나의 소수 목록만 표시됩니다.

<br>

- 튜플의 경우 list와 똑같이 수행 할 수 있으며 dict의 경우 *대신 `**`를 사용하면 됩니다.

<br>

```python
headers = {
    'Accept': 'text/plain',
    'Content-Length': 348, 
    'Host': 'http://mingrammer.com' 
}  

def pre_process(**headers): 
    content_length = headers['Content-Length'] 
    print('content length: ', content_length) 
    
    host = headers['Host']
    if 'https' not in host: 
        raise ValueError('You must use SSL for http communication')  
        
pre_process(**headers)
# content length:  348
```

<br>

- 또한 unpacking의 또 다른 유형이 있습니다. 리스트 또는 튜플 데이터를 다른 변수에 동적으로 unpack합니다.

<br>

```python
numbers = [1, 2, 3, 4, 5, 6]

# The left side of unpacking should be list or tuple.
*a, = numbers
# a = [1, 2, 3, 4, 5, 6]

*a, b = numbers
# a = [1, 2, 3, 4, 5]
# b = 6

a, *b, = numbers
# a = 1 
# b = [2, 3, 4, 5, 6]

a, *b, c = numbers
# a = 1
# b = [2, 3, 4, 5]
# c = 6
```