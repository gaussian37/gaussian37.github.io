---
layout: post
title: Python 기본 문법 및 코드 snippets
date: 2019-01-05 00:00:00
img: python/basic/python.jpg
categories: [python-basic] 
tags: [python, python 기본] # add tag
---

- 이 글에서는 Python을 사용하면서 필요하다고 느끼는 `Python 기본 문법 및 코드`에 대하여 알아보겠습니다. 
- 목차를 검색하셔서 사용하시기 바랍니다.

<br>

## **목차**

<br>

- ### **--- 문법 및 셋팅 관련 ---**
- #### [if __name__ == "__main__" 사용 이유](#if-__name__--__main__-사용-이유)
- #### [함수의 파라미터로 빈 리스트 대신 None을 사용할 것](#함수의-파라미터로-빈-리스트-대신-none을-사용할-것-1)
- #### [A,B = B,A 로 swap](#ab--ba-로-swap-1)
- #### [특정 함수에서만 사용하는 패키지는 함수 내부에서 import 할 것](#특정-함수에서만-사용하는-패키지는-함수-내부에서-import-할-것-1)
- #### [set으로 변환하여 리스트의 원소 유무 확인](#set으로-변환하여-리스트의-원소-유무-확인-1)
- #### [underscore(_)의 활용](#underscore_의-활용-1)
- #### [Function annotation](#function-annotation-1)
- #### [typing 모듈로 타입 표시하기](#typing-모듈로-타입-표시하기-1)
- #### [is와 == 연산의 차이점](#is와--연산의-차이점-1)
- #### [파이썬에서의 asterisk(*) 사용 방법](#파이썬에서의-asterisk-사용-방법-1)

<br>

- ### **--- 자주사용하는 함수 ---**
- #### [print 함수 관련](#print-함수-관련-1)
- #### [기본 자료 구조의 사용법](#기본-자료-구조의-사용법-1)
- #### [다양한 sort 방법](#다양한-sort-방법-1)
- #### [comparator를 이용한 클래스 정렬](#comparator를-이용한-클래스-정렬-1)
- #### [all과 any](#all과-any-1)
- #### [lambda](#lambda-1)
- #### [map](#map-1)
- #### [filter](#filter-1)
- #### [reduce](#reduce-1)
- #### [slice](#slice-1)
- #### [Counter](#counter-1)
- #### [defaultdict를 이용한 multimap 사용](#defaultdict를-이용한-multimap-사용-1)
- #### [OrederedDict](#oredereddict-1)
- #### [bisect를 사용한 이진 탐색](#bisect를-사용한-이진-탐색-1)
- #### [for loop 스타일 (zip, range, enumerate, sorted, filter)](#for-loop-스타일-zip-range-enumerate-sorted-filter-1)
- #### [deque은 사이즈 관리를 자동으로 한다.](#deque은-사이즈-관리를-자동으로-한다-1)
- #### [파이썬 프로그램 정상 종료 시키기](#파이썬-프로그램-정상-종료-시키기-1)
- #### [argparse 사용법](#argparse-사용법-1)
- #### [문자열 검색 : startswith, endswith, in, find, re](#문자열-검색--startswith-endswith-in-find)
- #### [디렉토리(+ 파일) 복사하기](#디렉토리-파일-복사하기-1)
- #### [디렉토리(- 파일) 복사하기](#디렉토리--파일-복사하기-1)
- #### [특정 문자를 기준으로 split](#특정-문자를-기준으로-split-1)
- #### [숫자에 0을 채워서 출력](#숫자에-0을-채워서-출력-1)
- #### [문자열 양 끝의 공백 문자 없애기](#문자열-양-끝의-공백-문자-없애기-1)
- #### [os 관련 함수 모음](#os-관련-함수-모음-1)
- #### [glob을 이용한 폴더 및 파일 접근](#glob을-이용한-폴더-및-파일-접근-1)
- #### [pickle 사용 방법](#pickle-사용-방법-1)
- #### [exec을 이용한 문자열로 코드 실행](#exec을-이용한-문자열로-코드-실행-1)
- #### [type과 isinstance를 통한 데이터 타입 확인](#type과-isinstance를-통한-데이터-타입-확인-1)
- #### [dir을 통한 module 확인](#dir을-통한-module-리스트-확인)
- #### [enumerate 함수의 응용](#enumerate-함수의-응용-1)
- #### [counting 방법](#counting-방법-1)
- #### [디렉토리 상위 레벨 패키지 import](#디렉토리-상위-레벨-패키지-import-1)
- #### [리스트를 딕셔너리로 변환](#리스트를-딕셔너리로-변환-1)
- #### [유니크한 리스트 생성 방법](#유니크한-리스트-생성-방법-1)
- #### [파이썬 실행 경로 추가](#파이썬-실행-경로-추가-1)
- #### [문자열을 이용하여 패키지 import](#문자열을-이용하여-패키지-import-1)
- #### [Dictionary와 JSON](#dictionary와-json-1)
- #### [Dicionary의 최대, 최소 value 찾기](#dicionary의-최대-최소-value-찾기-1)
- #### [Dictionary로 구성된 list 정렬 방법](#dictionary로-구성된-list-정렬-방법-1)
- #### [Dictionary 합치는 방법](#dictionary-합치는-방법-1)
- #### [copy를 이용한 deepcopy](#copy를-이용한-deepcopy-1)
- #### [zipfile을 이용한 압축 풀기](#zipfile을-이용한-압축-풀기-1)

<br>

- ### **---함수의 응용---**
- #### [file read/write 방법](#file-readwrite-방법-1)
- #### [개행(new line) 구분 텍스트 텍스트 리스트 변환](#개행new-line-구분-텍스트-텍스트-리스트-변환-1)
- #### [파일의 첫 행 또는 끝 행 출력](#파일의-첫-행-또는-끝-행-출력-1)
- #### [현재 시각 문자열로 출력하는 방법](#현재-시각-문자열로-출력하는-방법-1)
- #### [파이썬에서 폴더 및 파일 있는지 확인 후 생성](#파이썬에서-폴더-및-파일-있는지-확인-후-생성-1)
- #### [리스트 내부의 경우의 수 조합하기](#리스트-내부의-경우의-수-조합하기-1)
- #### [모듈이 설치 되었는 지 확인](#모듈이-설치-되었는-지-확인-1)
- #### [pip가 설치가 안되어 있으면 설치](#pip가-설치가-안되어-있으면-설치-1)
- #### [모듈이 설치가 안되어 있으면 설치](#모듈이-설치가-안되어-있으면-설치-1)
- #### [입력한 디렉토리의 부모 디렉토리 출력](#모듈이-설치가-안되어-있으면-설치-1)
- #### [특정 경로의 특정 확장자 파일명만 prefix 추가](#특정-경로의-특정-확장자-파일명만-prefix-추가-1)
- #### [특정 경로의 특정 확장자 파일명만 suffix 추가](#특정-경로의-특정-확장자-파일명만-suffix-추가-1)
- #### [local 영역에서 global로 import](#특정-경로의-특정-확장자-파일명만-suffix-추가-1)
- #### [주어진 index 목록에 해당하는 값 불러오기](#주어진-index-목록에-해당하는-값-불러오기-1)
- #### [숫자 형태의 list를 특정 문자로 구분하여 문자열로 변환](#숫자-형태의-list를-특정-문자로-구분하여-문자열로-변환-1)
- #### [List를 group 단위로 나누기](#list를-group-단위로-나누기-1)
- #### [현재 실행 중인 파이썬 파일의 경로 확인 (__file__)](#현재-실행-중인-파이썬-파일의-경로-확인-__file__)
- #### [폴더의 하위 전체 구조 및 파일 모두 복사](#폴더의-하위-전체-구조-및-파일-모두-복사-1)
- #### [파일 경로를 경로와 파일명으로 나누기](#파일-경로를-경로와-파일명으로-나누기-1)
- #### [특정 경로의 특정 확장자 파일명 가져오기](#특정-경로의-특정-확장자-파일명-가져오기-1)
- #### [HTML 랜덤 컬러 만들기](html-랜덤-컬러-만들기-1)

<br>

# **--- 문법 및 셋팅 관련 ---**

<br>

## **if __name__ == "__main__": 사용 이유**

<br>

- 어떤 파이선 파일을 만들었을 때, 그 파이썬 파일을 import 하면 그 파일 안에 있는 함수들을 사용할 수 있게 됩니다.
- 이 때 `if __name__ == "__main__"`을 이용하여 main 함수를 컨트롤 하지 않으면 전역 변수 영역에 선언된 모든 함수들이 실행되게 됩니다.
- 예를 들어 `print` 함수들이 전역 변수 영역에 정의되어 있다면 `import` 할 때, 실행되어 원하지 않는 결과가 출력될 수 있습니다.
- 즉, 다른 파이썬에서 import 하였을 때, `def`로 정의된 함수들만 `import`되도록 하기 위해서는 main 함수를 통해 컨트롤 해야 합니다. 

<br>

## **함수의 파라미터로 빈 리스트 대신 None을 사용할 것**

<br>

- 함수를 정의할 때, 어떤 파라미터를 비어 있는 리스트로 정의하고 싶으면 `[ ]` 와 같은 형태가 아니라 `None`을 사용하여 비어있음을 나타내어야 합니다. 그 이유에 대해서 차근 차근 설명해 보겠습니다.

<br>

```python
# define a function invovling the default value for a list
def append_score(score, scores=[]):
    scores.append(score)
    print(scores)

append_score(98)
# [98]
append_score(92, [100, 95])
# [100, 95, 92]
append_score(94)
# [98, 94]
```

<br>

- 마지막에 실행 된 함수를 보면 입력하지도 않은 `98`이란 숫자가 출력되는 것을 볼 수 있습니다. 이 문제로 인해 `[]` 대신 None을 사용하여 빈 리스트 임을 표시해야 합니다.
- 이런 문제가 발생하는 이유는 **파이썬에서는 함수 또한 객체로 인식되기 때문**입니다. 즉, 위 함수가 하나의 객체이기 때문에 `scores`에 리스트가 전달되지 않은 경우 **객체 내부적으로 가지고 있는 리스트를 사용**하게 됩니다. 물론 위의 2번째 함수 호출 처럼 `[100, 95]`라는 리스트를 입력하면 이 리스트를 사용하지만 1, 3번째 함수 호출 처럼 리스트를 전달하지 않으면 객체 내부의 리스트를 사용한다고 이해하시면 됩니다. 확인하면 다음과 같습니다.

<br>

```python
# updated function to show the id for the scores
def append_score(score, scores=[]):
    scores.append(score)
    print(f'scores: {scores} & id: {id(scores)}')

append_score.__defaults__
# ([],)
id(append_score.__defaults__[0])
# 4650019968
append_score(95)
# scores: [95] & id: 4650019968
append_score(98)
# scores: [95, 98] & id: 4650019968
```

<br>

- 앞에서 설명한 바와 같이 함수가 하나의 객체이고 같은 객체 안의 멤버 변수인 리스트의 `id`를 출력해 봤을 때, 같은 `id`인 것을 확인할 수 있습니다. 즉, 리스트가 재사용된것입니다.
- 따라서 위와 같은 목적으로 사용된 함수의 올바른 사용법은 다음과 같습니다.

<br>

```python
# use None as the default value
def append_score(score, scores=None):
    if not scores:
        scores = []
    scores.append(score)
    print(scores)

append_score(98)
# [98]
append_score(92, [100, 95])
# [100, 95, 92]
append_score(94)
# [94]
```

<br>

## **A,B = B,A 로 swap**

<br>

- 파이썬에서는 `swap`을 할 때, 함수를 사용하거나 temp 변수를 할 필요 없이 `A, B = B, A`와 같이 사용하면 됩니다. 코드도 간단하고 명확해 집니다.

<br>

## **특정 함수에서만 사용하는 패키지는 함수 내부에서 import 할 것**

<br>

- 일부 함수에서만 사용하는 패키지는 전역으로 선언하지 말고 로컬로 선언하는 것이 메모리, 속도면에서 유리합니다. 다음 코드를 참조하시기 바랍니다.

<br>

```python
def f(x):
  import warnings, xyz, numpy 
  #do something 
 
def g(x):
  import pandas, numpy 
```

<br>

## **set으로 변환하여 리스트의 원소 유무 확인**

<br>

- 어떤 리스트의 원소 유무를 확인할 때 다음과 같이 확인 합니다.

<br>

```python
a = [1,2,3]
1 in a
# True
```

<br>

- 이 방법 보다 리스트를 셋으로 변환 시키고 셋을 이용하여 찾는 것이 좀 더 빠릅니다. 리스트를 셋으로 변환하는 시간 복잡도가 $$ O(1) $$ 이기 때문입니다. (방법은 파이썬 내부 구현 사항임)
- 따라서 다음과 같이 사용하는 것이 효율적입니다.

<br>

```python
a = [1,2,3]
1 in set(a)
# True
```

<br>

## **underscore(_)의 활용**

<br>

- 파이썬에서는 다양한 목적으로 underscore를 사용합니다. 대표적으로 앞에 underscore 1개 또는 2개를 사용하는 것, 끝에 underscore를 1개 사용하는 것, 양 끝에 underscore를 2개사용하는 것 그리고 underscore만 사용하는 경우가 있습니다. 이 경우들의 사용 목적을 하나씩 살펴보도록 하겠습니다.

<br>

- ① `_var` : 클래스 내부에서 **private 하게 사용할 변수에 대하여 관습**적으로 underscore를 앞에 하나 붙여서 사용합니다. 다만, 다른 OOP 언어와 같이 외부 접근을 강제로 못하게 하는 기능이 추가되는 것은 아닙니다. 따라서 `_var` 형태는 private 이라고 생각하면 됩니다.
- ② `var_`, `__var` : 다른 변수명, 함수명, 클래스명과 **이름 충돌을 방지**하기 위해서 끝에 underscore를 붙이거나 앞에 underscore를 2개 붙여 사용하는 관습입니다.
- ③ __var__ : 양쪽에 2개의 underscore를 사용하는 것은 파이썬의 내장 기능 중 하나입니다. [참조 링크](https://docs.python.org/3/reference/datamodel.html#special-method-names)를 통하여 어떤 기능들이 있는 지 확인하시기 바랍니다. 물론 사용자가 단순히 변수명 처럼 사용할 수 있으나 내장 기능과 이름 충돌이 날 수 있으므로 사용을 권장하지 않습니다.
- ④ _ : underscore 한개 만을 사용할 수도 있는데, 주로 필요 없거나 중요하지 않은 변수를 임시로 받기 위해 사용합니다. 예를 들어 다음 예제를 참조하시기 바랍니다.

<br>

```python
for _ in range(5):
  do_something

beer = ('light', 'bitter', 70, 153)
 color, _, _, calories = beer
 ```

<br>

## **Function annotation**

<br>

- Function annotation은 python3 이상에서 사용 가능하며 function의 설명을 추가하기 위해 사용됩니다.
- annotation 이기 때문에 **annotation에 추가된 타입들을 맞추지 않아도 문법적으로 전혀 지장이 없습니다.** 사용 방법은 다음과 같습니다.

<br>

```python
def func(arg1: str, arg2: int, arg3: 'This is your name.') -> bool:
    print(arg1)
    print(arg2)
    print(arg3)
    return True
```

<br>

- 위와 같이 예제를 사용하였을 때, argument가 어떤 타입인 지, 또는 설명을 명시적으로 적을 수 있고 return 타입이 무엇인지 명시화 할 수 있습니다.
- arg1은 문자열이 입력되어야 하고, arg2는 정수, arg3는 설명을 통해서 어떤 입력이 들어와야 하는 지 적었습니다. 그리고 return 형은 bool로 설명하였습니다.
- 물론 annotation 타입으로 입력하지 않아도 전혀 상관없습니다. 강제하지 않기 때문입니다. 
- 하지만 이와 같이 설명을 적어주는 코드가 현재 많아지는 추세이니 사용하는 것을 추천드립니다.
- 또한 위와 같이 명시해 놓으면 IDE에서 annotation을 읽어서 보여주는 IDE가 많습니다. 따라서 실제 코드 작성할 때 도움이 많이 됩니다.

<br>

## **typing 모듈로 타입 표시하기**

<br>

- 타입 힌트(type hint)를 언어 차원에서 지원하기 위해서 파이썬 버전 3.5에서 typing 모듈이 추가되었습니다.
- typing 모듈 또한 일종의 annotation 이기 때문에 annotation 타입대로 사용하지 않아도 에러가 발생하지는 않습니다.

<br>

```python
from typing import List
nums: List[int] = [1, 2, 3]

from typing import Dict
countries: Dict[str, str] = {"KR": "South Korea", "US": "United States", "CN": "China"}

from typing import Tuple
user: Tuple[int, str, bool] = (3, "Dale", True)

from typing import Set
chars: Set[str] = {"A", "B", "C"}

# 여러 개의 annotation을 추가하고 싶을 때 사용
from typing import Union
def toString(num: Union[int, float]) -> str:
    return str(num)


# Optional은 Union에서 None을 추가한 것과 같은 의미를 가집니다.
from typing import Optional
def repeat(message: str, times: Optional[int] = None) -> list:
    if times:
        return [message] * times
    else:
        return [message]


# Call 가능한 인자를 표현할 때 사용합니다.
from typing import Callable
def repeat(greet: Callable[[str], str], name: str, times: int = 2) -> None:
    for _ in range(times):
        print(greet(name))


from typing import Iterable, List
def toStrings(nums: Iterable[int]) -> List[str]:
    return [str(x) for x in nums]

```

<br>

## **is와 == 연산의 차이점**

<br>

- 흔히 `같다`와 같은 비교 연산을 할 때, `is` 또는 `==` 연산자를 사용합니다. 비슷하지만 엄연히 다르므로 그 차이를 반드시 숙지하는 것이 좋습니다.
- `is`는 객체(Object)가 같은 지 비교합니다. 반면 `==`는 흔히 하는 값(Value)이 같은 지 비교합니다.
- 객체가 같다는 말은 파이썬에서 변수의 `id`가 같다는 뜻입니다. 반면 값이 같다는 것은 흔히 말하는 그 변수가 가지는 값이 같다는 뜻입니다. 다음 예제를 살펴보겠습니다.

<br>

```python
a = [1,2,3]
b = a
c = [1,2,3]

# id는 실행 환경에 따라 다릅니다.
print(id(a))
# 2763955372808
print(id(b))
# 2763955372808 
print(id(c))
# 2764303112200 

print(a is b)
# True
print(a is c)
# False
print(a == b)
# True
print(a == c)
# True
```

<br>

- 위 코드에서 a와 b는 같은 객체를 가리키기 때문에 id가 같습니다. 따라서 `is` 연산자로 비교하였을 때, True 입니다. 반면 a와 c는 서로 다른 객체를 가리키기 때문에 `is` 연산자로 비교 시 False 입니다. 하지만 a, b, c 모두 가지는 값은 같이 때문에 `==`연산자로 비교 시 모두 True입니다.

<br>

## **파이썬에서의 asterisk(*) 사용 방법**

<br>

- 참조 : https://medium.com/understand-the-python/understanding-the-asterisk-of-python-8b9daaa4a558
- C언어에서 주소값 변수의 값 참조를 할 떄 `*`를 사용하듯이 파이썬에서도 산술 연산과 그 이외의 기능에 대하여 알아보도록 하겠습니다.
- 가장 많이 사용되는 asterisk의 사용방법은 크게 4가지가 있습니다.
    - ① `산술 연산`
    - ② `리스트의 반복 확장`
    - ③ `가변 인수 packing`
    - ④ `컨테이터 unpacking`

#### **① 산술 연산**

<br>

- 먼저 가장 기본적인 연산인 산술 연산으로 사용되는 `곱하기`와 `제곱`은 다음과 같습니다.
- `2 * 5` : 10
- `2 ** 5` : 32

<br>

#### **② 리스트의 반복 확장**

<br>

- 리스트나 튜플에 `*` 연산자를 적용하면 내부 원소를 곱해진 원소만큼 붙여서 확장합니다.

```python
zero_list = [0] * 10
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

zero_tuple = (0,) * 10
# (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
```

<br>

#### **③ 가변 인수 packing**

<br>

- **asterisk는 가변 인수를 다룰 때**에도 사용됩니다.
- 종종 일부 함수에 대해 가변성 인수 (또는 변수)가 필요합니다. 예를 들어, 전달 해야 할 인수의 갯수를 모르거나 어떤 이유로 임의의 전달 인수로 무언가를 처리해야하는 경우가 필요하기 때문이지요.
- 파이썬에는 두 가지 종류의 인수가 있습니다. 하나는 `위치 인수`이고 다른 하나는 `키워드 인수`입니다. 전자는 위치에 따라 지정되고 후자는 인수의 이름인 키워드가 포함된 인수입니다.
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

- 위 함수에는 두 개의 위치 인수가 있습니다. (first, second 인수) 
- 또한 두 개의 키워드 인수가 있습니다. (third, fourth 인수)
- 위치 인수의 경우 생략 할 수 없고 **선언 된 각 인수 수에 대한 모든 위치 인수를 올바른 위치로 전달**해야합니다. 
- 그러나 `키워드 인수`의 경우 함수를 선언 할 때 기본값을 설정할 수 있으며 인수를 생략하면 해당 기본값이 인수의 값으로 입력됩니다. 즉, **키워드 인수는 생략 할 수 있습니다.**
- 따라서 키워드 인수를 생략 할 수 있으므로 키워드 인수는 위치 인수 전에 선언 할 수 없다는 것입니다. 따라서 다음 코드는 잘못된 코드입니다.

<br>

```python
def print_name(first, second=None, third, fourth=None):
    ...
```

<br>

- 위 예에서 세 번째 경우에는 3개의 위치 인수와 1개의 키워드 인수가 있음을 알 수 있습니다. `키워드 인수`의 경우 **전달된 위치가 선언된 위치와 동일**하면 키워드를 제외하고 **위치 인수로 전달**할 수 있습니다. 즉, 위에서 mike는 세 번째 키로 자동 전달됩니다.
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
    - 여기서 `*args`, `**kwargs`와 같이 처리하는 방식을 `패킹`이라고 합니다.
- 위에서 볼 수 있듯이 임의의 수의 위치 또는 키워드 값을 보유 할 수있는 인수를 전달합니다. 위치로 전달 된 인수는 `args`라는 `튜플`에 저장되고 키워드로 전달 된 인수는 `kwargs`라는 `dict`에 저장됩니다.
- 앞에서 언급했듯이 **키워드 인수는 위치 인수 앞에 선언 할 수 없으므로** 다음 코드는 에러를 발생시킵니다. **(위치 인수 → 키워드 인수 순서로 와야함)**

<br>

```python
# error code
def print_name(**kwargs, *args):
    ...
```

<br>

- `가변 인수`는 자주 사용되는 기능이며 많은 오픈 소스 프로젝트에서 볼 수 있습니다. 
- 일반적으로 많은 오픈 소스는 일반적으로 사용되는 인수 이름 (`*args` 또는 `**kwargs`)을 가변 인수 이름으로 사용합니다.
- 물론, `*required` 또는 `**optional` 처럼 고유 한 이름을 사용할 수도 있습니다. (단, 프로젝트가 오픈 소스이고 가변 인수에 특별한 의미가 없는 경우 `*args` 및 `**kwarg` 사용 규칙을 따르는 것이 좋습니다)

<br>

#### **컨테이터 unpacking**

<br>

- `*`는 `컨테이너 unpacking`에도 사용할 수 있습니다. 그 원리는 위에서 다룬 가변 인수를 사용하는 것과 유사합니다. 가장 쉬운 예는 리스트, 튜플 또는 dict 형식의 데이터가 있고 함수가 변수 인수를 취하는 것입니다.

<br>

```python
primes = [2, 3, 5, 7, 11, 13]

def product(*numbers):
    print(numbers, type(numbers))    
    
product(*primes)
product(primes)
```

<br>

- `product()`는 변수 인수를 취하므로 리스트 데이터를 unpacking하고 해당 함수에 전달해야합니다. 
- 이 경우, 소수를 `*primes`로 전달하면 **소수 목록의 모든 요소가 unpacking된 다음** numbers라는 목록에 저장됩니다. 압축을 풀지 않고 해당 리스트의 소수들을 함수에 전달하면 numbers에 소수의 모든 요소가 아닌 하나의 소수 목록만 표시됩니다.

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

<br>

# **--- 자주사용하는 함수 ---**

<br>

## **print 함수 관련**

<br>

- print 함수 사용 시 활용되는 팁들을 정리하겠습니다.
- `sep` : 데이터들을 연달아 출력할 때, 구분자로 사용하는 문자열을 지정할 수 있습니다. 기본적으로 공백 문자(` `)입니다.
    - `print('a', 'b', 'c', sep=', ')` : a, b, c
- `end` : print문 마지막에 특정 문자열을 출력합니다.
    - `print('a', 'b', 'end='!!!')` : a b!!!

<br>

## **기본 자료 구조의 사용법**

<br>

- 먼저 **list**에서 사용되는 주요 기능을 살펴보면 다음과 같습니다.
- `.append(obj)` : obj를 끝에 추가합니다.
- `.extend([obj1, ..., obj2])` : 리스트를 입력 받으며 입력 받은 리스트 원소들을 끝에 붙입니다.
- `.index(obj)` : obj의 인덱스를 찾습니다. 앞에서 부터 차례대로 찾으므로 중복된 obj 값이 있으면 가장 앞쪽의 obj 인덱스를 찾습니다. 
- `.sort()` : 리스트를 정렬합니다.

<br>

- **tuple**은 list와 유사하며 차이점은 list는 원소를 변경할 수 있는 반면 tuple은 변경이 불가능 하다는 점입니다.
- tuple 사용 시 주의할 만한 점은 원소가 1개인 튜플을 생성할 때에는 컴마를 붙여줘야 한다는 점입니다.
- 예를 들어 `(3)`은 int형인 반면 `(3,)`은 tuple 형입니다.

<br>

- **문자열**에서 사용되는 주요 기능을 살펴보면 다음과 같습니다.
- `.find(str)` : str이 발견되는 인덱스를 반환합니다.
- `.replace(str1, str2)` : str1을 str2로 대체합니다.
- `.count(str)` : str이 발견되는 횟수를 반환합니다.
- `.split(str)` : str을 기준으로 문자열을 자르고 잘린 부분들을 리스트로 만듭니다.
- `.strip()`, `.strip(str)` : 양 쪽 끝의 공백 문자를 제거합니다. 만약 어떤 문자열 str이 입력되면 그 문자열을 양쪽 끝에서 제거합니다.
    - 이와 유사하게 `.lstrip()`, `.rstrip()`도 사용가능합니다.
- `"str".join([str1, str2, ...])` : str1 + str + str2 + ... 형태로 이어줍니다.
- `+` : 문자열들을 이어 붙입니다. (concatenation)
- `* `: 문자열을 반복해서 이어 붙입니다.
- `format` 문법 : 

<br>

```python
'{0} is an {1}'.format('dog', 'animal')
'{name} is an {type}'.format(name='dog', type='animal')
```

<br>

- **set**에서 사용하는 주요 기능은 다음과 같습니다.
- `.intersection(set1)` : set1과 현재 기준 set의 교집합을 반환합니다.
- `.difference(set1)` : set에서 set1을 뺀 차집합을 반환합니다.
- `.union(set1)` : set과 set1의 합집합을 반환합니다.
- `.issubset(set1)` : set1이 set의 부분 집합인 지 반환합니다.
- `.update(set1)` : set1을 set에 업데이트 시킵니다.

<br>

## **다양한 sort 방법**

<br>

- 기본적으로 sort는 **파이썬에서 제공하는 기준**의 오름차순으로 정렬하며 `reverse=True` 옵션을 통해 내림차순으로 정렬할 수 있습니다. 

<br>

```python
# A list of numbers and strings
numbers = [1, 3, 7, 2, 5, 4]
words = ['yay', 'bill', 'zen', 'del']
# Sort them
print(sorted(numbers))
# [1, 2, 3, 4, 5, 7]
print(sorted(words))
# ['bill', 'del', 'yay', 'zen']

# Sort them in descending order
print(sorted(numbers, reverse=True))
# [7, 5, 4, 3, 2, 1]
print(sorted(words, reverse=True))
# ['zen', 'yay', 'del', 'bill']
```

<br>

- 만약 파이썬에서 제공하는 기준이 아닌 임의의 기준을 통하여 정렬하고 싶으면 `sorted`의 `key` 값에 lambda로 간단하게 기준을 줄 수 있습니다.

<br>

```python
#  Create a list of tuples
grades = [('John', 95), ('Aaron', 99), ('Zack', 97), ('Don', 92), ('Jennifer', 100), ('Abby', 94), ('Zoe', 99), ('Dee', 93)]
# Sort by the grades, descending
sorted(grades, key=lambda x: x[1], reverse=True)
# [('Jennifer', 100), ('Aaron', 99), ('Zoe', 99), ('Zack', 97), ('John', 95), ('Abby', 94), ('Dee', 93), ('Don', 92)]

# Sort by the name's initial letter, ascending
sorted(grades, key=lambda x: x[0][0])
# [('Aaron', 99), ('Abby', 94), ('Don', 92), ('Dee', 93), ('John', 95), ('Jennifer', 100), ('Zack', 97), ('Zoe', 99)]
```

<br>

- 위 처럼 `labmda`를 이용하면 쉽게 인라인으로 기준을 줄 수 있습니다. 이를 응용하여 좀 더 기준을 다양하게 줘보도록 하겠습니다.

<br>

```python

# Requirement: sort by name initial ascending, and by grades, descending
# Both won't work
sorted(grades, key=lambda x: (x[0][0], x[1]), reverse=True)
#  [('Zoe', 99), ('Zack', 97), ('Jennifer', 100), ('John', 95), ('Dee', 93), ('Don', 92), ('Aaron', 99), ('Abby', 94)]
sorted(grades, key=lambda x: (x[0][0], x[1]), reverse=False)
# [('Abby', 94), ('Aaron', 99), ('Don', 92), ('Dee', 93), ('John', 95), ('Jennifer', 100), ('Zack', 97), ('Zoe', 99)]
# This will do the trick
sorted(grades, key=lambda x: (x[0][0], -x[1]))
# [('Aaron', 99), ('Abby', 94), ('Dee', 93), ('Don', 92), ('Jennifer', 100), ('John', 95), ('Zoe', 99), ('Zack', 97)]
```

<br>

## **comparator를 이용한 클래스 정렬**

<br>

+ 리스트를 정렬할 때 리스트의 원소가 클래스라면 정렬하는 기준을 정해주어야 합니다.
+ 가장 유연하게 정할 수 있는 방법 중 하나인 `Comparator`를 사용하는 방법에 대하여 알아보겠습니다.

<br>

```python
sorted(iterable, key, reverse)
``` 

<br>

+ 위 함수를 이용하여 정렬을 합니다.
+ `iterable` 에는 탐색이 가능한 타입을 입력합니다. 대표적인 예로 정렬할 리스트를 넣으면 됩니다.
+ `key`에는 정렬에 사용할 `comparator`를 입력합니다.
+ `reverse`는 역으로 정렬할 것인지에 대한 파라미터로 필요시 사용하면 됩니다.

<br>

+ 여기서 중요한 `key`에 입력할 `comparator`에 대하여 알아보도록 하겠습니다.

<br>

```python
sorted(A, key=lambda x:foo(x))
```

<br>

+ A라는 리스트가 있다고 하겠습니다. A리스트의 원소는 클래스로 (name, age)의 값을 가지고 있다고 가정하겠습니다.
    + 예를들어 ``` A = [("JISU", 28), ("JINSOL", 30)] ```
+ 이 때, 나이 기준으로 정렬하고 싶으면 나이에 대한 기준을 주어야 합니다.

<br>

```python
def foo(x):
    return x.age
``` 

<br>

+ 위 코드와 같이 클래스의 어떤 값으로 기준을 주면 됩니다. 

<br>

```python
sorted(A, key=lambda x:foo(x))
```

<br>

+ 따라서 위와 같이 정의 하면 각 원소는 foo라는 함수를 통하여 어떤 값을 기준으로 가질 지 알게 됩니다.

<br>

## **all과 any**

<br>

- `all`을 이용하면 iterable한 객체의 값 중에 False 또는 0이 있으면 False를 반환합니다.
- 반면 모든 값이 True 또는 0이 아니어야 True를 반환합니다.
- iterable한 객체이므로 list, tuple, set, dictionary 모두 적용 가능하고 dictionary의 값에서는 key 값을 가지고 판단합니다.
- **empty string**과 **empty list**는 False로 간주합니다. 아래 코드를 참조하시기 바랍니다.

<br>

```python
# iterable - a list of booleans
all([True, True, True])
# True
all([True, False, False])
# False

# iterable - a list of integers (0 is considered false)
all([1, 2, 3])
# True
all([0, 1, 2])
# False

# iterable - a tuple of string (an empty string is considered false)
all(('AA', 'BB', 'CC'))
# True
all(('AA', 'BB', ''))
# False

# iterable - a nested list (an empty list is considered false)
all([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# True
all([[1, 2, 3], [4, 5, 6], []])
# False
```

<br>

- `any`를 이용하면 객체의 값 중에 True 또는 0이 아닌 값이 하나라도 있으면 True로 반환합니다.
- 반면 모든 값이 False 또는 0이어야 False를 반환합니다.

<br>

- `all`과 `any`를 사용하면 여러 조건이 있는 조건문을 좀 더 간단하게 사용할 수 있습니다. 다음을 참조하시기 바랍니다.

<br>

```python
# The typical ways
if a < 10 and b > 5 and c == 4:
    # do something
if a < 10 or b > 5 or c == 4:
    # do something
# Do these instead
if all([a < 10, b > 5, c == 4]):
    # do something
if any([a < 10, b > 5, c == 4]):
    # do something
```

<br>

## **lambda**

<br>

- 출처: https://offbyone.tistory.com/73
- 파이썬에서 `lambda` 는 런타임에 생성해서 사용할 수 있는 익명 함수 입니다. 파이썬에서 `lambda`의 역할은 혼자 쓰일 때라기 보다는 다양한 다른 함수와 같이 쓰일 때 큰 힘을 발휘합니다.
    - 예를 들어 `map`, `filter`, `reduce` 등이 있습니다.
- 먼저 `lambda`에 대한 간략한 소개와 그에 이어서 `map`, `filter`, `reduce`에 어떻게 lambda가 사용하는 지 예를 보여드리겠습니다.
- lambda는 일반적인 함수와 다르게 만들어진 곳에서 일시적으로 사용하고 버릴 수 있는 함수입니다.

<br>

```python
l = lambda x: x**2
print(l(3))
: 9 

f = lambda x, y: x + y

print(f(4, 4))
: 8
```

<br>

## **map**

<br>

- `map`은 입력 받은 **function**에 또다른 입력인 **iterable을 mapping 시켜주는 역할**을 합니다.
- map 함수와 lambda 함수가 함께 사용될 때 복잡한 기능들을 간단하게 구현할 수 있습니다.
- 먼저 map() 은 두 개 이상의 파라미터를 기본적으로 받습니다. 
- `첫번째 파라미터`는 **function**이고 `두번째 파라미터`부터는 **function에 적용할 iterable** 입니다. 여기서 iterable은 한번에 하나의 멤버를 반환할 수 있는 객체 입니다.
- return 값은 function에 의해 변경된 iterator 입니다

<br>

```python
r = map(function, iterable1, iterable2, ...)
```

<br>

- map을 이용한 예시를 한번 살펴보겠습니다.

<br>

```python
a = [1,2,3,4]
b = [17,12,11,10]
list(map(lambda x, y:x+y, a,b))
: [18, 14, 14, 14]
```

<br>

## **filter**

<br>

- filter함수는 두 개의 인자를 가집니다.

<br>

```python 
r = filter(function, iterable)
```

<br>

- `filter`에 인자로 사용되는 function은 처리되는 각각의 요소에 대해 Boolean 값을 반환합니다. 
- True를 반환하면 그 요소는 남게 되고, False 를 반환하면 그 요소는 제거 됩니다.

<br>

```
a = [1,2,3,4,5]
list( filter(lambda x: x % 2 == 0, a) )
: [2, 4]
```

<br>

## **reduce**

<br>

- reduce 함수는 두 개의 필수 인자와 하나의 옵션 인자를 가지는데 function 을 사용해서 iterable을 하나의 값으로 줄입니다.

<br>

```python
functools.reduce(function, iterable[, initializer])
```

<br>

- 예를 들어 reduce(function, [1,2,3,4,5]) 에서 list 는 [function(1,2),3,4,5] 로 하나의 요소가 줄고, 요소가 하나가 남을 때까지 reduce(function, [function(1,2),3,4,5]) 를 반복합니다.

<br>

```python
from functools import reduce
reduce(lambda x,y: x+y, [1,2,3,4,5])
: 15
```

<br>

## **slice**

<br>

- 파이썬의 `slice(start, stop, step)` 함수에 대하여 알아보도록 하겠습니다.
- 파이썬의 slice 함수를 이용하여 iterable한 자료 구조를 구간 별로 접근할 수 있습니다.
- 이 방법은 파이썬의 인덱싱 방법과 동일합니다. 다만 함수 형태로 사용 가능하다는 차이점이 있습니다.
- 예제를 살펴보면 다음과 같습니다.

<br>

```python
py_list = ['P', 'y', 't', 'h', 'o', 'n']
py_tuple = ('P', 'y', 't', 'h', 'o', 'n')

# contains indices 0, 1 and 2
slice_object = slice(3)
print(py_list[slice_object]) # ['P', 'y', 't']

# contains indices 1 and 3
slice_object = slice(1, 5, 2)
print(py_tuple[slice_object]) # ('y', 'h')    
```

<br>

- 만약 slice에 인자 하나만 입력하면 두번째 인자인 `stop`으로 인식 되며 처음부터 stop 지점전 까지 인덱싱 됩니다.

<br>

## **Counter**

<br>

- `Counter`는 `from collections import Counter`를 통해 사용 가능한 container 종류 중 하나입니다.
- Counter는 말 그대로 데이터의 갯수를 세는 자료 구조로 C++에서 `multiset`과 유사합니다. 사용방법은 다음과 같습니다.

<br>

```python
from collections import Counter
cnt = Counter()
for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
    cnt[word] += 1

print(cnt)
# Counter({'blue': 3, 'red': 2, 'green': 1})
```

<br>

- Counter의 대표적인 기능은 `.most_common(갯수)` 입니다. 위 예제에서 `cnt.most_common(2)`라고 하면 `[('blue', 3), ('red', 2)]`와 같은 결과가 나옵니다. 가장 빈도가 큰 순서대로 출력됩니다.
- Counter를 생성할 때, 대표적으로 다음과 같이 생성할 수 있습니다.

<br>

```python
c = Counter()                           # a new, empty counter
c = Counter('gallahad')                 # a new counter from an iterable (each chracter)
c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
c = Counter(cats=4, dogs=8)             # a new counter from keyword args
```

<br>

- Counter끼리 연산을 할 때, 다음과 같은 연산이 가능합니다. 두 Counter의 차를 구할 때, `subtract` 또는 `-` 연산자를 사용할 수 있습니다. 그 차이는 아래와 같습니다.
- 반대로 두 Counter의 합을 구할 때, `update` 또는 `+`를 사용할 수 있습니다.
- 교집합은 `&` 연산자를, 합집합은 `|` 연산자를 사용합니다.

<br>

```python
c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d)
print(c)
# Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
print(c - d)
# Counter({'a': 3})

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.update(d)
print(c)
# Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2})

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
print(c + d)
# Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2})

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
print(c & d)
# Counter({'a': 1, 'b': 2})
print(c | d)
# Counter({'a': 5, 'b': 4, 'c': 3, 'd': 2})
```

<br>

## **defaultdict를 이용한 multimap 사용**

<br>

- C++의 multiset 기능은 `Counter`를 이용하여 사용할 수 있습니다. 그러면 multimap 기능은 어떻게 사용할 수 있을까요?
- multimap에 정확하게 대응되는 자료구조는 없으나 `defaultdict`를 사용하면 유사하게 구현할 수 있습니다.
- 따라서 dictionary를 응용하여 구현하는 방법이 가장 간편합니다. 
- `defaultdict`은 dictionary 형태로 사용이 가능한 자료구조 입니다. 다만 일반 dictionary는 Key : Value 쌍에서 **Value의 타입을 고정해서 사용**할 수 있습니다. 먼저 defaultdict의 사용법을 알아보겠습니다.

<br>

```python
from collections import defaultdict
# 첫번째 케이스
md = defaultdict(list)
md[1].append('a')
md[1].append('b')
md[2].append('c')
md[1]
# ['a', 'b']
md[2]
# ['c']

# 두번째 케이스
md = defaultdict(set)
md[1].add('a')
md[1].add('b')
md[2].add('c')
md[1]
# {'a', 'b'}
md[2]
# {'c'}
```

<br>

- 위 예제와 같이 `default(데이터 타입)`와 같은 방법으로 객체를 생성할 수 있습니다. 첫번째 케이스에서는 Value 값으로 list를 사용하였고 두번째 케이스에서는 Value 값으로 set을 사용하였습니다.
- 물론 이 방식이 C++의 multimap과 완전히 동일하지 않고 단순히 dictionary 로도 구현할 수 있지만 Key, Value 쌍의 타입을 정해놓을 수 있기 때문에 좀 더 편하게 사용할 수 있어서 추천드립니다.

<br>

## **OrederedDict**

<br>

- 참조 : https://excelsior-cjh.tistory.com/98
- 파이썬에서는 배열처럼 사용할 수 있는 Dictionary가 있습니다. 바로 `collection` 패키지에 있는 `OrderedDict`입니다.
- OrderedDict는 dictionary와 거의 비슷하지만, 입력된 아이템들(items)의 순서를 기억하는 Dictionary 클래습입니다. 
- OrderedDict는 아이템들(items)의 입력 순서를 기억하기 때문에 sorted()함수를 사용하여 정렬된 딕셔너리(sorted dictionary)를 만들때 사용할 수 있습니다.

<br>

```python
from collections import OrderedDict
 
# 기본 딕셔너리 
d = {'banana': 3, 'apple': 4, 'pear': 1, 'orange':2}
 
# 키(key)를 기준으로 정렬한 OrderedDict
ordered_d1 = OrderedDict(sorted(d.items(), key=lambda t:t[0]))
print(ordered_d1)
# OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
 
# 값(value)를 기준으로 정렬한 OrderedDict
ordered_d2 = OrderedDict(sorted(d.items(), key=lambda t:t[1]))
print(ordered_d2)
#OrderedDict([('pear', 1), ('orange', 2), ('banana', 3), ('apple', 4)])
```

<br>

- dictionary의 성질을 가지면서 아이템의 순서를 기억하기 때문에 입력되는 순서가 중요합니다. 위 예제와 같이 key 또는 value를 기준으로 정렬한 상태로 OrderedDict을 만들면 OrderedDict의 목적에 맞게 잘 사용할 수 있습니다.

<br>

- OrderedDict에 값을 추가할 때에는 Dictionary의 방법과 똑같습니다. 아래 코드르 참조하시겠습니다.

```python
ordered_d1.update({'grape': 5}) # ordered_d1['grape'] = 5
print(ordered_d1)
# OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1), ('grape', 5)])
```

<br>

- `.update()` 또는 인덱스에 직접 입력 하는 방식 (`ordered_d1['grape'] = 5`)을 통하여 새로운 값을 입력할 수 있습니다. 물론 List와 마찬가지로 정렬 이후에 값을 추가하였기 때문에 정렬 순서는 깨지게 되며 가장 마지막 인덱스에 값이 추가됩니다.

<br>

- 반면 OrderedDict에 값을 삭제할 때에는 Stack 방식(LIFO) 처럼 사용할 수 있고 또는 Queue 방식 (FIFO) 처럼 사용할 수 있습니다.

<br>

```python
import operator
from collections import OrderedDict
 
# 기본 딕셔너리 
d = {'banana': 3, 'apple': 4, 'pear': 1, 'orange':2}
 
# 키(key)를 기준으로 정렬한 OrderedDict
ordered_d = OrderedDict(sorted(d.items(), key=operator.itemgetter(0), reverse=False))
print(ordered_d)
 
# popitem(last=True) 일경우 : LIFO(Last In Last Out)방식으로 pop, default는 True임
for i in range(len(ordered_d)):
    print(ordered_d.popitem(last=True))
    
print('='*50)
    
# 키(key)를 기준으로 정렬한 OrderedDict
ordered_d = OrderedDict(sorted(d.items(), key=operator.itemgetter(0), reverse=False))
print(ordered_d)
 
# popitem(last=False) 일경우 : FIFO(First In First Out)방식으로 pop
for i in range(len(ordered_d)):
    print(ordered_d.popitem(last=False))
    
'''
결과
OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
('pear', 1)
('orange', 2)
('banana', 3)
('apple', 4)
==================================================
OrderedDict([('apple', 4), ('banana', 3), ('orange', 2), ('pear', 1)])
('apple', 4)
('banana', 3)
('orange', 2)
('pear', 1)
```

<br>

## **bisect를 사용한 이진 탐색**

<br>

- C++의 lower_bound와 같은 기능을 이용하면 정렬된 배열(리스트)에서 이진 탐색으로 어떤 값을 찾을 수 있습니다.
- 정확한 기능을 설명하면 C++의 lower_bound 기능은 정렬된 배열에서 찾고자 하는 값의 인덱스를 반환합니다. 만약 찾고자 하는 값이 정확히 존재하면 그 값의 인덱스를 반환하고 만약 그 값이 배열에 없으면 그 값보다 큰 값 중 가장 작은 값의 인덱스를 반환합니다.
- 다음 예제를 통하여 살펴보겠습니다. 사용 할 함수는 `bisect.bisect_left(배열, 찾을 값, 시작 인덱스, 끝 인덱스)`와 `bisect.insort_left(배열, 삽입할 값, 시작 인덱스, 끝 인덱스)` 입니다. 인덱스의 범위는 `[시작 인덱스, 끝 인덱스)`입니다.

<br>

```python
import bisect 
a = [1,3,5,7,9]

# 찾고자 하는 값 2는 없으므로 2보다 크면서 가장 작은 값인 3의 인덱스인 1을 반환한다.
bisect.bisect_left(a, 2, 0, len(a))
# 1

# 찾고자 하는 값이 배열의 모든 값보다 크므로 배열의 마지막 인덱스 + 1 인 위치를 가리킨다.
bisect.bisect_left(a, 10, 0, len(a))
# 5

# 찾고자 하는 값이 배열의 가장 작은 값 보다 작으므로 배열의 첫번째 인덱스인 0의 위치를 가리킨다.
bisect.bisect_left(a, -1, 0, len(a))
# 0

# 정확히 찾고자 하는 값이 있는 경우 그 값의 인덱스를 반환한다.
bisect.bisect_left(a, 5, 0, len(a))
# 2

# 정렬된 배열에서 배열의 정렬 상태를 유지하면서 값을 삽입한다.
# 삽입 위치를 찾는 데에는 O(log(n))의 시간이 걸리지만 실제 삽입할 때, O(n)의 시간이 걸린다.
bisect.insort_left(a, 6, 0, len(a))

print(a)
# [1, 3, 5, 6, 7, 9]
```

<br>

## **for loop 스타일 (zip, range, enumerate, sorted, filter)**

<br>

- 파이썬에서 대표적으로 사용할 수 있는 for loop의 스타일에 대하여 정리해 보겠습니다.
- 크게 `zip`, `range`, `enumerate`, `sorted`, `filter` 를 이용한 스타일이 있습니다. 차례대로 알아보겠습니다.
- 먼저 `zip` 스타일 입니다. `zip` 스타일은 여러 종류의 저장된 데이터를 한 번에 사용하기 위해 사용됩니다.

<br>

```python
odds = [1,3,5,7,9]
evens = [2,4,6,8,10]
for oddnum, evennum in zip(odds,evens):
    print(oddnum)
    print(evennum)
```

<br>

- 다음으로는 `range`를 이용한 방법으로 주로 C언어에서 사용되는 타입과 같습니다.

<br>

```python
for i in range(10):
    print(i)    
```

<br>

- `enumerate`를 이용한 방법은 데이터와 그 데이터의 인덱스를 같이 사용할 때 사용합니다.
- 아래 코드에서 `i`는 기본적으로 0, 1, 2, ... 순서로 입력됩니다.
- 만약 시작 값을 바꾸고 싶으면 `start` 옵션을 주면 됩니다. 

<br>

```python

A = [10, 20, 30, 40, 50]
for i, data in ennerate(A, start=1):
    print(i, data)
```

<br>

- 만약 임시로 정렬된 데이터가 필요하다면 새로 할당할 필요 없이 `sorted`를 이용하여 for loop을 사용할 수 있습니다.

<br>

```python
l = [15,6,1,8]
for i in sorted(l):
    print(i)
    
for i in sorted(l,reverse = True):
    print(i)
```

<br>

- 이 때, 정렬하는 방법으로 `lambda` 식을 이용할 수도 있습니다.
    - `l.sort(key=lambda s: s[::-1])`

<br>

- 마지막으로 필요한 데이터만 for loop 에서 탐색하기 위하여 `filter` 함수를 사용하는 방법이 있습니다.

<br>

```python
people = [{"name": "John", "id": 1}, {"name": "Mike", "id": 4}, {"name": "Sandra", "id": 2}, {"name": "Jennifer", "id": 3}]
for person in filter(lambda i: i["id"] % 2 == 0, people):
...     print(person)
... 
{'name': 'Mike', 'id': 4}
{'name': 'Sandra', 'id': 2}
```

<br>

## **deque은 사이즈 관리를 자동으로 한다.**

<br>

- 파이썬의 deque에서도 queue와 마찬가지로 최대 사이즈를 지정할 수 있습니다.
- deque의 기본적인 자료 구조 특성으로 인한 편리함 이외에 파이썬에서 제공하는 편리한 기능은 최대 사이즈를 자동으로 관리해 준다는 것입니다.
- 예를 들어 deque의 최대 크기를 3이라고 하였을 때, 4개의 데이터를 넣으면 4번째 데이터를 넣을 때 내부적으로 첫번째 데이터를 pop 하는 것입니다.
- 즉 FIFO 구조에 맞게 deque의 최대 크기를 초과할 때 Firtst In 된 데이터를 Pop 하는 기능이 구현되어 있다는 것입니다.
- 따라서 deque에 최대 크기를 지정한 경우 따로 예외처리를 하지 않아도 되는 편리함이 있습니다.

<br>

```python
from collections import deque
dq = deque(maxlen=3)
dq.append(1)
dq.append(2)
dq.append(3)
dq.append(4)

print(dq)
: deque([2, 3, 4], maxlen=3)
```

<br>

## **파이썬 프로그램 정상 종료 시키기**

<br>

- 파이썬에서 프로그램을 정상 종료 시키려면 `sys` 라이브러리를 통하여 종료시키면 안정적으로 종료됩니다.

<br>

```python
import sys
sys.exit()
```

<br>

## **argparse 사용법**

<br>

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Enter the input')
args = parser.parse_args()
print(args.input)
```

<br>

## **문자열 검색 : startswith, endswith, in, find**

<br>

- 파이썬의 문자열 내부를 검색할 때 대표적으로 사용하는 함수에 대하여 알아보도록 하겠습니다.
- `startswith` : 특정 prefix로 시작하는 지 찾습니다.

<br>

```python
>> s = 'this is string.'
>> s.startswith('this')
True
```

<br>

- `endswith` : 특정 suffix로 끝나는 지 찾습니다.

<br>

```python
>> s = 'this is string.'
>> s.endswith('string.')
True
```

<br>

- `in`을 이용하면 특정 문자열을 포함하고 있는 지 확인합니다.

<br>

```python
>> s = 'this is string.'
>> 'is' in s
True
```

<br>

- `find`: 특정 문자열이 위치하는 인덱스를 찾습니다. 찾으면 그 위치의 인덱스를 반환하고 찾지못하면 -1을 반환합니다.

<br>

```python
>> s = 'this is string.'
>> s.find('string')
8
>> s.find('no')
-1
```

<br>

## **디렉토리(+ 파일) 복사하기**

<br>

- 어떤 디렉토리 내부의 모든 디렉토리 구조와 파일을 복사하고 싶으면 다음 코드를 사용하면 됩니다.
- 아래 코드는 현재 경로에 있는 `dir1` 이라는 폴더 전체를 `dir2`로 복사합니다. 내부의 폴더 구조와 파일 까지 모두 복사합니다.

<br>

```python
import shutil
src = './dir1'
dest = './dir2'
shutil.copytree(src, dest)
```

<br>

## **디렉토리(- 파일) 복사하기**

<br>

- 디렉토리 내부의 파일은 무시하고 디렉토리 구조만 복사하려면 다음 함수를 사용하면 됩니다.
- 앞에서 다룬 `shutil.copytree` 함수에서 `ignore` 옵션으로 아래 함수를 넘겨주면 파일은 복사하지 않습니다.

<br>

```python
import shutil

def OnlyDirectoryCopy(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

src = './dir1'
dest = './dir2'
shutil.copytree(src, dest, ignore=OnlyDirectoryCopy)
```

<br>

## **특정 문자를 기준으로 split**

<br>

- 특정 문자를 기준으로 문자열을 split 하는 방법은 문자열 자체의 split 함수를 이용하는 방법과 regular expression을 이용하는 2가지 방법이 있습니다.

<br>

```python
>> s = 'image.png'
>> s.split('.')
['image', 'png']
```

<br>

- 반면 split 해야 하는 문자의 기준이 많아지면 regular expression을 이용해서 처리하면 됩니다.

<br>

```python
import re

>> s = 'A_B_C.png'
>> re.split('[_.]', s)
['A', 'B', 'C', 'png']
```

<br>

## **숫자에 0을 채워서 출력**

<br>

- `str.zfill(숫자)` 를 통해 문자열의 공백을 0으로 채울 수 있습니다.
- `"%03d"`와 같은 형태로 출력하면 정수에 공백을 추가하여 출력할 수 있습니다.

<br>

## **문자열 양 끝의 공백 문자 없애기**

<br>

- 문자열을 다루다 보면 문자열 양 끝에 공백 문자들이 오는 경우가 있습니다. 스페이스, 탭, 뉴라인 등등이 올 수 있습니다.
- 이 경우 `strip`, `lstrip`, `rstrip` 함수를 이용하면 자연스럽게 공백 문자들을 제거할 수 있습니다. 
- `strip`의 경우 왼쪽, 오른쪽 끝 방향 모두 공백 문자를 제거하고, `lstrip`은 왼쪽, `rstrip`은 오른쪽 끝의 공백 문자를 제거합니다. 예를 들어 다음과 같습니다.

<br>

```python
a = '\n \t aaaa     '
a.strip()
>>> 'aaaa'

a.lstrip()
>>> 'aaaa     '

a.rstrip()
>>> '\n \t aaaa'

```

<br>

## **os 관련 함수 모음**

<br>

- 파이썬에서 현재 다루고 있는 컴퓨터를 접근하여 파일, 디렉토리 등을 조작할 때, os의 라이브러리 등을 사용할 수 있습니다.
- 이 글에서는 os 라이브러리를 이용하여 할 수 있는 기능들에 대하여 간략하게 정리해 보도록 하겠습니다. 먼저 자주 사용할 수 있는 함수 들을 쭉 적은 다음에 차례 대로 설명을 달아보겠습니다.

<br>

```python
os.environ['HOME'] # 'HOME' 대신에 다양한 시스템 정보들이 들어갈 수 있습니다.
os.environ.get('HOME')
os.path.isfile('filename')
os.path.isdir('dirname')

# path가 존재하는 지 확인
os.path.exists(path) 

# 현재 경로에 folder 생성
os.mkdir(folder_name) 

# 지정된 path에 folder 생성하며 path에 포함된 branch 까지 생성
# 예를 들어 path가 ./A/B/C 이고 B/C가 없다면 마지막 C를 생성할 때까지 중간에 필요한 폴더 계층을 모두 생성합니다.
# mkdir과 유사하지만 중간에 sub directory 까지 한번에 만들어 줍니다.
os.makedirs(path) 

# 현재 디렉토리의 절대 경로를 출력함
os.path.abspath(os.getcwd())
# 경로 변경
os.chdir(path)

# cmd 명령어 입력
command = "ls -al"
os.system(command)

# 작업하고 있는 디렉토리 변경
os.chdir(path)

# 현재 프로세스의 작업 디렉토리 얻기
os.getcwd()

# 파일이나 디렉토리 지우기
os.remove(file_path or path)

# 파일의 상대 경로를 절대 경로로 바꾸는 함수
os.path.abspath(file_path)

# 주어진 경로의 파일이 있는지 확인하는 함수
os.path.exists(file_path)

# 현재 디렉토리 얻기
os.curdir()

# 부모 디렉토리 얻기
os.pardir()

# 디렉토리 분리 문자 얻기. . windows는 \ linux는 / 를 반환합니다.
os.sep() 

# 파일명만 추출
os.path.basename(file_path)

# 디렉토리 경로 추출
os.path.dirname(file_path)

# 마지막 디렉토리 폴더명 추출
path = "folder_A/folder_B/folder_C/folder_D"
os.path.basename(os.path.normpath(path))
# folder_D

# 경로와 파일명을 분리
os.path.split(file_path)

# 드라이브명과 나머지 분리 (MS Windows의 경우)
os.path.splitdrive(file_path)

# 확장자와 나머지 분리
os.path.splitext(file_path)

# 파일 사이즈 계산
MB = 1024 ** 2
GB = 1024 ** 3
# Mega Byte 단위로 표현
os.path.getsize(file_path) // MB
# Giga Byte 단위로 표현
os.path.getsize(file_path) // GB

```

<br>

- `os.environ`은 현재 컴퓨터의 정보 들을 dictionary 형태로 저장합니다. 예를 들어 HOME, APPDATA, USERNAME 등의 정보와 환경 변수에 등록된 PATH 등이 key 값으로 저장되어 있습니다. 
- 따라서 os.emnviron['HOME']과 같이 입력하면 그 value 값들이 출력됩니다. 하지만 dictionary의 문법에 따라 만약 key 값으로 등록되지 않은 정보를 읽어들일 때, 에러가 발생하므로 코드의 안정성을 위해 다음과 같이 사용하길 추천 드립니다.
- `os.environ.get('HOME')` : 이 경우 key 값이 없으면 `None`을 출력하고 Key 값이 있으면 그 value를 출력합니다.

<br>

- file 및 directory가 존재하는 지 유무를 확인하려면 다음 명령어를 이용합니다.
- `os.path.isfile('filename')` 와 `os.path.isdir('dirname')`

<br>

- 파이썬에서 시스템 커맨드를 입력하려면 `os.system(command)`를 이용하면 됩니다.
- 시스템 상에서 사용할 수 있는 유용한 명령어들을 파이썬 함수 내에서 사용할 수 있기 때문에 굉장히 유용하게 사용할 수 있습니다.

<br>

## **glob을 이용한 폴더 및 파일 접근**

<br>

- `glob`은 파이썬을 이용하여 파일을 탐색할 때, 굉장히 유용한 라이브러리입니다. 보통 `os`의 `os.listdir()`과 같은 방식을 많이 이용합니다. 다만 검색 조건이 필요할 때에 `glob`을 사용하면 쉽게 구현할 수 있습니다.
- 아래는 현재 위치 기준에서 2-depth 아래에 있는 파이썬 파일을 찾는 코드 입니다. 핵심은 `path`에 사용된 문자열이며 문자열의 정규식을 어떻게 사용하는 지에 따라서 찾는 조건이 달라집니다. glob의 기본적인 사용방법은 다음과 같습니다.
    - `glob.glob(wildcard_type_path)` : 정규식 형태의 스타일로 파일 목록을 얻을 수 있습니다.

<br>

```python
import glob

path = "./*/*.py"
file_list = glob.glob(path)
file_list_py = [file for file in file_list]
# file_list_py = [file for file in file_list if file.endswith(".py")]

print ("file_list_py: {}".format(file_list_py))
```

<br>

## **pickle 사용 방법**

<br>

- `pickle`은 파이썬 전용으로 편리하게 데이터를 저장 및 불러오기 위한 도구입니다. 즉, 파이썬의 모든 것들을 그냥 `binary` 형태로 저장할 수 있습니다. 
- 예를 들어 파이썬에서 사용한 어떤 객체를 저장했다가 나중에 다시 쓰고 싶은데, 그냥 바로 이 객체를 저장할 때 사용할 수 있습니다.
- 또한 `pickle`은 binary 파일 자체로 저장하여 압축률도 높기 때문에 저용량으로 압축해서 저장할 수 있습니다.
- 따라서 데이터를 저장해야할 때, 1) 객체 자체로 저장하고 싶고 2) 파이썬에서 사용할 것이고 3) 저용량으로 압축하고 싶다면 `pickle`을 쓰길 강력하게 추천드립니다.
- 사용하실 때에는 아래 코드를 이용하셔서 사용 하시면 도움이 됩니다.

<br>

```python
import pickle

# 샘플 데이터를 생성하기 위해 numpy를 이용해 보겠습니다.
# 이 때, 어느 데이터든 상관없습니다. numpy 말고 list, dictionary 등등 아무 거나 사용하셔도 됩니다.
import numpy as np
A = np.random.rand(10, 10, 10)
 
## Save pickle
with open("data.pickle","wb") as fw:
    pickle.dump(A, fw)
 
## Load pickle
with open("data.pickle","rb") as fr:
    B = pickle.load(fr)

```

<br>

## **exec을 이용한 문자열로 코드 실행**

<br>

- 파이썬 코드를 짜다보면 가끔씩 문자열을 이용하여 코드를 실행할 필요가 있는 경우가 생깁니다.
- 예를 들어 변수를 선언할 때, 변수 명을 입력 받아서 선언해야 하는 경우가 있다면 다음과 같이 할 수 있습니다.

<br>

```python
>> exec("%s = %d" % ("x", 10))
>> print(x)

10
```

<br>

- 만약 local에서 사용한다면 반드시 `globals()` 또는 `locals()` 옵션을 넣어서 명확하게 해주어야 하며 함수 외에서도 사용하려면 `globals()`를 이용해야 합니다.

<br>

```python
def func():
    exec("a = 5", globals())
    
>> print(a)
5
```

<br>

## **type과 isinstance를 통한 데이터 타입 확인**

<br>

- 파이썬에서 어떤 데이터의 타입을 이용하여 조건문을 사용하고 싶을 때, `isinstance` 함수를 사용할 수 있습니다. 
- 사용 방법은 `isinstance(변수/상수, 타입명)` 형태로 사용하며 예를 들어 다음과 같이 사용할 수 있습니다.

<br>

```python
>> isinstance(1, int)
True
```

<br>

- 여기서 중요한 것은 내가 찾고자 하는 변수 또는 상수의 타입명을 어떻게 찾을 수 있을까 입니다. 이것은 `type`을 통하여 찾을 수 있습니다. 따라서 다음과 같이 이용하면 됩니다.

<br>

```python
temp_str = "sample"
temp_int = 1
temp_float = 1.23

class Temp():
    pass    
temp_class = Temp()

>> type(temp_str)
str
>> isinstance(temp_str, str)
True

>> type(temp_int)
int
>> isinstance(temp_int, int)
True

>> type(temp_float)
float
>> isinstance(temp_float, float)
True

>> type(temp_class)
__main__.Temp
isinstance(temp_class, Temp)
>> True
```

<br>

## **dir을 통한 module 리스트 확인**

<br>

- `dir` 함수를 이용하면 특정 모듈의 attribute를 확인할 수 있습니다. 

<br>

```python
>> dir('.')

['DirEntry',  'F_OK',  'MutableMapping',  'O_APPEND',  'O_BINARY',  'O_CREAT', ...]

class Test():
    def func1():
        pass
    def func2():
        pass

test = Test()
>> dir(test)
['__class__','__delattr__','__dict__','__dir__','__doc__',
'__eq__','__format__','__ge__','__getattribute__','__gt__',
'__hash__','__init__','__init_subclass__','__le__','__lt__',
'__module__','__ne__','__new__','__reduce__','__reduce_ex__',
'__repr__','__setattr__','__sizeof__','__str__','__subclasshook__',
'__weakref__','func1','func2']
```

<br>

## **enumerate 함수의 응용**

<br>

- `enumerate`는 주로 for-loop에서 index를 활용하기 위해서 많이 사용되곤 합니다.
- 이번 글에서는 `enumerate`를 어떻게 응용하여 사용할 수 있을 지 알아보겠습니다.\
- 먼저 아래와 같이 enumerate를 이용하여 generator를 만들 수 있습니다. 내용을 확인할 때에는 `next` 함수를 사용하여 접근할 수 있습니다.

<br>

```python
alphabets = ['A', 'B', 'C', 'D']

# create an enumerate object
enumberate_alphabet = enumerate(alphabets)

# obtain the next tuple from the enumerate object
>> next(enumerate_object)
(0, 'A')

>> next(enumerate_object)
(1, 'B')

>> next(enumerate_object)
(2, 'C')

>> next(enumerate_object)
(3, 'D')
```

<br>

- enumerator를 이용하여 다양한 형태로 `list`를 만들 수 있습니다.

<br>

```python
alphabets = ['A', 'B', 'C', 'D']
enumberate_alphabet = enumerate(alphabets)
>> list(enumberate_alphabet)
[(0, 'A'), (1, 'B'), (2, 'C'), (3, 'D')]

alphabets = [index_alphabet for index_alphabet in enumerate(alphabets, start=1)]
>> print(alphabets)
[(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D')]
```

<br>

## **counting 방법**

<br>

- 문자열을 counting을 할 수 있는 대표적인 방법은 문자열 내장 함수를 사용하거나 또는 `Counter`를 사용하는 방법이 있습니다.
- 먼저 문자열 내장 함수를 사용해서 카운팅 해보겠습니다. 카운팅 하기 전에 `set`을 이용하여 중복 문자열을 제거하여 카운팅 할 문제열 리스트를 만듭니다.

<br>

```python
words = ['an', 'boy', 'girl', 'an', 'boy', 'dog', 'cat', 'Dog', 'CAT',
 'an','GIRL', 'AN', 'dog', 'cat', 'cat', 'bag', 'BAG', 'BOY', 'boy', 'an']
unique_words = {x.lower() for x in set(words)}
for word in unique_words:
    print(f"* Count of {word}: {words.count(word)}")
# * Count of cat: 3
# * Count of bag: 1
# * Count of boy: 3
# * Count of dog: 2
# * Count of an: 5
# * Count of girl: 1
```

<br>

- 이번에는 `Counter`를 이용해서 카운팅 해보겠습니다.

<br>

```python
from collections import Counter

word_counter = Counter(x.lower() for x in words)
print("Word Counts:", word_counter)
# Word Counts: Counter({'an': 5, 'boy': 4, 'cat': 4, 'dog': 3, 'girl': 2, 'bag': 2})
print("Most Frequent:", word_counter.most_common(1))
# Most Frequent: [('an', 5)]
print("Most Frequent:", word_counter.most_common(2))
# Most Frequent: [('an', 5), ('boy', 4)]
```

<br>

- 위 예제를 보면 `Counter` 객체를 이용하여 문자열을 카운팅 하고 `mose_common()` 함수를 통해 가장 빈도수가 높은 문자열 순으로 확인할 수 있습니다.

<br>

## **디렉토리 상위 레벨 패키지 import**

<br>

- 파이썬에서는 현재 위치의 하위 경로의 패키지를 import 하는 것이 원칙입니다.
- 하지만 경우에 따라서는 실행 되는 파이썬 파일의 상위 경로의 패키지를 접근해야 하는 경우가 있습니다.

<br>

```
├─A
│  └─AA
│          aa.py
│          
└─B
        b.py
```

<br>

- 현재 working directory가 `AA`라고 하겠습니다.
- `aa.py`를 import하려면 `from AA import aa`라고 하면 됩니다. 반면 `bb.py`를 import 하려면 어떻게 해야 할까요?
- 일반적인 접근 방법으로는 접근이 안됩니다. `from`에서 상위 디렉토리로 접근이 안되기 때문입니다.
- 이런 경우 `sys.path.append()`를 이용하여 import 하는 위치를 바꿔줄 수 있습니다. 아래 코드를 참조하여 `b.py`를 import 할 수 있습니다.

<br>

```python
import sys
sys.path.append("..") # 상위 디렉토리 접근
from B import bb # 이동된 디렉토리에서는 B가 하위디렉토리 이기 때문에 접근 가능하다.
```

<br>

<br>

## **리스트를 딕셔너리로 변환**

<br>

- 리스트를 딕셔너리로 변환할 때, 단순히 반복문을 이용하여 저장하는 것 외에도 파이썬 문법을 응용하여 편하게 변환할 수 있습니다.
- 먼저 리스트의 각 값을 key, 입력된 순서를 value로 딕셔너리를 구성해 보겠습니다.

<br>

```python
A = ["a", "b", "c", "d"]
A_dict = {A[i] : i for i in range(len(A))}
# {'a': 0, 'b': 1, 'c': 2, 'd': 3}
```

<br>

- 만약 Key, Value 쌍이 있는 리스트가 있다면 다음과 같이 간단하게 만들 수 있습니다.

<br>

```python
A = ["a", "b", "c", "d"]
B = [1,2,3,4]
AB_dict = dict(zip(A, B))
# {'a': 1, 'b': 2, 'c': 3, 'd': 4}
```

<br>

- 아래 코드의 list_of_tuple 또는 list_of_list와 같은 형태의 데이터가 있으면 바로 딕셔너리로 변환 가능합니다.

<br>

```python
list_of_tuple = [("A", 1), ("B", 2), ("C", 3), ("D", 4)]
list_of_list = [["A", 1], ["B", 2], ["C", 3], ["D", 4]]
dict_tuple = dict(list_of_tuple)
# {'A': 1, 'B': 2, 'C': 3, 'D': 4}
dict_list = dict(list_of_list)
# {'A': 1, 'B': 2, 'C': 3, 'D': 4}
```

<br>

## **유니크한 리스트 생성 방법**

<br>

- 파이썬 기본 패키지에는 유니크 함수가 없습니다. 따라서 리스트에서 중복값을 제거 하려면 `set`을 사용해서 중복값을 제외하면 간단하게 구현할 수 있습니다.

<br>

```python
a = [1,1,2,2,3,3,4,4,5,5]
a_unique = list(set(a))
# a = [1,2,3,4,5]
```

<br>

## **파이썬 실행 경로 추가**

<br>

- 파이썬에서 어떤 임의의 특정 경로를 코드 상에서 사용할 수 있도록 하고 싶으면 `sys.path.append()`를 통하여 경로를 추가할 수 있습니다.
- 추가된 경로의 어떤 파일이나 파이썬 패키지등을 바로 접근할 수 있으므로 상당히 편한 방법입니다. 사용법은 다음과 같습니다.

<br>

```python
import sys
path = "path/to/the/file"
sys.path.append(path)
```

<br>

## **문자열을 이용하여 패키지 import**

<br>

- 일반적으로 코드의 시작부에 필요한 패키지들을 입력해 놓지만 경우에 따라서는 입력 받은 문자열을 이용하여 유동적으로 패키지를 import 해야할 경우가 있습니다.
- 예를 들어 numpy를 import할 때 단순히 `import numpy as np`와 같이 입력하지만 `numpy`라는 문자열을 입력받아서 패키지를 import 하고 `np`라는 변수로 받고 싶을 때 다음과 같이 사용할 수 있습니다.

<br>

```python
import importlib
np = importlib.import_module('numpy')
```

<br>

- 위 코드와 같이 `np`를 리턴 받으면 일반적으로 사용하는 `np.함수()`와 같은 형태로 사용 가능합니다.

<br>

## **Dictionary와 JSON**

<br>

- `JSON`은 **JavaScript Object Notation**의 줄임말로 현재 많이 사용하고 있는 데이터 포맷 중 하나입니다. 단순히 JSON은 JSON에서 정한 규칙을 이용하여 문자열로 데이터를 나타냅니다. 이 규칙이 Dictionary와 비슷하여 파이썬에서는 Dictionary를 JSON으로 변환하거나 JSON을 Dictionary로 변환하는 기능을 제공합니다.
- 그러면 코드를 통해서 어떻게 Dictionary를 JSON으로 변환하고 반대로 JSON을 Dictionary로 변환하는 지 살펴보겠습니다.
- 먼저 살펴볼 기능은 `json.dumps(Object)` 입니다. 여기서 `dumps = dump + string`입니다. 즉, JSON 형태로 변환하되 문자열로 변환하는 기능을 뜻합니다. Object에는 Dictionary 뿐 아니라 list, tuple 등 어떤 object도 들어올 수 있으나 JSON의 목적에 맞게 Dictionary를 사용해 보도록 하겠습니다.
- 이 때, 영어 이외의 글자 (한글)를 입력하기 위해서는 `open()` 함수에서 `encoding=utf-8` 을 입력해야 영어 이외의 문자가 정상적으로 입/출력이 됩니다. 특히 쓰기를 할 때에는 `json.dump()`에서 `ensure_ascii=False` 옵션을 주어야 ASCII 이외의 문자로 저장이 되니 이 점을 주의하시면 됩니다.
    - json 파일 쓰기 : `open(filename, "w", encoding='utf-8')`,  `json.dump(..., ensure_ascii=False)`
    - json 파일 읽기 : `open(filename, "r", encoding='utf-8')`

<br>

```python
import json
Person = {
    'Name' : "Jinsol",
    'Man': True,
    "age" : 30,
    'friend': ["jisu", "suji"],
    "Pet" : None
}

print(Person)
# {'Name': 'Jinsol', 'Man': True, 'age': 30, 'friend': ['jisu', 'suji'], 'Pet': None}

print(type(Person))
# dict

json_string = json.dumps(Person)
print(json_string)
# {"Name": "Jinsol", "Man": true, "age": 30, "friend": ["jisu", "suji"], "Pet": null}
print(type(json_string))
# str
```

<br>

- `json.dupmps()`에서 사용할 수 있는 대표적인 인자는 `indent`와 `sort_keys` 입니다. `indent`는 JSON을 출력할 때 정렬 포맷의 indent 크기를 뜻하고 `sort_keys=True`로 두면 Key 기준으로 정렬하게 됩니다.

<br>

```python
json_string = json.dumps(Person, indent=4, sort_keys=True)
print(json_string)

# {
#     "Man": true,
#     "Name": "Jinsol",
#     "Pet": null,
#     "age": 30,
#     "friend": [
#         "jisu",
#         "suji"
#     ]
# }

```

<br>

- 그 다음으로 알아볼 기능은 `json.dump()`입니다. `dumps()`는 문자열을 출력한 반면 `dump()`는 파일 형태로 출력합니다. 
- 사용 방법은 `json.dump(Object, I/O, options)`과 같습니다.

<br>

```python
import json
Person = {
    'Name' : "Jinsol",
    'Man': True,
    "age" : 30,
    'friend': ["jisu", "suji"],
    "Pet" : None
}

with open("file.json", "w", encoding='utf-8') as fp:
    json.dump(Person, fp, indent=4, ensure_ascii=False)
```

<br>

- 위 코드는 Person이라는 Object를 file.json 파일에 저장합니다. 

<br>

- 이번에는 JSON 파일을 불러오는 방법에 대하여 알아보겠습니다. 불러오는 데이터는 크게 이미 저장된 JSON 파일 또는 문자열이 있습니다. 
- JSON을 생성할 때, `json.dump()`, `json.dumps()`를 사용하여 각각 파일 또는 문자열로 JSON을 생성하였습니다.
- 이와 유사하게 `json.load()`, `json.loads()`를 사용하여 각각 파일 또는 문자열을 불러와서 `Dictionary` 형태로 만듭니다.

<br>

```python
# json.load를 이용하여 json 파일을 읽는 예시
with open("file.json","r", encoding='utf-8') as fp:
    data = json.load(fp)

# json.loads를 이용하여 json 문자열을 읽는 예시
print(json_string)
# {
#     "Man": true,
#     "Name": "Jinsol",
#     "Pet": null,
#     "age": 30,
#     "friend": [
#         "jisu",
#         "suji"
#     ]
# }

json_dict = json.loads(json_string)
print(type(json_dict))
# dict
```

<br>

## **Dicionary의 최대, 최소 value 찾기**

<br>

- Dictionary의 모든 Key:Value 쌍에서 최대 또는 최소 Value에 해당하는 쌍을 찾기 위해서는 파이썬의 기본 max, min 내장 함수에 lambda 식으로 조건을 주어서 해결할 수 있습니다.

<br>

```python
a = {"a" : 1, "b" : 2, "c" : 3, "d":4}
print(max(a, key=lambda k : a[k]))
# d
print(min(a, key=lambda k : a[k]))
# a
```

<br>

## **Dictionary로 구성된 list 정렬 방법**

<br>

- 다음과 같은 딕셔너리를 담고 있는 리스트가 있을 때, 어떤 `key`를 기준으로 정렬하고 싶으면 다음과 같이 정렬할 수 있습니다.

<br>

```python
people = [
    { 'name': 'John', "age": 64 },
    { 'name': 'Janet', "age": 34 },
    { 'name': 'Ed', "age": 24 },
    { 'name': 'Sara', "age": 64 },
    { 'name': 'John', "age": 32 },
    { 'name': 'Jane', "age": 34 },
    { 'name': 'John', "age": 99 },
]

import operator

people.sort(key=operator.itemgetter('age'))
people.sort(key=operator.itemgetter('name'))

# [{'name': 'Ed', 'age': 24},
#  {'name': 'Jane', 'age': 34},
#  {'name': 'Janet', 'age': 34},
#  {'name': 'John', 'age': 32},
#  {'name': 'John', 'age': 64},
#  {'name': 'John', 'age': 99},
#  {'name': 'Sara', 'age': 64}]
```

<br>

## **Dictionary 합치는 방법**

<br>

- Dictionary를 merge하는 방법은 크게 2가지 방법이 있습니다. 3.9 버전 이하 까지는 `*`를 이용하여야 하고 3.9 버전 부터는 or 연산자로 합칠 수 있습니다. (참조 : [파이썬에서의 asterisk 사용 방법](#파이썬에서의-asterisk-사용-방법-1))
- 사용 방법은 아래를 참조하시기 바랍니다.

<br>

```python
dict1 = { 'a': 1, 'b': 2 }
dict2 = { 'b': 3, 'c': 4 }
merged = { **dict1, **dict2 }

print (merged)
# {'a': 1, 'b': 3, 'c': 4}

# Python >= 3.9 only
merged = dict1 | dict2

print (merged)
# {'a': 1, 'b': 3, 'c': 4}
```

<br>

## **copy를 이용한 deepcopy**

<br>

- 어떤 객체를 deepcopy 하기 위한 범용적인 방법으로 다음과 같습니다.

<br>

```python
from copy import deepcopy
object2 = deepcopy(object1)
```

<br>

## **파일 목록 얻기**

<br>

```python


# 지정된 디렉토리의 전체 파일 목록을 얻을 수 있다. 
os.listdir(path)
```

<br>

## **zipfile을 이용한 압축 풀기**

<br>

- 파이썬에서 압축 파일을 해제하려면 `zipfile`을 이용하여 해제할 수 있습니다. 코드는 아래와 같고 아래 코드는 압축파일을 해제하고 압축파일을 삭제하는 것 까지 적용하였습니다.
- 참고로 아래 코드의 `dest_path`에 압축 파일을 해제할 때, 현재 directory가 없다고 하더라도 자동적으로 생성하여 압축 해제한 파일을 풀어놓으므로 사전에 폴더 구조를 미리 만들 필요 없습니다.

<br>

```python
import zipfile
import os

def extract_zipfile(src_file_path, dest_path):
    with zipfile.ZipFile(src_file_path,"r") as zip_ref:
        zip_ref.extractall(dest_path)
    os.remove(src_file_path)

src_file_path = "src_path/.../.../file.zip"
dest_path = "dest_path/../../"
extract_zipfile(src_file_path, dest_path)
```

<br>

# **---함수의 응용---**

<br>

## **file read/write 방법**

<br>

- 먼저 텍스트 파일을 읽는 코드 입니다.

<br>

```python
# 읽을 파일의 경로와 파일 이름을 입력합니다.
filepath = "C://"
filename = "test.txt"

# 텍스트 파일을 입력 받기 위한 stream을 open 합니다.
with open(filepath + '/' + filename, 'r') as file_read:
    while (1):
        # 텍스트에서 1줄을 읽습니다.
        line = file_read.readline()
        # 개행 문자를 찾고 그 인덱스를 저장합니다.
        # 개행 문자가 가장 처음에 있으면 0이 저장됩니다.
        try:
            valid = line.index('\n')
        # 개행 문자가 없으면 입력받은 텍스트의 길이를 저장합니다.
        except:
            valid = len(line)
        # 만약 line이 없으면 valid == 0 또는 line은 있으나 개행문자만 있으면 valid == 0
        # 만약 line은 있으나 개행문자가 없으면 valid > 0
        # 만약 line이 있고 개행문자도 있으면 valid > 0
        # 따라서 valid > 0 경우만 parsing 실행
        if valid:
            # parsing 구분자로 parsing 처리
            line_to_list = line.split(' ')       

            # 아래 parsing 처리한 리스트를 처리하는 코드 필요
            ############## 코드 ##############
            ############## 코드 ##############
        else:
            break
```

<br>

- 위 코드를 보면 `file_read`를 `with`문을 통해서 여는 방식을 통하여 파일 읽기를 사용하였습니다.
- 보통 파일을 읽는 경우 한번에 전부 읽는 경우가 대다수이기 때문에 `with` 구문을 이용하여 한번에 파일을 모두 읽고 파일 스트림을 닫는 방식이 좋습니다.

<br>

- 위 코드를 함수 형태로 다시 정의하면 다음과 같습니다. 각 행 별로 parsing한 결과를 리스트 형태로 저장합니다.

<br>

```python
def FileToList(filepath, filename, sep=','):
    # 텍스트 파일을 입력 받기 위한 stream을 open 합니다.
    with open(filepath + '/' + filename, 'r') as file_read:
        file_to_list = []
        while (1):
            # 텍스트에서 1줄을 읽습니다.
            line = file_read.readline()
            # 개행 문자를 찾고 그 인덱스를 저장합니다.
            # 개행 문자가 가장 처음에 있으면 0이 저장됩니다.
            try:
                valid = line.index('\n')
            # 개행 문자가 없으면 입력받은 텍스트의 길이를 저장합니다.
            except:
                valid = len(line)
            # 만약 line이 없으면 valid == 0 또는 line은 있으나 개행문자만 있으면 valid == 0
            # 만약 line은 있으나 개행문자가 없으면 valid > 0
            # 만약 line이 있고 개행문자도 있으면 valid > 0
            # 따라서 valid > 0 경우만 parsing 실행
            if valid:
                # parsing 구분자로 parsing 처리
                line_to_list = line.split(sep)                
                # 아래 parsing 처리한 리스트를 처리하는 코드 필요
                ############## 코드 ##############
                ############## 코드 ##############
                file_to_list.append(line_to_list)
            else:
                break
    return file_to_list
```

<br>

```python
# 텍스트 파일을 출력 하기 위한 stream을 with를 통하여 open 합니다.
with open(filepath + '/' + filename, 'r') as file_write:
    s = 'contents...'
    file_write.write(s)

# 텍스트 파일을 출력 하기 위한 stream을 open 합니다.
file_write = open(filepath + '/' + filename, 'w')
s = 'contents...'
file_write.write(s)
file_write.close()
```

<br>

- 반면 파일을 쓸 때에는 한번에 모두 쓰는 경우도 있는 반면에 상황에 따라서 한번에 쓰지 못하는 경우도 생깁니다.
- 파일 스트림 접근은 최소하 하는 것이 좋습니다. 따라서 한번에 파일 스트림을 쓸 경우 `with`문을 이용한 파일 쓰기를 사용하고 그렇지 않은 경우에만 따로 파일 스트림 변수를 받아서 사용하시길 바랍니다.

<br>

## **개행(new line) 구분 텍스트 텍스트 리스트 변환**

<br>

- 어떤 텍스트 파일이 개행 문자로 구분되어 있어서 행 방향으로 계속 쌓여져 있는 경우가 있습니다. 다음과 같습니다.

<br>

```
aaa
bbb
ccc
```

<br>

- 이와 같은 경우 다음 코드를 이용하여 간단하게 리스트로 변환할 수 있습니다.

<br>

```python
text_to_list = open(file_path).read().split("\n")
# ["aaa", "bbb", "ccc"]
```

<br>

## **파일의 첫 행 또는 끝 행 출력**

<br>

- 파이썬에서 어떤 파일을 읽었을 때, 그 파일의 첫 행 또는 끝 행을 읽는 방법은 크게 2가지 방법이 있습니다.
- ① 실제 파일을 읽어서 각 라인 별로 읽을 때, 첫 행을 읽거나 마지막 행까지 읽어서 마지막 행을 확인하는 방법

<br>

```python
file_name = "sample.txt"
with open(file_name, "r") as file:
    first_line = file.readline()
    for last_line in file:
        pass
print(first_line)
print(last_line)
```

<br>

- ② 시스템 함수를 이용하여 첫 행 또는 끝 행을 읽는 방법 (ex. 리눅스의 head, tail)

 <br>
 
 ```python
 import os
 file_name = "sample.txt"
 first_line = os.popen("head -1" + file_name).read()
 last_line = os.popen("tail -1" + file_name).read()
 ```
 
<br>

## **현재 시각 문자열로 출력하는 방법**

<br>

```python
import time

def GetCurrentTime():
    now = time.localtime()
    now_str = "%02d_%02d_%02d_%02d_%02d_%02d" % (now.tm_year - 2000, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return now
```

<br>

- 또는 다음과 같은 방법이 있습니다.

<br>

- `datetime`을 이용하면 현재 시각을 출력할 수 있습니다.

<br>

```python
from datetime import datetime
now = datetime.now()
print( now )
print( now.year )
print( now.month )
print( now.day )
print( now.hour )
print( now.minute )
print( now.second )
print ( '%s-%s-%s' % ( now.year, now.month, now.day ) )

def GetPresentTime():
    now = datetime.now()
    ret = "%s_%s_%s_%s_%s_%s" % ( now.year, now.month, now.day, now.hour, now.minute, now.second)
    return ret
```

<br>

## **파이썬에서 폴더 및 파일 있는지 확인 후 생성**

<br>

- 파이썬에서 어떤 폴더나 파일이 있는 지 확인하고 없으면 생성해야 하는 경우가 있습니다.
- 먼저 폴더가 있는 지 확인하고 폴더가 없으면 폴더를 생성하는 코드는 다음과 같습니다.

<br>

```python
import os

folder_name = "test"

if os.path.isdir(folder_name) == False:
    os.mkdir(folder_name)

if os.path.exists(folder_name) == False:
    os.makedirs(folder_name)
```

<br>

- 이번에는 파이썬에서 어떤 파일이 있는 지 확인하고 없으면 파일을 생성하는 코드입니다.

<br>

```python
import os

file_name = "test.txt"

if os.path.isfile(file_name) == False:
    f = open("file_name", "w")
    ...
```

<br>

## **리스트 내부의 경우의 수 조합하기**

<br>

- 아래 코드를 활용하면 리스트 내부의 element들을 조합하여 만들어 낼 수 있는 전체 조합의 경우를 만들어 줍니다.

```python
from itertools import product

items = [['a', 'b', 'c,'], ['1', '2', '3', '4'], ['!', '@', '#']]
list(product(*items))
# [('a', '1', '!'), ('a', '1', '@'), ('a', '1', '#'), ('a', '2', '!'), ('a', '2', '@'), ('a', '2', '#'), ('a', '3', '!'), ('a', '3', '@'), ('a', '3', '#'), ('a', '4', '!'), ('a', '4', '@'), ('a', '4', '#'), ('b', '1', '!'), ('b', '1', '@'), ('b', '1', '#'), ('b', '2', '!'), ('b', '2', '@'), ('b', '2', '#'), ('b', '3', '!'), ('b', '3', '@'), ('b', '3', '#'), ('b', '4', '!'), ('b', '4', '@'), ('b', '4', '#'), ('c,', '1', '!'), ('c,', '1', '@'), ('c,', '1', '#'), ('c,', '2', '!'), ('c,', '2', '@'), ('c,', '2', '#'), ('c,', '3', '!'), ('c,', '3', '@'), ('c,', '3', '#'), ('c,', '4', '!'), ('c,', '4', '@'), ('c,', '4', '#')]

# Calculate all profucs of an input
list(itertools.product('abc', repeat=2))
# [('a', 'a'), ('a', 'b'), ('a', 'c'), 
#  ('b', 'a'), ('b', 'b'), ('b', 'c'), 
#  ('c', 'a'), ('c', 'b'), ('c', 'c')]
 
# Calculate all permutations
list(itertools.permutations('abc'))
# [('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), 
#  ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]

>>> # Take elements for iterator as long as predicate is True
>>> list(itertools.takewhile(lambda x: x<5, [1,4,6,4,1]))
[1, 4]
```

<br>

## **모듈이 설치 되었는 지 확인**

<br>

```python
import sys
'numpy' in sys.modules
>> True
```

<br>

## **pip가 설치가 안되어 있으면 설치**

<br>

```python
import sys
import subprocess

# pip가 없으면 pip를 설치한다.
try:
    import pip
except ImportError:
    print("Install pip for python3")
    subprocess.call(['sudo', 'apt-get', 'install', 'python3-pip'])
```

<br>

## **모듈이 설치가 안되어 있으면 설치**

<br>

```python
import sys
import subprocess

# pandas가 없으면 pandas를 설치한다.
try:
    import pandas as pd
except ModuleNotFoundError:
    print("Install pandas in python3")
    subprocess.call([sys.executable, "-m", "pip", "install", 'pandas'])
finally:
    import pandas as pd
```

<br>

## **입력한 디렉토리의 부모 디렉토리 출력**

<br>

- 아래 코드는 입력한 `input_path`의 부모 디렉토리를 가져옵니다.

<br>

```python
import os
os.path.abspath(os.path.join(input_path, os.pardir))
```

<br>

## **특정 경로의 특정 확장자 파일명만 prefix 추가**

<br>

- `path`와 그 `path` 안의 확장자인 `ext`를 입력 받고 그 확장자에 해당하는 파일명에 어떤 `prefix`를 넣고 싶으면 다음 코드를 사용하면 됩니다.
- 예를 들어 `path=C:\test`이고 `ext=png` 이고 `prefix=200101_`이면 **C:\test** 경로에서 **png**파일만 선택하여 입력받은 접두사인 **200101_**을 파일명 앞에 붙입니다,

<br>

```python
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True)
parser.add_argument('--ext', required = True)
parser.add_argument('--prefix', required= True)
args = parser.parse_args()

temp_filenames = os.listdir(args.path)
filenames = []

for filename in temp_filenames:
    extension = filename.split('.')[-1]
    if extension == args.ext:
        filenames.append(filename)

for filename in filenames:
    os.rename(args.path + '/' + filename, args.path + '/' + args.prefix + filename)

```

<br>

## **특정 경로의 특정 확장자 파일명만 suffix 추가**

<br>

- `path`와 그 `path` 안의 확장자인 `ext`를 입력 받고 그 확장자에 해당하는 파일명에 어떤 `suffix`를 넣고 싶으면 다음 코드를 사용하면 됩니다.
- 예를 들어 `path=C:\test`이고 `ext=png` 이고 `prefix=_200101`이면 **C:\test** 경로에서 **png**파일만 선택하여 입력받은 접두사인 **_200101**을 파일명 앞에 붙입니다,

<br>

```python
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', required = True)
parser.add_argument('--ext', required = True)
parser.add_argument('--suffix', required= True)
args = parser.parse_args()

temp_filenames = os.listdir(args.path)
filenames = []

for filename in temp_filenames:
    extension = filename.split('.')[-1]
    if extension == args.ext:
        filenames.append(filename)

for filename in filenames:
    os.rename(args.path + '/' + filename, args.path + '/' + filename.split('.')[-2] + args.suffix + '.' + filename.split('.')[-1])
```

<br>

## **local 영역에서 global로 import**

<br>

- 필요에 따라 어떤 함수 내에서 import를 사용해야 하는 경우가 발생할 수 있습니다. 
- 이 경우 그 함수 내에서만 import 한 라이브러리를 사용하는 것이라면 상관없지만 모든 영역에서 다 사용하게 만들고 싶으면 아래 코드를 이용하여 import 하면 문제 없이 전체 영역에서 사용할 수 있습니다.

<br>

```python
def GlobalImport(modulename, shortname = None, asfunction = False):
    if shortname is None: 
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:        
        globals()[shortname] = eval(modulename + "." + shortname)

```

<br>

- 위 함수를 이용하여 local 영역에서 global 하게 import할 수 있습니다. 다음과 같이 사용할 수 있습니다.

<br>

```python

def func():
    # import numpy as np 를 global에서 선언한 것과 같은 효과
    GlobalImport('numpy', 'np)

```

<br>

## **주어진 index 목록에 해당하는 값 불러오기**

<br>

- 만약 어떤 list가 주어지고 그 list에서 필요한 값의 인덱스만을 저장한 또다른 리스트가 다음과 같이 있다고 가정해 보겠습니다.

<br>

```python
A = [9, 4, 5, 8, 10, 14]
index_list = [1, 3, 4]
```

<br>

- `index_list`의 인덱스에 해당하는 값만을 가져오려고 합니다. 예를 들어 `A[1] = 4, A[3] = 8, A[4] = 10` 를 불러와야 합니다. 이 때, 다음 코드를 이용할 수 있습니다.

<br>

```python
result_list = [ A[i] for i in index_list ] 
```

<br>

## **숫자 형태의 list를 특정 문자로 구분하여 문자열로 변환**

<br>

- 1. 먼저 숫자를 문자로 변환한 형태의 list를 만들어야 합니다. 이는 `map`을 이용하여 쉽게 구현할 수 있습니다.
- 2. 그 다음 특정 문자 (seperator)를 기준으로 list의 문자열 원소를 연결하여 하나의 문자열로 만들 수 있습니다.

<br>

```python
A = [1,2,3,4]

# 1. 먼저 숫자를 문자로 변환한 형태의 list를 만들어야 합니다. 이는 `map`을 이용하여 쉽게 구현할 수 있습니다.
list(map(str, A)

# 2. 그 다음 특정 문자 (seperator)를 기준으로 list의 문자열 원소를 연결하여 하나의 문자열로 만들 수 있습니다.
seperator = ','
seperator.join(list(map(str, A))
```

<br>

## **List를 group 단위로 나누기**

<br>

- 다음과 같이 리스트에서 인접한 값들을 단위로 묶어야 할 때가 있습니다.
- 예를 들어 [1,2,3,4,5,6] 에서 3개씩 묶는다면 (1, 2, 3)과 (4, 5, 6)으로 묶을 수 있습니다.
- 위 예와 같이 앞에서 부터 차례대로 n개씩 묶을 때 사용할 수 있는 방법을 소개하겠습니다.

<br>

```python
a = [1, 2, 3, 4, 5, 6]  
group_adjacent = lambda a, k: zip(*([iter(a)] * k)) 
group_a = [e for e in group_adjacent(a, 3)]
print(group_a)
# [(1, 2, 3), (4, 5, 6)]
```

<br>

- 위 코드에서 사용된 `lambda` 함수인 `lambda a, k: zip(*([iter(a)] * k))`를 이용하여 함수를 간단하게 만들 수 있습니다.
- `group_adjacent(list_name, group_number)`에서 group_number의 갯수 만큼 그룹을 나눌 수 있습니다.

<br>

## **현재 실행 중인 파이썬 파일의 경로 확인 (__file__)**

<br>

- 현재 실행 중인 파이썬 파일의 경로를 알기 위해서는 다음 명령어를 통해 확인 가능합니다.

<br>

```python
import os
current_file_path = os.path.abspath(__file__)
```

<br>

- 여기서 사용된 `__file__`은 어떤 `.py` 파일이 실행될 때, 그 파일을 나타냅니다.
- 따라서 파이썬 파일이 실행되는 도중에는 그 파일의 경로를 쉽게 확인할 수 있습니다.

<br>

- 조금 더 응용하여 만약 이 파일의 디렉토리 경로만 알고 싶으면 다음과 같이 알 수 있습니다.

<br>

```python
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
```

<br>

## **폴더의 하위 전체 구조 및 파일 모두 복사**

<br>

- 폴더의 하위 전체 구조를 복사하기 위해서는 `shutil.copytree(src, dst)`를 사용하면 됩니다. 만약 기존에 경로가 존재한다면 복사가 안되므로 경로 체크를 한 뒤 기존에 경로가 있으면 삭제 하고 복사하는 방법을 추천드립니다. 방법은 다음과 같습니다.

<br>

```python
import os
import shutil

src = "path/to/the/src"
dst = "path/to/the/dst"

if os.path.isdir(dst):
    shutil.rmtree(dst)

shutil.copytree(src, dst)
```

<br>

## **파일 경로를 경로와 파일명으로 나누기**

<br>

- 어떤 파일의 경로 "path/to/the/file.txt"과 같이 있을 때, 파일의 경로는 "path/to/the" 이고 파일명은 "file.txt"이라고 하겠습니다. 파일의 경로를 경로와 파일명으로 다음 방법을 통하여 쉽게 분리할 수 있습니다.

<br>

```python
import os
file_path = "path/to/the/file.txt"
os.path.dirname(file_path)
# path/to/the
os.path.basename(file_path)
# file.txt
```

<br>

## **특정 경로의 특정 확장자 파일명 가져오기**

<br>

- 아래 함수는 path 경로에 extensions로 정의한 확장자에 해당하는 파일을 가져옵니다.
- 이 때, abs_path가 True이면 절대 경로로 가져오고 False이면 파일명만 가져옵니다.

<br>

```python
from typing import List
def GetFileList(path:str, extensions:List, abs_path:bool=False) -> List:
    file_paths = []
    file_names = os.listdir(path)
    for file_name in file_names:
        extension = file_name.split('.')[-1]
        if extension in extensions:
            if abs_path:
                file_paths.append(os.path.abspath(path + os.sep + file_name))
            else:
                file_paths.append(file_name)
    return file_paths
```

<br>

## **HTML 랜덤 컬러 만들기**

<br>

- HTML에서 CSS에 사용되는 컬러는 `#AD22F5`과 같은 6개의 16진법 순자로 이루어진 코드로 정의 됩니다.
- 임의의 HTML 랜덤 컬러를 만들려면 아래 함수를 사용하여 만들 수 있습니다.

<br>

```python
def RandomColor():
    import random
    r = lambda: random.randint(0,255)
    random_color_code = '#%02X%02X%02X' % (r(),r(),r())
    return random_color_code    
```
