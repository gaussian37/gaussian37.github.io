---
layout: post
title: Python 기본 문법 및 코드 snippets
date: 2019-03-18 00:00:00
img: python/basic/python.jpg
categories: [python-basic] 
tags: [python, python 기본] # add tag
---

- 이 글에서는 Python을 사용하면서 필요하다고 느끼는 `Python 기본 문법 및 코드`에 대하여 알아보겠습니다.

<br>

## **목차**

<br>

- ### if __name__ == "__main__": 사용 이유
- ### Comparator를 이용한 클래스 정렬
- ### file read/write 방법
- ### 현재 시각 문자열로 출력하는 방법
- ### all과 any
- ### lambda
- ### map
- ### filter
- ### reduce
- ### for loop 스타일 (zip, range, enumerate, sorted, filter)
- ### deque은 사이즈 관리를 자동으로 한다.
- ### 파이썬 프로그램 정상 종료 시키기
- ### 파이썬에서 폴더 및 파일 있는지 확인 후 생성
- ### 리스트 내부의 경우의 수 조합하기
- ### 모듈이 설치 되었는 지 확인
- ### pip가 설치가 안되어 있으면 설치
- ### 모듈이 설치가 안되어 있으면 설치
- ### argparse 사용법
- ### 문자열 검색 : startswith, endswith, in, find, re
- ### 디렉토리(+ 파일) 복사하기
- ### 디렉토리(- 파일) 복사하기
- ### 입력한 디렉토리의 부모 디렉토리 출력
- ### 특정 문자를 기준으로 split
- ### 특정 경로의 특정 확장자 파일명만 prefix 추가
- ### 특정 경로의 특정 확장자 파일명만 suffix 추가
- ### 숫자에 0을 채워서 출력

<br>

## **if __name__ == "__main__": 사용 이유**

<br>

- 어떤 파이선 파일을 만들었을 때, 그 파이썬 파일을 import 하면 그 파일 안에 있는 함수들을 사용할 수 있게 됩니다.
- 이 때 `if __name__ == "__main__"`을 이용하여 main 함수를 컨트롤 하지 않으면 전역 변수 영역에 선언된 모든 함수들이 실행되게 됩니다.
- 예를 들어 `print` 함수들이 전역 변수 영역에 정의되어 있다면 `import` 할 때, 실행되어 원하지 않는 결과가 출력될 수 있습니다.
- 즉, 다른 파이썬에서 import 하였을 때, `def`로 정의된 함수들만 `import`되도록 하기 위해서는 main 함수를 통해 컨트롤 해야 합니다. 

<br>

## **Comparator를 이용한 클래스 정렬**

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

## **file read/write 방법**

<br>

- 먼저 텍스트 파일을 읽는 코드 입니다.

<br>

```python
# 읽을 파일의 경로와 파일 이름을 입력합니다.
filepath = "C://"
filename = "test.txt"

# 텍스트 파일을 입력 받기 위한 stream을 open 합니다.
file_read=open(filepath + '/' + filename, 'r')
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
file_read.close()
```

<br>

```python
# 텍스트 파일을 출력 하기 위한 stream을 open 합니다.
file_write = open(filepath + '/' + filename, 'w')
s = 'contents...'
file_write.write(s)
file_write.close()
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
    ret = "%s-%s-%s-%s-%s-%s" % ( now.year, now.month, now.day, now.hour, now.minute, now.second)
    return ret
```

<br>

## **all과 any**

<br>

-`all`을 이용하면 iterable한 객체의 값 중에 False 또는 0이 있으면 False를 반환합니다.
- 반면 모든 값이 True 또는 0이 아니어야 True를 반환합니다.
- iterable한 객체이므로 list, tuple, set, dictionary 모두 적용 가능하고 dictionary의 값에서는 key 값을 가지고 판단합니다.

<br>

- `any`를 이용하면 객체의 값 중에 True 또는 0이 아닌 값이 하나라도 있으면 True로 반환합니다.
- 반면 모든 값이 False 또는 0이어야 False를 반환합니다.

<br>

## **lambda**

<br>

- 출처: https://offbyone.tistory.com/73
- 파이썬에서 `lambda` 는 런타임에 생성해서 사용할 수 있는 익명 함수 입니다. 파이썬에서 `lambda`의 역할은 혼자 쓰일 때라기 

보다는 다양한 다른 함수와 같이 쓰일 때 큰 힘을 발휘합니다.
    - 예를 들어 `map`, `filter`, `reduce` 등이 있습니다.
- 먼저 `lambda`에 대한 간략한 소개와 그에 이어서 `map`, `filter`, `reduce`에 어떻게 lambda가 사용하는 지 예를 보여드리곘

습니다.
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

- map 함수와 lambda 함수가 함께 사용될 때 기능들을 간단하게 구현할 수 있습니다.
- 먼저 map() 은 두 개 이상의 파라미터를 기본적으로 받습니다. 첫 파라미터는 function이고 두번째 파라미터부터는 function에 

적용할 iterable 입니다. 여기서 iterable은 한번에 하나의 멤버를 반환할 수 있는 객체 입니다.
- 결과물로 function에 의해 변경된 iterator를 반환합니다.

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
a = foo = [1,2,3,4,5]
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
- 아래 코드에서 `i`는 0, 1, 2, ... 순서로 입력됩니다.

<br>

```python

A = [10, 20, 30, 40, 50]
for i, data in ennerate(A):
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

## **입력한 디렉토리의 부모 디렉토리 출력**

<br>

- 아래 코드는 입력한 `input_path`의 부모 디렉토리를 가져옵니다.

<br>

```python
import os
os.path.abspath(os.path.join(input_path, os.pardir))
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

## **숫자에 0을 채워서 출력**

<br>

- `str.zfill(숫자)` 를 통해 문자열의 공백을 0으로 채울 수 있습니다.
- `"%03d"`와 같은 형태로 출력하면 정수에 공백을 추가하여 출력할 수 있습니다.

