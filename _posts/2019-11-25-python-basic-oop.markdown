---
layout: post
title: 객체 지향 프로그래밍 with 파이썬
date: 2019-11-25 00:00:00
img: python/basic/python.jpg
categories: [python-basic] 
tags: [python, oop, object oriented, 객체 지향] # add tag
---

<br>

- 이 글에서는 파이썬을 이용하여 객체 지향을 하는 전체적인 과정(클래스, 상속, 다형성 등등)을 한번 담아보려고 합니다. 
- 기초적인 내용부터 시작하려고 하오니 차근 차근 읽어보시면 객체 지향 내용을 간략하게 확인하실 수 있을 것입니다.

<br>

## **목차**

<br>

- ### __init__과 __call__의 차이점(#__init__과-__call__의-차이점-1)
- ### 상속과 다형성(#상속과-다형성)
- ### 객체의 생성과 초기화(#객체의-생성과-초기화)
- ### __str__과 __repr__의 차이점(#__str__과-__repr__의-차이점-1)
- ### __getattr__ 와 __getattribute__(#__getattr__-와-__getattribute__-1)
- ### __bytes__(#__bytes__)

<br>

## **__init__과 __call__의 차이점**

<br>

```python
class A:
    def __init__(self):
        print("init")
           
    def __call__(self):
        print("call")
           
a = A()
>> init

a()
>> call
```

<br>

- `__init__`은 객체를 생성할 때 실행 되는 constructor이고 `__call__`은 생성된 객체에서 객체 이름을 그대로 이용하여 함수처럼 사용할 때 호출되는 함수입니다.

<br>

## **상속과 다형성**

<br>

- `super()`에 대하여 알아보도록 하겠습니다. 다음 링크를 참조하였습니다. (https://youtu.be/6-XxUwTvsP8)
- `super()`는 자식 클래스에서 부모 클래스의 멤버 함수를 사용하고 싶은 경우에 사용합니다.
    - 따라서 `super().부모클래스_멤버함수`

```python
class Mammal():
    def __init__(self, name):
        self.name = name

    def walk(self):
        print("walking")

    def eat(self):
        print("eating")

    def greet(self):
        print("{} is greeting".format(self.name))

class Human(Mammal):

    def __init__(self, name, hand):
        super().__init__(name)
        self.hand = hand

    def wave(self):
        print("With shacking {} hand,".format(self.hand))

    def greet(self):
        self.wave()
        super().greet()

person = Human("Peter", "right")
person.greet()

With shacking right hand,
Peter is greeting
```

<br>

- 위 클래스에서 Mammal이 부모 클래스이고 Human이 자식 클래스 입니다.
- 이 때, Human 클래스의 `__init()__`을 살펴보면 `super().__init__(name)`이 있습니다.
- 이 함수에 사용된 `super()`를 이용하여 부모 클래스인 Mammal에 접근하고 Mammal의 멤버 함수인 `__init__`에 접근하여 초기화 작업까지 하게 됩니다.

<br>

## **객체의 생성과 초기화**

<br>

- `__init__(self[, ...])` : 인스턴스가 생성될 때 호출되는 메소드입니다. 인자를 받아서 내부에 지정해 줄 수 있습니다.
- `__del__(self)` : 객체가 소멸 될 때 해야할 작업을 지정할 수 있습니다. 예를 들어 객체 소멸시 파일을 닫아 주는 역할을 할 때 사용할 수 있습니다.

<br>

```python

class Test:
    def __init__(self):
        print("Init")
    def __del__(self):
        print("del")

test = Test()
# Init

del test
# del
```

<br>

## **__str__과 __repr__의 차이점**

<br>

- 참조 : https://shoark7.github.io/programming/python/difference-between-__repr__-vs-__str__
- 참조 : https://medium.com/swlh/understanding-repr-and-str-in-python-65dd41538943
- 파이썬에는 내장된 많은 자료형들에, 해당하는 자료형에 대한 연산을 정의하는 메소드들이 있습니다. 그 메소드들은 메소드의 이름 앞뒤에 `‘__‘`(double underscore)를 지니고 있습니다.
- 예를 들어 파이썬에서는 모든 변수들이 클래스이고 인스턴스입니다. 따라서 정수 ‘3’에 대하여 이 값이 내장 int 클래스의 인스턴스임을 알 수 있습니다.

<br>

```python
isinstance(3, int)
# True
```

<br>

- 즉, int 타입도 단순히 정수 라기 보다는 클래스로 정의가 되어 있습니다. 따라서 클래스이므로 사칙연산이나 대소 관계 등에서도  `__add__`, `__sub__`과 같은 연산자가 정의되어 있어야 합니다. 예를 들어 다음과 같습니다.

<br>

```python
3 + 5 # 일반적으로 생각하는 정수의 사칙연산
# 8
(3).__add__(5) # 내부적으로는 다음과 같은 연산자에 의해 계산이 됩니다.
# 8

[1, 2, 3] + [4, 5, 6] # 일반적으로 생각하는 파이썬 리스트의 concatenation
# [1, 2, 3, 4, 5, 6]
[1, 2, 3].__add__([4,5,6]) # 내부적으로 다음과 같은 연산자에 의해 concatenation이 됩니다.
# [1, 2, 3, 4, 5, 6]
```

<br>

- 첫 번째 예에서 `3 + 5`는 `3`이라는 정수 인스턴스에 대해 `__add__` 메소드를 호출합니다. 그 값은 `5`를 받아 새로운 정수 8을 반환하게 된다.
- 두 번째 예는 같은 `+` 연산자에 대해 클래스마다 다른 구현이 되어 있음을 보여줍니다. list 자료형은 `+` 연산에 대해 concatenate를 하고 새로 생성된 list를 반환합니다. 
- 실제 클래스들에서 구현하는 위와 같은 메소드들을 `Magic method` 라고 하며 매우 많은 목록이 존재합니다.

<br>

- 그러면 Magic method에 대하여 알아보았으니 `__repr__`과 `__str__`에 대하여 알아보고 그 차이점에 대하여 비교하도록 하겠습니다.
- `__repr__`과 `__str__`은 클래스의 정보를 사람이 읽을 수 있게 제공하기 위한 기능입니다.
- 먼저 다음 예제를 살펴보도록 하겠습니다.

<br>

```python
class Animal:
    def __init__(self, animal, breed):
        self.animal = animal
        self.breed = breed

a = Animal("Dog", "Pomeranian")
print(a)
# <__main__.Animal object at 0x000002081F368408>
```

<br>

- 위와 같이 객체 `a`를 출력하였을 떄, 사람은 전혀 알 수 없는 정보가 출력이 됩니다.
- 이 문제를 개선하기 위하여 `__repr__`과 `__str__`이 제공됩니다. 다음 코드를 통하여 어떻게 사용되는 지 살펴보도록 하겠습니다.

<br>

```python
class Animal:
    def __init__(self, animal, breed):
        self.animal = animal
        self.breed = breed
        
    def __str__(self):
        return "__str__, animal is {}, breed is {}".format(self.animal, self.breed)
    def __repr__(self):
        return "__repr__, animal is {}, breed is {}".format(self.animal, self.breed)

a = Animal("Dog", "Pomeranian")
print(a)
# __str__, animal is Dog, breed is Pomeranian
str(a)
# __str__, animal is Dog, breed is Pomeranian
a
# __repr__, animal is Dog, breed is Pomeranian
```

<br>

- 위 사용 용도를 살펴보면 `print` 또는 `str` 을 통하여 객체의 정보를 확인할 떄에는 `__str__`이 사용되고 단순히 객체를 확인할 경우 `__repr__`이 사용됩니다.
- 아래 예제를 통하여 다시 한번 사용 용도를 확인해 보겠습니다.

<br>

```python
import datetime
today = datetime.date.today()

print(today)
# 2020-11-07
str(today)
# 2020-11-07

today
# datetime.date(2020, 11, 7)
repr(today)
# 'datetime.date(2020, 11, 7)'
```

<br>

## **__getattr__ 와 __getattribute__**

<br>

- 참조 : https://medium.com/@starriet87/%ED%8C%8C%EC%9D%B4%EC%8D%AC-getattr-%EC%99%80-getattribute-%EC%9D%98-%EC%B0%A8%EC%9D%B4-46ef0174e8e0
- `__getattr__` : 객체에 해당 attribute가 없을경우 호출됩니다. 즉, 객체의 멤버값들을 모두 찾아봐도 필요한 값이 없으면 `__getattr__`를 호출하게 되며 입력값의 인자는 attribute가 됩니다. 따라서 `__getattr__`의 호출은 객체 내 다른 멤버값을 모두 찾아보고 없으면 사용하는 것으로 우선 순위가 가장 낮습니다.
- `__getattribute__` : 객체의 해당 attribute 존재 유무와 상관없이 호출되며 입력값의 인자는 attribute가 됩니다.

<br>

```python
class Test:
    def __getattr__(self, name):
        return ('__getattr__ test : ' + name)

a = Test()
a.ace = 'ace value'
print(a.ace)
# ace value

print(a.ace2)
# __getattr__ test : ace2

print(a.__dict__)
# {'ace': 'ace value'}
```

<br>

- 위 예제를 보면 `a.ace`는 클래스 내부에 멤버 변수로 할당되었기 때문에 값을 그대로 출력합니다.
- 반면 `a.ace2`는 멤버 변수 값이 없기 때문에 `__getattr__`에 정의된 값을 이용하여 출력합니다. 마치 예외처리 하듯이 사용할 수 있습니다.
- 마지막으로 `__dict__`를 이용하면 멤버 변수의 변수명과 그 값을 살펴볼 수 있습니다.

<br>

```python
class Test:
    def __getattr__(self, name):
        return ('__getattr__ : ' + name)
    def __getattribute__(self,name):
        return ('__getattribute__ : ' + name)

a = Test()
a.ace = 'ace value'
print(a.ace)
# __getattribute__ : ace

print(a.ace2)
# __getattribute__ : ace2

print(a.__dict__)
# __getattribute__ : __dict__
```

<br>

- 반면 `__getattribute__`의 경우 멤버 변수 값이 있음에도 불구하고 `__getattribute__`가 실행되었습니다. 심지어 `__dict__`의 경우에도 실행된 것을 확인할 수 있습니다. `__getattribute__`의 경우 가장 높은 우선순위로 실행되는 것을 알 수 있습니다.

<br>

- 정리하면 다음과 같습니다.
- `__getattribute__`는 해당 attribute의 유무에 상관없이 가장 **선순위**로 호출됩니다.
- 반면 `__getattr__`는 해당 attribute를 찾을 수 없을경우 가장 **후순위**로 호출됩니다.

<br>

## **__bytes__**

<br>

- bytes에 의해 호출되어 객체의 바이트열 표현을 계산할 때 사용됩니다. 따라서 반환 값은 반드시 bytes 객체이어야 합니다.

<br>

```python
class Test:
     def __bytes__(self):
        return str.encode(self.string)

a =  Test()

a.string = "abc"

bytes(a)
# b'abc'
```

<br>