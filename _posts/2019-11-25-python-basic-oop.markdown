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

- ### __init__과 __call__의 차이점
- ### 상속과 다형성(#상속과-다형성-1)
- ### 객체의 생성과 초기화(#객체의-생성과-초기화-1)
- ### 객체의 표현(#객체의-표현-1)
- ### 객체의 속성 관리(#객체의-속성-관리-1)

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

## **객체의 표현**

<br>

- `__repr__(self)` : 객체를 나타내는 공식적인 문자열입니다. repr()로 호출 할 수 있습니다.
- `__str__(self)` : 객체를 나타내는 비공식적인 문자열이지만 객체를 이해하기 쉽게 표현할 수 있는 문자열입니다. __repr__보다 사용자에게 보기 쉬운 문자열을 출력하는 것에 지향점이 있습니다. str()로 호출 할 수 있습니다. 마찬가지로 string타입의 문자열을 반환해야 합니다. `__repr__()`만 구현되어있고 `__str__()`이 구현되어 있지 않은 경우에는 `str()`이 `__repr__()`을 불러오게 됩니다.
- `__bytes__(self)` : 객체를 나타내는 byte 문자열입니다. bytes()로 호출 할 수 있습니다.
- `__format__(self)` : 객체를 나타내는 format을 지정하고 싶을때 사용합니다.

<br>

## **객체의 속성 관리**

<br>

