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

- ### 클래스와 객체지향 프로그래밍 기초
- ### __init__과 __call__의 차이점
- ### 상속과 다형성

<br>

## **클래스와 객체지향 프로그래밍 기초**

<br>

## __init__과 __call__의 차이점

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


