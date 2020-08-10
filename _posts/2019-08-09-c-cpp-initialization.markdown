---
layout: post
title: C++ 변수, STL, 클래스 멤버 변수(public) 등의 초기화
date: 2019-08-07 00:00:00
img: cpp/cpp.png
categories: [c-cpp] 
tags: [cpp, c++, 객체 지향, oop, object oriented] # add tag
---

- C++에서 변수를 초기화 하는 다양한 방법 중 가장 간단한 방법을 간략하게 소개하겠습니다.
- `중괄호`를 이용하여 변수를 초기화 하는 방법이 가장 간단한 방법 중 하나입니다.
- 아래 예제 코드를 살펴보면 **선언과 동시에 바로 초기화**가 가능합니다.

<br>

```cpp
#include <iostream>
#include <vector>
#include <set>
#include <map>

using namespace std;

class ClassTest {

public:
	int a;
	int b;
};

int main() {

	// 정수형 변수 테스트
	int a{ 1 };
	cout << "Int variable test : " << a << endl;
	
	// 벡터 테스트
	vector<int> v{ 1,2,3 };
	cout << "Vector test : ";
	for (auto& n : v)
		cout << n << " ";
	cout << endl;

	// 셋 테스트
	set<int> s{ 1,2,3 };
	cout << "Set test : ";
	for (auto& iter : s)
		cout << iter << " ";
	cout << endl;

	// 클래스 테스트
	ClassTest test{ 1, 2 };
	cout << "Class test : " << test.a << " " << test.b << endl;

}
```

<br>

```cpp
Int variable test : 1
Vector test : 1 2 3
Set test : 1 2 3
Class test : 1 2
```

<br>

- 위 코드를 보면 정수형 변수, STL의 벡터 및 셋, 그리고 클래스 모두 선언과 동시에 중괄호로 초기화 값을 선언하면 초기값 지정이 가능합니다.
- 물론 클래스의 경우 멤버 변수가 **public**인 경우에만 중괄호를 통한 초기화가 가능하고 일반적으로 멤버 변수는 **private**으로 선언되어 있으므로 constructor를 통하여 초기화를 해주어야 합니다. 