---
layout: post
title: C++ 객체 지향 기초  
date: 2019-08-07 00:00:00
img: cpp/cpp.png
categories: [cpp-concept] 
tags: [cpp, c++, 객체 지향, oop, object oriented] # add tag
---

## 객체 지향 프로그래밍과 클래스

<br>

- C에서 C++로 업그레이드 되면서 적용 된 것 중에 하나가 바로 **객체 지향**입니다.
- 그러면 객체 지향 프로그래밍은 왜 쓰는 것일까요? 의미는 여러가지가 있지만 가장 간단하게 접근해 보도록 하겠습니다.
- 객체 지향은 당연히 편하려고 사용 합니다. 반복되는 것을 피하기 위해서 하나의 패키지로 만드는 것이지요.
- 객체와 클래스를 설명할 때에 붕어빵과 틀 같은 개념이 많이 사용되고 있지만 그냥 간단하게 객체는 **데이터 + 기능**이라고 보시면 되고 클래스는 객체를 구현하는 방법이라고 보셔도 됩니다. 

<br>

```cpp
#include <iostream>
#include <string>

using namespace std;

class Friend {

public: //access specifier (public, private, protected)

	string name;
	string address;
	int age;
	double height;
	double weight;

	void print() {
		cout << name << " " << address << " " << age << " " << height << " " << weight << endl;
	}
};


int main() {

	Friend fr{ "He", "Seoul", 30, 178, 70 };
	fr.print();

}
```

<br>

- 먼저 위 예제를 보면 **class**를 이용하여 데이터를 쉽게 저장을 하였습니다. 
- 만약 **class**를 쓰지 않는 다면 이름, 주소, 나이 등을 한 세트로 관리하기도 어렵고 상당히 중복된 코드도 많이 발생할 것입니다. 그것을 피할 수 있게 만들어 주는 좋은 기능입니다.
- 예제에서 사용된 **class**의 **public, private, protected**는 추후에 다루어 보겠습니다. 간단하게 **public** 이하의 코드는 외부에서 접근 가능하게 공개적으로 둔다 라고 이해하시면 됩니다.
- **main** 함수를 보면 class를 선언하는 데 이 때 보면, 사실상 `메모리에 실제로 공간을 확보`하는 단계라고 할 수 있습니다.
    - 코드의 ```Friend fr{ "He", "Seoul", 30, 178, 70 };``` 부분 입니다.
    - 이 단계를 `instanciation`이라고 합니다. 
 
<br> 
 
```cpp
vector<Friend> friends;
friends.resize(2);

for (auto &ele : friends) {
    ele.print();
}
```

<br>

- main 함수에 위와 같은 코드를 추가한다면 아주 쉽게 데이터를 출력할 수 있습니다. **class**없이는 모든 데이터를 매번 직접 입력하여 출력해야 하는 반면에 **class**를 사용하면 상당히 코드 중복을 피할 수 있습니다.
- 즉, 제가 여기서 강조한 부분은 코드의 **재활용**입니다. 클래스를 사용해야 하는 가장 큰 이유 중의 하나는 바로 **재활용**입니다.

<br>

## 캡슐화, 접근 지정자, 접근 함수

<br>

- 위에서 언급한 접근 지정자(access specifier)에 대하여 한번 알아보겠습니다. 
- 접근 지정자는 **private, public, protected**가 있습니다. 먼저 알아볼 것은 **private, public**입니다. **proteced**는 상속의 개념이 필요하므로 나중에 다루겠습니다.
- **class**의 기본 접근 지정자는 **private**입니다. 즉, 접근 지정자를 따로 지정해 주지 않으면 기본 값은 **private**으로 설정된다는 뜻입니다.
- **private**과 **public**의 가장 큰 차이점은 클래스 외부에서의 접근 가능성입니다.
- 즉, **public**의 경우 클래스 외부에서 자유롭게 접근 및 수정이 가능하지만 **private**의 경우 클래스 외부에서 접근이 불가능합니다.
    - 따라서 **private**으로 멤버 변수를 선언한 경우 멤버 변수를 접근 및 수정할 수 있는 **public**영역의 함수를 만들어 주어야 합니다.
    - 이렇게 하는 이유는 의도치 않은 멤버 변수의 변경을 막기 위함입니다.
    - 또한 멤버 변수명 변경 시에도 사용한 변수를 전부 찾아서 변경해 줄 필요는 없고 접근 함수 내의 멤버 변수 값만 바꾸어 주면되어서 편리합니다. 
- 위와 같이 실제 구현한 함수 등은 외부에서 접근 가능하도록 하고 멤버 변수는 감추는 것을 `캡슐화`라고 합니다. 
- 객체 지향의 3가지 원리 `캡슐화`, `상속`, `다형성` 중의 한가지 속성으로 객체 지향의 중요 속성 중 하나이니 꼭 기업합시다.  
- **private** 멤버 변수를 접근 하기 위한 **public** 함수 중에 `get` 또는 `set`으로 시작하는 함수들이 있습니다. 이 함수들이 멤버 변수를 변경하기 위한 함수 입니다.

<br>

```cpp
#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Date {

	int year_;
	int month_;
	int day_;
	
public:

	void setDate(const int& year, const int& month, const int& day) {
		
		year_ = year;
		month_ = month;
		day_ = day;
		
	}

	const int& getDay() {
		return day_;
	}

	void copyFrom(const Date& original) {
		year_ = original.year_;
		month_ = original.month_;
		day_ = original.day_;
	}

	void print() {
		cout << year_ << " " << month_ << " " << day_ << endl;
	}
};

int main() {

	Date today;
	today.setDate(2019, 8, 8);

	Date copy;
	copy.copyFrom(today);

	copy.print();

 } 
```   

<br>

- 위 코드를 보면 앞서 언급한 **get, set** 함수를 선언한 것을 볼 수 있습니다. **set**함수는 멤버 변수를 변경하기 위한 함수이고 **get**함수는 멤버 변수를 읽기 위한 용도 입니다.
- **copyFrom** 함수를 보면 original 매개변수의 멤버변수는 그냥 접근을 할 수 있는데 그 이유는 클래스 내에서는 같은 클래스의 객체는 접근할 수 있기 때문입니다.  

<br>

## 생성자 Constructors

<br>

 


<br>

※ 참조 자료 : 홍정모의 따라하며 배우는 C++