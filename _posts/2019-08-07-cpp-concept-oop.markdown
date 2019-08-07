---
layout: post
title: C++ 객체 지향 기초  
date: 2019-08-07 00:00:00
img: cpp/cpp.png
categories: [cpp-concept] 
tags: [cpp, c++, 객체 지향, oop, object oriented] # add tag
---

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

<br>

※ 참조 자료 : 홍정모의 따라하며 배우는 C++