---
layout: post
title: C++ 객체 지향 기초  
date: 2019-08-07 00:00:00
img: cpp/cpp.png
categories: [cpp-concept] 
tags: [cpp, c++, 객체 지향, oop, object oriented] # add tag
---

<br>

# **객체 지향 글 목차**

<br>

- 1) 객체지향 프로그래밍과 클래스
- 2) 캡슐화, 접근 지정자, 접근 함수
- 3) 생성자 Constructors
- 4) 생성자 멤버 초기화 목록
- 5) 위임 생성자
- 6) 복사 생성자
- 7) 소멸자 destructor
- 8) this 포인터와 연쇄 호출
- 9) 클래스 코드와 헤더 파일
- 10) 클래스와 const
- 11) 정적 멤버 변수
- 12) 정적 멤버 함수
- 13) 친구 함수와 클래스 friend
- 14) 익명 객체
- 15) 클래스 안에 포함된 자료형 nested types
- 16) 실행 시간 측정하기

<br>

## **1. 객체 지향 프로그래밍과 클래스**

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

## **2. 캡슐화, 접근 지정자, 접근 함수**

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

## **3. 생성자 Constructors**

<br>

- 생성자는 클래스를 생성할 때, 멤버 변수의 값을 초기화 해주는 역할을 합니다.
- 물론 클래스 내에서 멤버 변수에 접근하기 때문에 **private**이라도 생성자에서 접근 가능합니다.
- 아래 코드의 `Fraction` 함수가 생성자 입니다. 생성자는 클래스 명과 동일하게 지정해 주어야 하고 return 타입은 적어주지 않습니다.

<br>

```cpp
#include <iostream>

using namespace std;

class Fraction {

	int numerator_;
	int denominator_;

public:

	Fraction(const int& numerator, const int& denominator = 1) {
		cout << "Run constructor" << endl;
		numerator_ = numerator;
		denominator_ = denominator;
	}

	void print() {
		cout << numerator_ << " / " << denominator_ << endl;
	}
};

int main() {

	Fraction f(1, 2);
	f.print();

}
```

<br>

- 위 코드의 `Fraction`이 생성자에 해당합니다. 
- 만약 위의 코드와 같이 생성자를 지정해 주지 않으면 어떻게 될까요? 그러면 클래스 내부적으로 기본 생성자를 사용하게 됩니다.
    - 기본 생성자는 `클래스이름(){}` 형태로 위 예제에서는 `Fraction(){}`가 됩니다.
    - 즉 멤버 변수에 어떤 초기화도 하지 않으므로 쓰레기 값이 들어가게 됩니다.
- 인자를 보면 `const int& denominator = 1`라고 세팅되어 있습니다. 이것은 기본값을 지정하는 것이고 매개변수를 지정하지 않으면 기본값 1을 자동으로 받게 됩니다.
- 여기서 중요한 것은 생성자 또한 함수의 일종입니다. 즉, 생성자 자체가 어떤 동작을 한다기 보다는 **객체를 생성할 때 생성자라는 함수가 자동으로 실행된다고 생각**해야 합니다.

<br>

- C++ 11 부터는 중괄호를 이용하여 변수 초기화가 가능합니다.
- 먼저 생성자를 명시적으로 **생성하지 않은 경우** 부터 살펴보겠습니다.
    - 물론 클래스의 멤버 변수 또한 중괄호를 이용하여 초기화가 가능하지만 멤버 변수가 **public**일 때에만 가능합니다.
        - 예를 들어 위 예제에서 멤버 변수가 **public**일 때, `Fraction f{1,2}`로 선언하면 **numberator=1**, **denominator=2**로 세팅이 됩니다.
    - 생성자를 명시적으로 생성하지 않았고 멤버 변수가 **private**라면 중괄호를 이용한 변수 초기화는 불가능 합니다.
- 반면 생성자를 명시적으로 **생성**한 경우는 어떨까요?
    - 생성자를 생성한 상태에서는 생성자를 사용한 초기화나 중괄호를 사용한 초기화 모두 사용될 수 있습니다.
    - 예를 들어 `Fraction f{1,2}`로 생성하면 중괄호 내의 인자가 생성자로 전달됩니다.
    - 또한 `Fraction f(1,2)`로 생성하여도 소괄호 내의 인자가 생성자로 전달됩니다.
    - 그러면 차이점은 무엇일까요? 중괄호를 이용한 경우 자동 타입 변환이 안되는 반면 소괄호를 이용한 경우 변환이 됩니다.
    - 예를 들어 생성자에서 파라미터는 **int**로 선언되었는데 만약 `Franction{1.2, 2.3}`으로 생성하면 중괄호의 경우 오류가 나고 소괄호의 경우 경고만 발생합니다.   

<br>

- 다음 생성자 관련 예제를 살펴보겠습니다.

<br>

```cpp
class First {

public:

	First() {
		cout << "Run First constructor" << endl;
	}
};

class Second {

	First first;

public:

	Second() {
		cout << "Run Second constructor" << endl;
	}

};

int main() {

	Second second;
}

```

<br>

- 위 예제의 실행 결과는 `Run First constructor`이 먼저 출력되고 다음으로 `Run Second constructor`이 출력됩니다.
- 즉, 컴파일러가 생성자를 실행하기 전에 멤버 변수를 먼저 생성하는 것을 알 수 있습니다.

<br>

## **4. 생성자 멤버 초기화 목록**

<br>

- 멤버 변수를 초기화하는 방법에는 여러가지 방법이 있습니다.
- 그 중에서 3가지를 다루어 보려고 합니다.
    - **1. 멤버 변수 선언과 동시에 초기화**
    - **2. 생성자 초기화 리스트**
    - **3. 생성자 내부에서 초기화**     
- 아래 코드를 살펴 보겠습니다. 

<br>

```cpp
class Test {
	//생성과 동시에 초기화.
	// static 변수는 여기서 초기화 불가능
	int num1 = 1;
	double num2 = 1.0;
	int arr[3] = { 1, 1, 1 };

public:

	// 생성자 초기화 리스트
	Test() : num1(2.0), num2{ 2.0 }, arr{ 2, 2, 2 } {
		// 생성자 내부에서 초기화
		num1 = 3;
		num2 = 3.0;	
	}


	void print() {
		cout << "num1 : " << num1 << endl;
		cout << "num2 : " << num2 << endl;
		
		for (auto& e : arr)
			cout << e << " ";
		cout << endl;
	}

};

int main() {

	Test t;
	t.print();
}
```

<br>

- 만약 위와 같이 3가지 경우 모두 사용한다면 어떻게 될까요?
- 먼저 가장 먼저 실행 되는 코드는 **멤버 변수 선언**되는 영역입니다.
    - 이 영역에서 멤버 변수 생성과 동시에 초기화가 가능합니다. 하지만 전역 변수(static 변수)는 여기서 접근이 안됩니다. 물론 생성 또한 안됩니다.
- 그 다음으로 실행되는 코드는 **생성자 초기화 리스트**입니다.
    - 위 코드에서 `Test() : num1(2), num2(2.0), arr{ 2, 2, 2 }`에 해당합니다.
    - 이 영역에 보면 변수를 소괄호 또는 중괄호를 이용하여 초기화를 하였습니다.
        - 소괄호를 이용한 경우 초기화 할 때, 형 변환이 이루어 질 수 있습니다. 즉, num1의 변수가 정수형인데 소수가 입력되면 자동 형변환이 발생합니다.
        - 반면에 중괄호를 이용하면 자동 형 변환이 이루어 지지 않아서 에러가 발생합니다.
        - 또한 배열 타입을 초기화 할 때에는 중괄호를 이용하여 초기화 할 수 있습니다.
- 마지막으로 실행되는 코드는 **생성자 내부** 영역입니다.
    - 생성자 내부에서는 자유롭게 코드가 실행 될수 있으므로 작업들이 이 영역에서 이루어 지도록 많이 사용하고 있습니다.
- 그러면 위 코드의 실행 결과는 어떻게 될까요? 각 영역이 실행되다가 마지막에 **생성자 내부**영역에서 덮어쓰기 된 결과가 출력됩니다.

```cpp
num1 : 3
num2 : 3
2 2 2
``` 

<br>

### **5. 위임 생성자 (delegating constructor)**

- 위임 생성자란 무엇일까요? 위임 생성자는 **생성자가 다른 생성자를 사용하는 것**을 위임 생성자 라고 합니다.
- 위임 생성자는 왜 사용하는 것일까요? 그 이유를 알기 위해 다음 코드를 한번 살펴보겠습니다.

<br>

```cpp
#include <iostream>
#include <string>

using namespace std;

class Student {

	int id_;
	string name_;

public:

	Student(const int& id, const string& name)
		: id_(id), name_(name) {}

	Student(const string& name)
		: Student(0, name) {}

	void print() {
		cout << "ID : " << id_ << ", " << "Name : " << name_ << endl;
	}
};

int main() {

	Student student("Jinsol");
	student.print();
}
```

<br>

- 위 코드를 보면 `Student` 클래스 안에 2개의 생성자가 있음을 알 수 있습니다.
- 이 때, 2번째 생성자의 **멤버 초기화 리스트**를 보면 `Student(0, name)`이 있습니다.
- 즉 생성자의 멤버 초기화 리스트에 생성자를 넣어서 초기화 함으로써 생성자가 실행될 때, 다른 생성자를 실행시키는 것입니다.
    - 즉 생성자의 역할을 다른 생성자에게 위임시키는 것입니다. 
- 이렇게 하는 이유는 무엇일까요?

<br>

```cpp
Student(const int& id, const string& name)
    : id_(id), name_(name) {}

Student(const string& name)
    : id_(0), name_(name) {}
```

<br>

- 만약 생성자 2개를 위 처럼 구성하였다면 기능상으로는 문제가 없지만 사실상 별로 좋지 못한 코드입니다.
- 왜냐하면 id랑 name 모두를 초기화 하는 똑같은 기능의 코드가 여러 곳에 있기 때문입니다. 이렇게 코드를 짜면 코드 관리 및 디버깅에 상당히 불리해 집니다.
- 따라서 코드 관리의 용이성 때문에 같은 역할(멤버 변수 초기화)을 하는 코드는 한 곳(한 생성자)안에 두기 위해서 `위임 생성자`를 이용합니다.

<br>

### 6. 복사 생성자

<br>

- 함수의 매개변수로 어떤 변수나 객체를 전달할 때, 레퍼런스로 전달하는 것이 아니라 값으로 전달하는 경우 함수 호출 시 `복사`가 발생합니다.
- 예를 들어 함수 호출 시 객체를 값으로 전달하는 경우 복사로 인하여 새로운 메모리 영역에 객체를 할당하게 되므로 원래 객체와 함수에서 사용되는 객체의 주소값이 달라지게 됩니다.  

<br>

```cpp
#include <iostream>

using namespace std;

class Something {

public:

	Something() {
		cout << "Constructor" << endl;
	}
};

void func(Something s) { 

	cout << &s << endl;

}

int main() {

	Something s;
	cout << &s << endl;

	func(s);

}
```

<br>

- 위 코드의 출력은 다음과 같습니다. (주소값은 개인마다 다릅니다.)

<br>

```
Constructor
06DFFE43
06DFFE3C
```

<br>
 
- 그런데 조금 이상한 점은 값 복사가 일어나서 객체가 새로 생성이 되면 위 코드를 기준으로 "Constructor"가 2번 출력이 되어야 하는데 1번 출력 된것으로 보아서 파라미터로 받은 객체를 복사할 때에는 생성자 호출이 안된다는 것을 유추할 수 있습니다.
- 이것은 `복사 생성자(Copy Constructor)`를 내부적으로 사용한 것입니다.
- **복사 생성자**는 생성자와 동일한 형태이지만 파라미터로 **레퍼런스 타입의 객체**를 받습니다. 예를 들어 다음과 같습니다.

<br>

```cpp
class Something {

	int n;

public:

	// 생성자
	Something() {
		cout << "Constructor" << endl;
		n = 0;
	}

	// 복사 생성자
	Something(const Something& something) {
		cout << "Copy Constructor" << endl;
		this->n = something.n;
	}
	
	int getValue() {
		return n;
	}
};

int main() {

	Something s1;
	cout << &s1 << endl;
	cout << s1.getValue() << endl;

	Something s2 = s1;
	cout << &s2 << endl;
	cout << s2.getValue() << endl;
}
```

<br> 

- 위 코드의 복사 생성자를 보면 파라미터로 `const Something& something` 즉, 레퍼런스 타입의 객체를 받습니다.
    - `const`를 사용한 것은 레퍼런스로 불러왔기 때문에 값이 변경되지 않기 위한 안전한 방법이기 때문입니다.
- 레퍼런스로 참조만 하여 복사하려는 객체의 값을 참조하여 복사할 객체에 값을 복사해 주는 역할을 합니다.
- 만약 함수 호출 시 값 복사가 일어나거나 또는 간단하게 현재 존재하는 객체를 새로운 객체에 할당(복사)하려고 할때, 생성자가 호출되지 않고 복사 생성자가 호출됩니다.
- 그러면 위 코드와 같이 복사 생성자 안에서 멤버 변수의 값을 복사해 주면 됩니다.
- **만약 복사 생성자를 만들지 않았다면** 위와 같은 작업은 자동적으로 처리됩니다. 이것이 복사 생성자 입니다.  

<br>

### **7. 소멸자 (destructor)**

<br>

- 클래스에서 소멸자란 생성자와 반대 개념의 역할을 하는 기능이라고 할 수 있습니다.
- 생성자는 객체가 메모리에 잡힐 때, 수행하는 함수라고 생각한다면 소멸자는 객체가 메모리에서 사라질 때 수행되는 함수라고 생각하면 됩니다.
- 즉, 생성자에서는 객체가 생성될 때 필요한 `초기화`를 한다고 하면 소멸자에서는 객체가 사라질 때, `깔끔하게 객체 제거`를 끝내기 위해서 존재합니다.
- 깔끔하게 제거한다는 것은 무슨 의미일까요? 만약 C++에서 `new`를 통하여 메모리 생성을 하였다면 `delete`를 통하여 메모리를 해주어야 메모리 누수가 발생하지 않습니다.
- 만약 이런 작업을 매번 수작업으로 한다면 귀찮을 뿐더러 놓칠 수도 있어서 사용하지 않는 변수가 계속 메모리에 잡혀져 있는 비효율성을 야기합니다.
- 이런 문제점들을 개선하기 위하여 소멸자에 `delete` 명령어로 메모리 해제를 해주는 역할이 소멸자에서 사용되는 가장 큰 역할입니다.
    - 메모리 해제 조건은 객체가 존재하는 영역이 끝났을 때입니다. 예를 들어 중괄호 안에 있는 객체는 중괄호를 벗어나면 객체는 소멸됩니다.
- 아래 코드를 한번 살펴보도록 하겠습니다.

<br>

```cpp
#include <iostream>
#include <string>

using namespace std;

class Sample {

	int id_;

public:

	Sample(const int& id) :id_(id) {
		cout << "Constructor : " << id_ << endl;
	}
	~Sample() {
		cout << "Destructor : " << id_ << endl;
	}

};

int main() {
	  
	Sample s1(1);
	Sample s2(2);

}

``` 

<br>

- 먼저 위와 같은 코드가 있으면 `~Sample(){}`이 바로 소멸자 역할 입니다.
- 소멸자는 객체가 소멸될 때 소멸이 되는데 어떤 순서로 소멸이 되는지 위의 코드를 실행해 보겠습니다.
- 일단 객체가 소멸되는 순서대로 소멸자가 실행됩니다.

<br>

```
Constructor : 1
Constructor : 2
Destructor : 2
Destructor : 1 
```

<br>

- 위 코드를 실행하면 생성자와 반대 순서로 소멸자가 실행됩니다. 마치 객체를 생성할 때, 스택에 쌓아 놓고 Last In First Out 순서로 객체가 제거되면서 소멸자가 실행됩니다.

<br>

```cpp
Sample* s1 = new Sample(1);
Sample s2(2);

delete s1;

Constructor : 1
Constructor : 2
Destructor : 1
Destructor : 2
```

<br>

- 위 코드의 메인 부분만 실행하게 되면 위와 같은 순서로 소멸자가 실행 됩니다. `delete s1`을 통하여 s1을 먼저 메모리 해제해 주었기 때문에 먼저 소멸자가 실행되었습니다.
- `new`를 이용하여 동적 할당을 하게 되면 반드시 `delete`를 이용하여 제거를 해주어야 메모리 누수가 없게 되는데 사람이 매번 해주기가 어려우므로 반드시 소멸자에 `delete`를 지정해 놓아야 문제가 생기지 않습니다.
- 또는 `STL`을 이용하면 `STL`내부적으로 소멸자에 `delete` 역할이 수행되기 때문에 메모리 누수를 걱정할 필요는 없으니 잘 만들어 놓은 `STL`을 쓰길 추천드립니다. 

<br>

```cpp
#include <iostream>
#include <string>

using namespace std;

class Array {

	int* arr_ = nullptr;
	int length_ = 0;

public:

	Array(const int length) {
		length_ = length;
		arr_ = new int[length_];

		cout << "Constructor" << endl;
	}

	/*~Array() {
		delete arr_;
	}*/


};

int main() {
	  
	while (true) {
		Array arr(100);
	}

}
```

<br>

- 예를 들어 위와 같이 코드를 만든다면 소멸자에서 메모리 해제가 일어나지 않아서 아래와 같이 메모리 누수가 계속 발생하게 됩니다.

<br>
<center><img src="../assets/img/cpp/concept/oop/1.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

### **8. this 포인터와 연쇄 호출(chaining)**

<br>

- `this` 포인터는 클래스 내에서 클래스 자체를 가리키는 포인터에 해당합니다. 
- 만약 파이썬을 써보셨다면 파이썬 클래스에서 `self`와 동일한 역할을 하는 것으로 이해하시면 됩니다.
- **this**를 사용하면 클래스 내부의 함수 또는 멤버 변수를 명확하게 지칭할 수 있고 또 다른 장점으론 연쇄 호출이 가능해 집니다.
- 아래 코드를 한번 살펴 보시면 **this** 포인터의 역할을 잘 이해할 수 있습니다.

<br>

```cpp
class Simple {

	int id_;

public:

	Simple(int id) {
		this->setID(id); // setID(id)와 동일함
	}

	void setID(int id) {
		this->id_ = id; // id_와 동일함
	}

};
```

<br>

- 위 클래스의 코드를 보면 **생성자**와 **setID** 함수 내에 `this`라는 키워드를 볼 수 있습니다.
- 여기서 **this**는 클래스 내부의 함수나 멤버 변수를 접근하기 위해 사용되었습니다.
- 사실상 위와 같은 상황에서는 this를 사용하지 않아도 똑같은 결과가 나오는데, 그 이유는 **this**는 클래스 자체의 주소를 저장하고 있기 때문에 클래스 자신을 가리키고 있기 때문입니다.
    - 즉, 자기 자신을 명확히 지칭한 다음에 그 안에 있는 멤버 변수 또는 함수를 접근 하는 것이지요.
- 이런 **this** 포인터는 명확하게 클래스 내부의 멤버 변수나 함수를 지칭하기 위해서도 사용되지만 사실 더 좋은 용도로는 `연쇄 호출`이 있습니다.

<br>

```cpp
#include <iostream>
#include <string>

using namespace std;

class Calc {

	int value_;

public:

	Calc(int value) :value_(value) {}

	void add1(int value) {
		value_ += value;
	}
	
	Calc& add2(int value) {
		value_ += value;
		return *this;
	}

};

int main() {

	Calc c1(0);
	Calc c2(0);

	c1.add1(10);
	c1.add1(10);

	c2.add2(10).add2(10);	

}
```

<br>

- 클래스 내부의 **add1**함수를 보면 리턴 타입이 void 이므로 덧셈 연산을 매번 해줄 때 마다 함수를 호출해 주어야 합니다.
- 반면 **add2**함수를 보면 리턴 타입이 레퍼런스이므로 이 레퍼런스를 받아서 연달아 함수 호출이 가능해져서 편리하게 사용할 수 있습니다.
- 이 방법을 연쇄 호출 또는 chaining 이라고 합니다. 이 과정을 풀어 쓰면 다음과 같습니다. 이해하기가 편하지요?

<br>

```cpp
Calc c(0);
Calc &temp1 = c.add(10);
Calc &temp2 = temp1.add(10);
```

<br>

- 레퍼런스 타입으로 받아서 계속 호출하는 것으로 **this**를 이용해서 연쇄 호출 하는 것과 완전히 동일하다고 할 수 있습니다.      

<br>

### **9. 클래스 코더와 헤더 파일**

- 클래스 내부에 생성자, 소멸자, 함수들 모두를 다 정의해 놓으면 한 코드 파일의 코드가 너무 길어져서 읽기가 어려워 집니다.
- 일반적으로 한 개의 클래스는 **1개의 헤더 파일, 1개의 CPP 파일**을 이용하여 만듭니다.
- **헤더 파일**에는 클래스의 선언과 및 함수의 정의 등만 두고 **CPP 파일**에는 함수의 상세 코드를 따로 정의해 놓습니다.
- 예를 들어 다음과 같이 한 코드에 클래스의 정의와 코드를 모두 둔 코드를 한번 보겠습니다.  

<br>

```cpp
class Calc {

	int value_;

public:

	Calc(int value) :value_(value) {}

	Calc& add(int value) {
		value_ += value;
		return *this;
	}

	Calc& subtract(int value) {
		value_ -= value;
		return *this;
	}

	Calc& multiply(int value) {
		value_ *= value;
		return *this;
	}

	Calc& divide(int value) {
		value_ /= value;
		return *this;
	}
};
```

<br>

- 아주 간단한 역할만 하는데도 코드 량이 상당이 깁니다. 비주얼 스튜디오등과 같이 에디터의 기능을 이용하여 함수 접기 기능이 있으면 보기 편하지만 그렇지 않으면 상당히 보기 번거롭습니다.
- 따라서 많이 사용하는 방법인 **헤더 파일과 cpp 파일로 분리**하는 방법을 통하여 코드를 보기 좋게 나눠보겠습니다.

<br>

```cpp
// Calc.hpp
class Calc {

	int value_;

public:

	Calc(int value) :value_(value) {}
	Calc& add(int value);
	Calc& subtract(int value);
	Calc& multiply(int value);
	Calc& divide(int value);
};


// Calc.cpp
Calc& Calc::add(int value) {
	value_ += value;
	return *this;
}

Calc& Calc::subtract(int value) {
	value_ -= value;
	return *this;
}

Calc& Calc::multiply(int value) {
	value_ *= value;
	return *this;
}

Calc& Calc::divide(int value) {
	value_ /= value;
	return *this;
}

``` 

<br>

- 코드를 위와 같이 header와 cpp 파일로 나누어 놓으면 main에서는 header만 읽어 와서 클래스를 사용할 수 있고 header에서는 클래스의 정의된 부분과 주석등을 통해서 클래스 사용방법을 알 수 있습니다.
- 만약 각 함수의 상세 내용을 알고 싶으면 cpp 파일에서 찾아보면 됩니다. 이렇게 구분해 놓으니 코드가 좀 더 역할 별로 간결해 졌다고 볼 수 있습니다. 
- 참고로 비주얼 스튜디오에서 위와 같이 쉽게 나누려면 클래스 내부에 선언된 함수 정의와 상세 내용을 모두 블록을 씌운 다음 마우스 오른쪽 키를 누르면 `Quick Action and Refactoring`이라는 메뉴가 가장 상단에 뜹니다.
- 이것을 클릭한 후 `Move definition location`을 클릭하면 쉽게 함수 정의 부분과 코드 부분이 분리 됩니다. 아래 그림을 참조하세요.

<br>

<center><img src="../assets/img/cpp/concept/oop/2.PNG" alt="Drawing" style="width: 800px;"/></center>

<br>

### **10. 클래스와 const**

<br>

- 이번 글에서는 클래스를 `const`로 선언하였을 때, 어떻게 사용하는 지 알아보겠습니다.
- 먼저 다음과 같이 코드를 선언하면 오류가 발생합니다. 

<br>

```cpp
class Something {

	int value_;

public:

	void setValue(int value) {
		value_ = value;
	}

	int getValue() {
		return value_;
	}
};

int main() {
	const Something some;
	some.getValue();
}
```

<br>

- 오류가 발생하는 이유는 클래스가 **const** 타입으로 선언되었는데 클래스 내의 함수가 **const** 타입이 아니기 때문입니다.
- **const** 타입의 클래스 함수는 `반환형 함수명() const {}`와 같이 선언하면 **const** 타입이 됩니다. 예를 들어 다음과 같습니다.

<br>

```cpp
class Something{

    int n;
    
public:
    
    int func() const{
        ...
        return n;
    }
}
```
    
<br>

- 클래스 함수에서 사용된 **const**는 어떤 의미를 지닐까요? 일단 **const** 타입의 객체에서 사용될 수 있다는 의미가 있습니다.
    - 즉, **const** 타입의 객체에서 함수를 호출하려면 반드시 **const** 타입으로 선언되어 있어야 합니다.
- 그리고 **const** 타입의 함수 안에서는 값을 변경하려는 작업이 금지 됩니다. 예를 들어 다음과 같은 작업은 오류가 발생합니다.

<br>

```cpp
class Something{
    
    int n_;
    
public:
    
    void setValue(int n) const{
        n_ = n; // 오류 발생
    }
    
    int getValue() const{
        return n_; // 오류 발생하지 않음
    } 
}

```

<br>

- 위와 같이 `const`를 사용하면 실수로 값을 변경하는 코드를 작성하지 않을 수 있습니다. 따라서 최대한 `const`는 사용해주는 것이 좋을 수 있습니다.
- 그러면 리턴 타입에 `const`를 사용하는 것은 어떤 의미가 있을까요? 다음 코드를 한번 살펴보겠습니다.

<br>

```cpp
class Something {

	int n_ = 0;

public:

	const int& getVal() const{
		cout << "const version" << endl;
		return n_;
	}

	int& getVal() {
		cout << "non-const version" << endl;
		return n_;
	}
};

int main() {

	Something s1;
	const Something s2;
	cout << (s1.getVal() = 3);
	//s2.getVal() = 3; : const 리턴 타입 값은 수정 자체가 안됨
	
}
```

<br>

- 위 코드처럼 return 타입에 **const** 설정을 해놓으면 리턴 시 값 수정이 안됩니다.
- 그리고 위 두개의 함수는 함수명이 같아서 오버로딩이 된 상태입니다. 만약 이 때, `const int& getVal() const{}`에서 마지막 **const**를 빼서 `const int& getVal() {}`으로 두면 오버로딩이 안되니 참조하시기 바랍니다.

<br>

### **11. 정적 멤버 변수**

<br>

- 이번에는 클래스에서 사용되는 `정적 멤버 변수`에 대하여 배워보도록 하겠습니다.
- 먼저 아래 코드를 한번 실행해 보시겠습니다.

<br>

```cpp
class Something {

public:

	int m_value = 1;
};

int main() {

	Something st1;
	Something st2;

	st1.value_ = 2;

	cout << &st1.m_value << " " << st1.m_value << endl;
	cout << &st2.m_value << " " << st2.m_value << endl;
}
```

<br>

- 위 코드를 보면 두 객체의 주소와 멤버 변수 값을 출력하게 됩니다. 이 때, 객체 **st1**과 **st2**의 주소값과 멤버 변수 값은 서로 다르게 출력되는 것을 확인하실 수 있습니다.
- 이번에는 멤버 변수를 `정적 변수` 타입으로 선언해 본 다음에 아래 코드와 같이 실행해 보겠습니다.
- 참고로 **정적 변수**는 클래스 내부에서 선언과 동시에 초기화가 되지 않으므로 외부에서 초기화 해주어야 합니다. (내부에서 초기화 하는 방법은 조금 있다가 다루겠습니다.)

<br>

```cpp
class Something {

public:

	static int s_value;
};

int Something::s_value = 1;

int main() {

	Something st1;
	Something st2;

	st1.s_value = 2;

	cout << &st1.s_value << " " << st1.s_value << endl;
	cout << &st2.s_value << " " << st2.s_value << endl;
}
```

<br>

- 위 코드를 실행해 보면 두 객체의 멤버 변수 `s_value`의 주소값과 값이 모두 같은 것을 확인하실 수 있습니다. 즉, **서로 다른 객체가 같은 멤버 변수를 공유하고 있는 셈**입니다.

<br>

```cpp
class Something {

public:

	static int s_value;
};

int Something::s_value = 1;

int main() {

	cout << &Something::s_value << " " << Something::s_value << endl;

	Something st1;
	Something st2;

	st1.s_value = 2;

	cout << &st1.s_value << " " << st1.s_value << endl;
	cout << &st2.s_value << " " << st2.s_value << endl;
}
```

<br>

- 더 재미있는 부분은 객체를 생성하지 않고도 **main**함수의 첫 출력문이 오류 없이 실행됩니다. 

<br>

```
00B0C008 1
00B0C008 2
00B0C008 2
```

<br>

- 예를 들어 위와 같은 결과를 얻을 수 있는데, 결국 **정적 멤버 변수**는 클래스에서 선언되면 모든 객체에서 공유되며 객체가 생성되지 않더라도 그 값은 생성되어 있음을 알 수 있습니다.
- 위의 코드에서 `int Something::s_value = 1;`으로 정적 변수의 값을 초기화 해주는 코드가 있는데 이런 역할은 헤더 파일이 아닌 `cpp`파일에 두는 것이 일반적이오니 참조하시기 바랍니다.

<br>

```cpp
class Something {

public:

	static const int s_value = 1;
};

// int Something::s_value = 1; 
```

<br>

- 반면 멤버 변수가 `static` 이고 `const`이면 이 때에는 클래스 내부에서 초기화를 해주어야 합니다. 약간 헷갈릿 수 있겠지만 상식적으로 생각해보면 수긍이 갈 것입니다.
    - **const**의 경우 초기화 값이 바뀌면 안되므로 선언과 동시에 값이 정해져야 하기 때문입니다.
- 정리하면, `static 멤버 변수`는 선언과 동시에 초기화가 안되는 반면, `static const 멤버 변수`는 선언과 동시에 초기화가 되고 그렇게 꼭 해주어야 합니다.

<br>

### **12. 정적 멤버 함수**

<br>

- 이번에는 `정적 멤버 함수`에 대하여 알아보도록 하겠습니다. **정적 멤버 함수**는 **정적 멤버 변수**와 연관되어서 사용됩니다. 아래 코드를 참조해 보겠습니다.

<br>

```cpp
class Something {

	static int s_value;

public:
	
	int getValue() {
		return 3;
	}
};

int Something::s_value = 10;

int main() {

	Something st1;
	cout << st1.getValue() << endl;
	cout << Something::getValue() << endl; //오류 발생
}
```

<br>

- 위 코드에서 마지막 출력문은 에러가 발생하게 되는 반면 첫번째 출력문은 정상 작동합니다.
- **static** 멤버 변수를 접근할 때, 객체를 생성하여 접근할 때와 클래스를 직접 접근할 때 방식이 조금 다릅니다. 
    - 심지어 `getValue()`에서 **static**변수를 사용하지 않아도 에러가 발생합니다.
- 이 에러를 해결하려면 `static int getValue(){}`형태로 **static 멤버 함수**를 만들어야 합니다. 

<br> 

```cpp
class Something {

	static int s_value;
	int m_value;

public:
	
	static int getValue() {
		s_value; //정상 접근
		m_value; //오류 발생
		this; //오류 발생
	}
};
```

<br>

- 또한 `static 멤버 함수`에서 사용가능한 변수는 **static 멤버 변수**만 사용가능합니다. 
- 일반 멤버 변수도 사용 불가능하고 특히, `this`를 통해 접근 가능한 모든 것이 사용 불가합니다.(그래서 일반 멤버 변수도 사용 불가한 것이지요)
    - **this**에 발생한 에러를 읽어보면 `this may only be used inside a nonstatic member function`
- 이와 같이 **static 멤버 함수**는 상당히 많은 제약이 있습니다. 실질적으로 이런 불편함이 있어서 **static member function**을 자주 사용하지는 않습니다. 
- 하지만, 왜 이런 문제가 발생하는지는 한번 확인해 보도록 하겠습니다.
 

<br>

### **13. 친구 함수와 클래스 friend**
 

<br>

※ 참조 자료 : 홍정모의 따라하며 배우는 C++