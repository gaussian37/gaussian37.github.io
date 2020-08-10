---
layout: post
title: 2. Prefer const, enum and inline to define (define 대신 const, enum, inline을 사용하자.)
date: 2019-09-13 00:00:00
img: cpp/effective.png
categories: [c-cpp] 
tags: [cpp, c++, effective c++] # add tag
---

- 이번 글에서는 이펙티브 C++의 2번째 주제인 **Prefer const, enum and inline to #define**에 대하여 다루어 보겠습니다.
- 책에서 언급하길 이번 주제의 제목은 **Compiler**가 **Preprocessor**보다 낫다라고 하는 것이 더 나은 제목이라고 하기도 하였습니다.
- 여기서 비교가 되는 것은 `#define`과 `const, enum, inline` 인데, `#define`의 경우 랭귀지 자체로 취급되지 않기 때문에 문제가 발생하여 `const, enum, inline`을 추천한다고 적혀있습니다.
- 간단히 말해 `#define` 사용을 피하자. 라고 생각하면 되곘습니다.

<br>

### #define을 이용하여 `constant`를 선언하지 말 것

<br>

- 예를 들어 다음과 같은 코드가 있다고 가정해 보겠습니다

<br>

```cpp
#define ASPECT_RATIO 1.653
```

<br>

- 위 코드에서 정의된 symbolic name인 **ASPECT_RATIO**는 compiler가 보지 않습니다. 왜냐하면 compiler가 source code를 보기 전에 preprocessor에 의해서 제거되기 때문입니다. 그 이유는 `#define`을 C++ 언어 구성요소로 포함시키지 않기 때문에 컴파일러로 전달하지 않습니다.
- 결과적으로 symbolic name인 **ASPECT_RATIO**는 symbolic name을 관리하는 symbol table에 저장되지 않습니다.
- 이 동작 방식은 만약에 컴파일 단계에서 에러가 발생하면 에러 메시지는 symbolic name을 반환하지 않고 정의된 숫자를 그대로 반환하게 됩니다. 즉, 에러 메시지에 **ASPECT_RATIO**는 찾아볼 수 없고 1.653만 있게 됩니다. 이렇게 되면 디버깅 하기 상당히 까다로워집니다.
- 따라서 `macro`를 사용하기 보다 `const`를 사용하는 것이 디버깅 측면에서 효율적인 프로그래밍 방법입니다.

<br>

```cpp
const double AspectRatio = 1.653;
```

<br>

- 위와 같이 코드를 작성하면 const상수 **AspectRatio**는 compiler로 전달됩니다. 앞의 `#define`과의 차이점은 `const`는 C++ 언어 자체의 구성 요소로 취급하여 compiler로 전달되어 compile 된다는 차이점이 있습니다.
- 추가적으로 `#define`보다 `const`가 더 효율적인 이유로 preprocessor 단계에서 `#define`으로 정의된 macro 상수는 preprocessor가 사용할 때 매번 값을 복사하여 사용하도록 내부적으로 설계되어 있는 반면에 const 상수는 중복된 복사가 발생하지 않도록 설계되어 있는 이유도 있습니다.

<br>

- 지금까지 `const`상수를 사용하는 것이 좋고 `#define` 매크로 상수를 사용하는 것이 좋지 않다 라는것에 대하여 다루어 봤는데, 이 방법이 어떤 경우에 효율적으로 사용될 수 있는지 한번 알아보겠습니다.
- 먼저, `constant pointer`를 선언할 때입니다. ([constant pointer 참조](https://gaussian37.github.io/c-concept-const_char_pointer/))
- 보통 상수들은 특정 헤더파일에 모아두게 되는데 그 때 `#define`으로 상수들을 모으지 말고 `const`로 헤더 파일에 모으고 특히 **constant pointer**나 **pointer to constant**를 선언할 때에는 특히 `constant`를 이용해야 합니다.
    - 예를 들어 `const char * const authorName = "Scott Meyers";`와 같이 사용하여 **constant pointer, pointer to constant, constant pointer to constant**를 선언합니다.
- 첨언하면 이렇게 사용하는 문자열 방식은 `C언어` 방식의 약간 올드한 방식으로 `C++`에서는 string을 이용하여 상수를 만드는 것이 더 좋습니다.
    - 예를 들면 `const std::string authorName("Scott Meyers");`

<br>

- 그 다음으로 상수를 사용하면 좋은 경우는 **class-specific constant**입니다.
- 만약 `constant`를 특정 클래스에서만 사용하도록 범위를 제한하려면 클래스 내부에서 `constant`를 선언해야 합니다.
- 이 때, 멤버 변수로 상수 형태로 선언해야 하고 특히, 아래 코드와 같이 상수의 사용이 발생하면 `static`형태로 선언되어져야 합니다.

<br>

```cpp
class GamePlayer {

private:
	static const int NumTurns = 5; // 상수 선언
	int scores[NumTurns]; // 상수 사용
};
``` 

<br>

- 다루어야할 내용은 **class-specific constant**를 사용하는 것이 `#define`을 사용하는 것 보다 더 낫다라는 것인데, 잠시 위 코드를 자세히 한번 살펴보고 지나가겠습니다.
- 위 코드에서 **NumTurns**는 `declaration`이지 `definition`이 아닙니다. ([`declaration`과 `definition`의 차이](https://gaussian37.github.io/c-etc-declaration_definition/)) 
- 보통 C++에서는 사용하려는 변수의 `definition`이 필요합니다. 하지만 `static` 타입과 `integral` 타입(ex. ing, char, long..)의 **class-specific constant**는 예외입니다.
- 만약 주소값을 사용하지 않는다면, `definition` 없이 `declare`만 하여 변수를 사용할 수 있습니다.
- 만약 주소값을 사용해서 `definition`을 해야하거나 또는 컴파일러가 `definition`을 강제로 원하는 경우에는 다음과 같이 `definition`을 정할 수 있습니다.

<br>

```cpp
const int GamePlayer::NumTurns; //definition
```

<br>

- 위 코드는 보통 헤더 파일이 아니라 클래스가 선언된 파일에 같이 넣습니다.
- 왜냐하면 클래스의 상수들의 초기값은 상수들이 `declaration`된 곳에서 초기화가 되어야 하기 때문입니다.
- 위 코드에서는 `static const int NumTurns=5;`로 클래스 내부에서 선언과 동시에 초기화가 되었습니다.
- 만약 클래스 내부에서 선언과 동시에 초기화가 안되었다면 `definition`에서 초기화가 되었어야 합니다.
    - 예를 들면 `const int GamePlayer::NumTurns = 5;`와 같이 정의되어야 합니다.
- 정리하면 다음 둘 중 한가지 경우를 따라야 합니다.
- 첫번째, 클래스 내부에서 `declaration` 시 초기화를 하고 클래스 외부에서 `definition`을 합니다.

<br>

```cpp
class GamePlayer {

private:
	static const int NumTurns = 5; // 상수 선언

public:

	static int getNum() {
		return NumTurns;
	}
};

const int GamePlayer::NumTurns;

```

<br>

- 두번째, 클래스 외부의 `definition`에서 초기화를 합니다.

<br>

```cpp
class GamePlayer {

private:
	static const int NumTurns; // 상수 선언

public:

	static int getNum() {
		return NumTurns;
	}
};

const int GamePlayer::NumTurns = 5;
```

<br>

- 오래된 컴파일러 버전에는 첫번째 케이스가 오류가 날 수도 있습니다. (아마 요즘 사용하는 대부분의 컴파일러는 첫번째, 두번째 방법 모두 사용가능할 것입니다.)
- 이전에는 `static` 클래스 멤버 변수를 `declaration` 동시에 초기화 하는것이 원칙이 아니었기 때문입니다.

<br>

- 다시 본론으로 돌아오면, 위에서 설명한 **class-specific constant**를 사용하는 것이 `#define`을 이용한 상수를 사용한 것 보다 더 낫다는 것입니다.
- 왜냐하면 `#deinfe`은 특정 클래스에서만 사용할 수 있도록 범위를 제한할 수 없습니다. 일단 매크로 상수가 만들어지면 모든 클래스에서 강제로 사용되어 집니다.
- 또한 클래스의 `encapsulation` 역할도 해낼 수 없기 때문에 C++에서 사용되는 장점들이 무시되는 문제도 있습니다. 즉, private 타입의 매크로가 없다는 것이 문제입니다.
- 반면에 **const data member**는 `encapsulation`이 됩니다.

<br>

- 위의 코드에서 사용한 바와 같이 클래스 내부에서 배열을 선언하는 것과 같이 `constant`가 필요한 경우가 종종 있습니다.
- 예전에는 클래스 내부에서 `static integral class constant` 타입에 대한 `declaration`과 동시에 초기화를 하는 것이 컴파일러에 의해 금지되었기 때문에 우회해서 사용하는 방법이 있었는데 그것이 `enum hack`이라는 방법입니다.
- 이 방법은 정의해야하는 `constant`가 `int` 타입인 경우에 사용이 가능한 방법입니다.

<br>

```cpp
class GamePlayer{

private:

    enum {NumTurns = 5}; // 상수 5에 symbolic name을 만듭니다.
    int scores[NumTurns];
    ...
}
```

<br>

- 먼저 `enum hack` 방법은 클래스 내부에서 사용되기 때문에 `encapsulation`도 되고 특정 클래스에서만 사용하도록 할 수 있습니다.
- 반면 `enum hack`의 몇가지 특성은 `constant`보다 `#define` 처럼 동작합니다.
- 예를 들어, `const`의 주소값을 사용하는 것은 가능하지만 `#define`의 주소값을 사용하는것은 불가능합니다.
- 만약 의도적으로 상수의 주소값을 사용하거나 레퍼런스로 사용하려고 하는 것을 막으려면 `#define`이나 `enum`으로 상수를 선언하면 접근을 막을 수 있습니다.
- 또한 `const`의 경우 좋은 컴파일러에서는 integral 타입의 `constant`를 위한 저장 공간을 따로 만들어서 메모리 낭비를 하지 않지만 좋지 못한 컴파일러에서는 이렇게 따로 공간을 마련하여 메모리를 낭비하곤 합니다.
- 하지만 `#define`이나 `enum`을 위한 공간을 따로 두지 않기 때문에 메모리를 낭비를 줄일 수 있습니다. 물론 요즘의 좋은 컴파일러에서는 이런 문제는 없을 것입니다...
- `enum hack`의 또다른 장점으로 `template metaprogramming`에서 도움이 됩니다. 이 내용은 좀 길어지므로 이후의 글 (48. Be aware of template metaprogramming) 글에서 다루어 보겠습니다.

<br>

### #define을 이용하여 함수를 만들지 말것

<br>

- `#define` 매크로를 이용하여 함수를 선언하여 사용하는 경우가 있는데, 매크로 함수는 예상치 못한 결과를 종종내기 때문에 사용하지 않는 것을 권장합니다.
- 먼저 매크로 함수에서 모든 변수에 괄호를 같이 입력해줘서 강력하게 연산자 우선순위를 넣어줘야 한다는 불편함이 있습니다.
- 또한 매크로 함수는 일반 함수와는 조금 다르게 동작되는 것들이 있는데 예를 들면 다음과 같습니다.

<br>

```cpp
int f(int n) {
	return n;
}

#define CALL_WITH_MAX(a, b) f((a) > (b) ? (a) : (b))

int a = 5, b = 0;

int main() {

	CALL_WITH_MAX(++a, b); // a = 7, b = 0으로 출력됨, 즉 a가 2번 증가됨
	CALL_WITH_MAX(++a, b + 10); // a = 6, b = 0으로 출력됨, 정상적으로 a가 1번 증가됨
}
```

<br>

- 위 코드를 보면 의도치 않은 결과는 `CALL_WITH_MAX(++a, b);` 이 부분입니다. a가 2번 증가되었는데 이것은 매크로 함수 동작에 따라서 increment 연산이 2번되었기 때문입니다.
- 이와 같은 예를 보더라도 매크로 함수 사용은 권장되지 않습니다. 연산자 우선순위 때문에 괄호를 사용해야하는 불편함도 있기도 하구요.
- 대신에 `inline` 함수를 사용하는 것을 권장합니다. 위 매크로 함수를 `inline` 함수로 변경하면 다음과 같습니다.
- 물론 아래와 같은 형태로 함수를 사용하면 increment가 2번 발생하지 않습니다. 또한 함수 형태로 사용되기 때문에 특정 클래스 내부에서만 사용될 수 있습니다.

<br>

```cpp
int f(int n) {
	return n;
}

template<typename T>
inline void callWithMax(const T& a, const T& b) // know what T is, we
{
	f(a > b ? a : b);
}
```

<br>

- 이번 글에서 살펴 본것을 정리하면
    - `constant`를 사용하고 싶으면 `const` 또는 `enum`을 사용합시다.
    - 매크로 함수를 사용하고 싶으면 `inline function`을 사용합시다.
- 이렇게 `#define` 사용을 회피하면 `preprocessor`이 해야할 일이 줄어들게 되는 장점도 있습니다.
    - `preprocessor`는 중요한 `#include` 기능을 수행해야하고 `#ifdef/#ifndef` 기능도 수행해야 하므로 바쁩니다.
