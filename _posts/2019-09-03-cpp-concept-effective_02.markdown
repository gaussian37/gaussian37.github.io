---
layout: post
title: Prefer const, enum and inline to #define (#define 대신 const, enum, inline을 사용하자.)
date: 2019-09-13 00:00:00
img: cpp/effective.png
categories: [cpp-concept] 
tags: [cpp, c++, effective c++] # add tag
---

- 이번 글에서는 이펙티브 C++의 2번째 주제인 **Prefer const, enum and inline to #define**에 대하여 다루어 보겠습니다.
- 책에서 언급하길 이번 주제의 제목은 **Compiler**가 **Preprocessor**보다 낫다라고 하는 것이 더 나은 제목이라고 하기도 하였습니다.
- 여기서 비교가 되는 것은 `#define`과 `const, enum, inline` 인데, `#define`의 경우 랭귀지 자체로 취급되지 않기 때문에 문제가 발생하여 `const, enum, inline`을 추천한다고 적혀있습니다.
- 간단히 말해 `#define` 사용을 피하자. 라고 생각하면 되곘습니다.

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
- 먼저, `constant pointer`를 선언할 때입니다.