---
layout: post
title: 1. View C++ as a federation of languages (C++는 언어들의 연합체이다.)
date: 2019-09-03 00:00:00
img: cpp/effective.png
categories: [c-cpp] 
tags: [cpp, c++, effective c++] # add tag
---

- 이번 글에서는 이펙티브 C++의 첫번째 주제인 **View C++ as a federation of languages**에 대하여 다루어 보겠습니다.

<br>

- 제목에 나와있는 뜻과 같이 C++을 언어들의 연합체로 바라보라는 것이 첫번째 주제입니다.
- C++은 원래 객체 지향 C언어 라고 이름을 지으려고 한 것처럼 **C에다가 객체 지향의 특성을 더한 것**이라고 말할 수 있습니다.
- C++은 발전해 가면서 점차 C언어와 다른 방향으로 발전하게 되었는데 그것은 C++만의 특성이 더해졌기 때문입니다.
- 예를들어 C++의 `Template`은 새로운 디자인에 대한 관점을 제공해 주었고 `STL`은 확장성에 새로운 접근 방법을 제시해 주었습니다.
- 오늘날에 C++을 어떤 언어의 관점으로 바라봐야 할 지 간혹 혼동스럽긴 한데(절차지향인지, 객체지향인지, 함수형인지 등등) C++은 여러 하위 언어의 복합체라고 생각하는 것이 가장 좋습니다.
- 사실 책에서는 언어들의 연합체라고 표현해 놓았지만 개인적으로는 여러가지 특성의 연합체라고 하는 것이 다 낫다고 생각은 합니다. 그러면 그 4가지 특성에 대하여 알아보도록 하겠습니다.

<br>

- `C언어`: 
    - C++자체는 C언어에 기반이 되어 있기 때문에 C언어의 관련 속성들은 당연히 C++에서도 중요합니다. 
    - 더 나아가서 C++에는 C에 비하여 다양한 기능들이 추가되었기 때문에 C에서는 생각하기 어려웠던 많은 문제들의 해결책을 C++에서는 다양한 방법으로 해결할 수 있습니다.

- `Object - oriented C++`:
    - 클래스 기능을 이용하면 **Object oriented programming**이 가능해집니다.
    - C++에서는 객체 지향의 성질들이 추가되면서 다양한 문제를 객체 지향의 방법으로 해결할 수 있습니다. 객체 지향의 대표적인 성질은 다음과 같습니다.
    - constructor, destructor, encapsulation, inheritance, polymorphism, virtual function 등.
    - 특히 여기서 **encapsulation, inheritance, polymorphism**은 객체 지향의 3대 원리로 알려진 만큼 가장 중요한 기능들입니다.

- `Template C++`:
    - Template란 기능을 이용하면 **Generic programming**이 가능해집니다.
    - Template 기능을 이용하면 새로운 방식의 프로그래밍이 가능해지고 그것을  **Template metaprogramming**이라고 합니다.
- `STL`:
    - STL은 Standard Template Library의 약자입니다.
    - 즉, STL은 Temlplate Library이고 Template의 성질과 다른 C++의 기능들을 잘 어우러져서 사용하기 매우 편리한 library들을 만들어 냅니다.

<br>

- 위에서 언급한 4가지의 큰 기능이 C에서 C++로 넘어오게 되면서 추가되었습니다.
- 이 기능들의 추가는 완전히 새로운 언어가 되었을 만큼 큰 차이를 만들었기 때문에 기존의 C에서 사용한 관습들이 C++에서는 비효율적일 수 있습니다.
- 바꾸어 말하면 C++에서 효율적인 프로그래밍을 하기 위해서는 C++의 기능에 맞게 프로그래밍을 해야 효율적인 프로그램을 만들 수 있습니다.
- 예를 들어, C에서는 함수 파라미터로 Pass-by-Value를 하는 것이 효율적이었습니다.(왜 효율적인지 잘 기억이 안나네요...)
- 반면에 C++에서는 Pass-by-Value는 비효율적입니다. Pass-by-Value를 이용하면 값이 복사되어 메모리 낭비가 발생하고 객체의 경우 (copy) constructor와 destructor가 추가적으로 실행되기도 하기 때문입니다.

<br>

- 이번 주제에서 다시 언급하려는 것의 요지는 C++은 C의 성질 뿐만 아니라 Object-oriented, Template, STL등의 큰 기능이 추가되었으므로 효율적인 프로그래밍을 하기 위해서는 C의 기존 관습대로 프로그래밍을 하기 보다는 C++에 맞도록 하는 것이 효율적인 프로그램에 아주 중요하다고 할 수 있겠습니다.
