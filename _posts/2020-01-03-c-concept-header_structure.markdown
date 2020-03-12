---
layout: post
title: (제가 사용하는) 효율적인 header 구조 관리법
date: 2020-01-03 00:00:00
img: c/c.png
categories: [c-concept] 
tags: [c, c language, c 언어] # add tag
---

<br>

- 이번 글에서는 제가 사용하는 효율적으로 header를 관리하는 방법에 대하여 소개드리려고 합니다.
- 비효율적인 방법 또는 틀린 것이 있으면 댓글로 남겨주시면 정말 감사하겠습니다. 개선하는 것은 언제나 대 환영입니다.

<br>
<center><img src="../assets/img/c/concept/header_structure/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 저의 경우 위와 같은 형태로 코드를 관리합니다.
- 각각의 `c파일` 에서는 동일한 이름의 `header` 파일만을 인클루드 합니다.
    - 예를 들어 `test1.c`에서는 `#include "test1.h"`만 인클루드 합니다.
- 각각의 `header`에는 필요한 헤더를 모두 인클루드 합니다. 또한 `c코드`에서 사용되는 함수의 선언을 포함시킵니다.
    - 이 때, 각 헤더 파일들은 서로를 참조할 수 있습니다. 예를 들어 test1.h에서 test2.h 헤더를 포함시킬 수 있고 그 반대도 가능합니다.
- **많은 헤더 파일에서 공통적으로 사용해야 하는 코드 부분**은 위 그림처럼 `common.h`에 두고 사용합니다.

<br>

- 이렇게 사용하다보면 같은 헤더 파일이 여러 곳에서 선언되어 중복 정의될 수 있습니다.
- 이 문제를 해결하기 위해서 헤더 파일 첫줄에 `#pragma once`라고 적어두면 중복 정의를 방지할 수 있습니다.
    - `#pragma once` 방법은 컴파일러에 따라 지원이 안될 수도 있다고 하는데, 왠만한 컴파일러에서 잘 되었으며 주로 사용하는 **표준 C99**에서도 문제 없이 사용됩니다.
    - 다른 방법으로 `#ifndef`를 사용하는 방법이 있는데, `#pragma once` 보다 상대적으로 정의하기가 귀찮으므로 저는 사용하지 않고 있습니다. (물론 이 방법도 쉽습니다.)
- 그럼 위의 구조와 유사하게 아래에 한번 코드를 구성해 보았습니다.

<br>

<iframe height="800px" width="100%" src="https://repl.it/@gaussian37/headerstructure?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>

<br>

- main.c의 `test1()` 함수가 호출 되는 경로를 보면 main.h → test1.h → test1.c 가 됩니다.
- main.c의 `test2()` 함수가 호출 되는 경로를 보면 main.h → test2.h → test2.c 가 됩니다.
- main.c의 `MIN(1,2)` 매크로가 호출되는 경로를 보면 main.h → test1.h 또는 test2.h → common.h가 됩니다.
    - 이 때, `common.h`는 test1.h와 test2.h 두 헤더 모두에 선언되어 있지만 `#pragma once`가 첫줄에 선언되어 있기 때문에 먼저 최초로 선언되었을 때에만 정의되어 중복 정의 되지 않게 됩니다.