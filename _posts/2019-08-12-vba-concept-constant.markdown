---
layout: post
title: 상수(Constant) 선언
date: 2019-08-13 00:00:00
img: vba/vba.png
categories: [vba-concept] 
tags: [vba, 상수, Constant] # add tag
---

- 이번 글에서는 상수를 선언하는 방법에 대하여 간략하게 언급하겠습니다.
- 상수는 변수와는 다르게 처음에 초기값을 지정하면 그 이후에 절대 바꿀 수 없는 값을 말합니다.
- 따라서 상수는 절대불변의 어떤 값을 저장하는 데 사용해야 합니다.
  - 예를 들면 원주율등과 같은 상수가 있습니다.
- 상수는 숫자나 문자열에 사용될 수 있고 상수를 사용하려면 `Const`문을 사용하여 상수를 선언하면 됩니다.

<br>

```vb
Const password As String = "abcd"
Const num As Long = 123123
Const password As String = "abcd", num As Long = 123123
```

<br>

- 상수를 만드는 방법은 변수를 만들 때 사용하였던 `Dim` 대신에 `Const`를 사용하면 됩니다.
- 위 코드 처럼 상수 또한 한 줄에 여러 개의 상수를 선언할 수 있습니다.
- 변수와 마찬가지로 상수도 선언 위치에 따라 사용 범위가 달라집니다. 상수를 프로시저 내에서 선언하면 해당 프로시저 내에서만 사용할 수 있고, 모듈의 시작 부분에서 선언하면 모듈의 모든 프로시저에서 사용할 수 있습니다.
- 모든 모듈의 모든 프로시저에서 사용하길 원한다면 `Public` 키워드를 사용하여 모듈의 시작 부분에서 다음과 같이 선언합니다.

<br>

```vb
Public Const program_name As String = "Analysis"
```

<br>
