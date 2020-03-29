---
layout: post
title: C 언어 문법 snippet
date: 2020-01-03 00:00:00
img: c/c.png
categories: [c-concept] 
tags: [c, c language, c 언어] # add tag
---

<br>

## **목차**

<br>

- ### char[]와 char*의 차이
- ### 문자 상수 리스트
- ### 조건 연산자 (? : )
- ### 쉬프트 연산자 (<<, >>)
- ### printf 출력 관련
- ### 변수 주소 구조

<br>

## **char[]와 char*의 차이**

<br>

- 문자열을 정의하는 방법에는 두가지 방법인 `char[]`와 `char*`가 있습니다.
- 두 방법의 차이는 `char str[] = "abc";`에서 str은 문자열 변수이고 `char* s = "abc";`는 문자열 상수입니다. 즉 문자열 변수는 변경이 가능하지만 문자열 상수는 변경이 불가능 합니다.

<br>

## **문자 상수 리스트**

<br>

- `\n` : printf() 함수 등에 의해 출력을 다음 줄로 이동하는 역할
- `\t` : 4개 또는 8개의 공백을 띄는 역할
- `\\` : 역슬래쉬를 문자 또는 문자열에서 사용
- `\0` : 널 문자임을 표시

<br>

## **조건 연산자 (? : )**

<br>

- `(조건) ? (조건이 참) : (조건이 거짓)` 형태로 사용되며 예를 들어 `1 < 3 ? 1 : 0` 이라는 예제가 있으면 1이 선택됩니다.
- 조건 연산자는 중첩해서 사용할 수 있으며 예를 들어 `max_value = (x > y) ? x : (y > 5) ? y : (x + y);`와 같이 사용할 수 있습니다.

<br>

## **쉬프트 연산자 (<<, >>)**

<br>

- 쉬프트 연산자인 `<<`를 이용하면 *2 와 `>>` /2 를 효과적이고 빠르게 계산할 수 있습니다. 왜냐하면 비트 자체를 이동시키는 것이기 때문입니다.
- `a << 3` 이란 연산의 결과를 쉽게 이해하려면 $$ a \times 2^{3} $$과 같이 이해하면 됩니다.
- `a >> 3` 의 결과는 $$ \text{int}(a \times 2^{-3}) $$으로 이해하면 됩니다.

<br>

## **printf 출력 관련**

<br>

- `%p` 를 이용하여 주소값을 출력할 수 있습니다.

<br>

```cpp
int a = 3;
printf("value : %d, address : %p\n", a, &a);
: value : 3, address : 0x7ffc30b6feac
```

<br>

## **변수 주소 구조**

<br>

- 포인터를 이용하여 변수의 시작 주소와 끝 주소를 입력하면 변수의 크기를 알 수 있습니다.
- 다음 코드는 int형 (4바이트), char형 (1바이트), 구조체(선언된 자료형 크기들의 총합)의 크기를 알 수 있습니다.
- 특히 포인터 변수에 증감 연산자를 이용하면 **데이터형의 크기만큼 증감**된다는 것을 확인할 수 있습니다. 일반 변수에 대한 증감 연산은 크기가 1씩 증감하는 것과 차이가 있습니다.

<br>

<iframe height="800px" width="100%" src="https://repl.it/@gaussian37/addresscheck?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>

<br>