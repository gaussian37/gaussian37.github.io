---
layout: post
title: const char *와 char const *의 차이
date: 2019-09-03 00:00:00
img: cpp/cpp.png
categories: [cpp-etc] 
tags: [cpp, c++, const, char] # add tag
---

- 이번 글에서는 `const char *`와 `char const *`의 차이점에 대하여 알아보도록 하겠습니다.

<br>

### const char * : pointer to constant (상수를 가리키는 포인터)

<br>

- `const char *`형태는 영어로 **pointer to constant**라고 하고 한글로는 상수를 가리키는 포인터 라고 합니다.
    - 컴퓨터 용어에서 한글로 번역하면서 뜻이 애매해지는 것이 있어서 그냥 **pointer to constant**라고 하겠습니다.
- **pointer to constant**는 

```cpp
#include<stdio.h> 
#include<stdlib.h> 

int main()
{
	char a = 'A', b = 'B';
	const char* ptr = &a;

	//*ptr = b; 포인터가 가리키는 값이 변경되는 것은 에러가 발생합니다.
	 

	// 포인터는 변경될 수 있습니다.
	printf("value pointed to by ptr: %c\n", *ptr);
	ptr = &b;
	printf("value pointed to by ptr: %c\n", *ptr);
}

```

### char const * : constant pointer (상수 포인터)

<br>





