---
layout: post
title: const char *와 char const *의 차이
date: 2019-09-03 00:00:00
img: cpp/cpp.png
categories: [c-etc] 
tags: [c, const, char, constant pointer, pointer to constant```] # add tag
---

- 이번 글에서는 `const char *`와 `char const *`의 차이점에 대하여 알아보도록 하겠습니다.

<br>

### const char * : pointer to constant (상수를 가리키는 포인터)

<br>

- `const char *`형태는 영어로 **pointer to constant**라고 하고 한글로는 상수를 가리키는 포인터 라고 합니다.
    - 컴퓨터 용어에서 한글로 번역하면서 뜻이 애매해지는 것이 있어서 그냥 **pointer to constant**라고 하겠습니다.
- **pointer to constant**는 포인터가 가리키는 값이 변경될 수는 없고 포인터가 변경될 수는 있습니다.

<br>
<img src="../assets/img/c/concept/const_char_pointer/1.png" alt="Drawing" style="width: 600px;"/>
<br>

- 즉 위 그림과 같이 포인터가 가리키는 주소 자체를 주소1 에서 주소2로 변경하는 것은 가능하지만 만약 주소1의 값을 값1'로 바꾸려고 하면 에러가 납니다.
- 따라서 **pointer to constant**에서는 포인터가 가리키는 값이 상수가 됩니다.
- 아래 코드를 보면 포인터는 변경될 수 있지만 포인터가 가리키는 값이 변경되면 에러가 발생합니다. 

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

<br>

### char const * : constant pointer (상수 포인터)

<br>

- 반면에 `char const *`는 `const`의 위치가 변경되어 상수화 되는 대상이 변경됩니다. 먼저 이런 방법은 **constant pointer**라고 합니다.
- **constant pointer**는 포인터 주소값을 상수화 시킵니다. 즉, 주소값이 변경이 안됩니다. 
- 반면에 주소가 가리키는 값에 대한 제한은 없기 때문에 값은 변경할 수 있습니다. 

<br>
<img src="../assets/img/c/concept/const_char_pointer/2.png" alt="Drawing" style="width: 600px;"/>
<br>  

- 예를 들면 위 코드와 같이 주소값은 변경할 수 없지만 주소값이 가리키는 값은 변경이 가능합니다.
- 이 내용을 코드로 알아보면 다음과 같습니다.

<br>

```cpp
// char* const p 
#include<stdio.h> 
#include<stdlib.h> 

int main()
{
	char a = 'A', b = 'B';
	char* const ptr = &a;
	printf("Value pointed to by ptr: %c\n", *ptr);
	printf("Address ptr is pointing to: %d\n\n", ptr);

	//ptr = &b; 주소값을 변경하는 것은 오류가 발생합니다.

	// 포인터가 가리키는 주소의 값을 변경하는 것은 가능합니다.
	*ptr = b;
	printf("Value pointed to by ptr: %c\n", *ptr);
	printf("Address ptr is pointing to: %d\n", ptr);
}
```

<br>

- 정리하면 `const char*` 즉, **pointer to constant**는 포인터가 가리키는 대상(값)이 상수화 됩니다. 따라서 포인터 변수가 가지는 주소값은 변경 가능하지만 주소가 가리키는 값은 변경 불가합니다.
- 반면 `char const *`즉, **constant pointer**는 포인터 주소값이 상수화가 됩니다. 따라서 포인터 변수가 가지고 있는 주소값은 변경 불가능 합니다. 하지만 주소값이 가리키는 값에 대한 제약은 없으므로 값 변경은 가능합니다.

<br>

### const char const * : constant pointer to constant (상수를 가리키는 상수 포인터)

<br>

- 앞에서 설명한 **pointer to constant** 조건과 **constant pointer**의 조건을 결합한 형태입니다.
- 따라서 포인터 변수의 주소값을 변경하는 것도 불가능하고 주소값이 가리키는 값을 변경하는 것도 불가능합니다.

<br>

```cpp
// C program to illustrate 
//const char * const ptr 
#include<stdio.h> 
#include<stdlib.h> 

int main() 
{ 
	char a ='A', b ='B'; 
	const char *const ptr = &a; 
	
	printf( "Value pointed to by ptr: %c\n", *ptr); 
	printf( "Address ptr is pointing to: %d\n\n", ptr); 

	// ptr = &b; illegal statement (assignment of read-only variable ptr) 
	// *ptr = b; illegal statement (assignment of read-only location *ptr) 

} 

```

<br>

- 위 코드를 보면 `const char *cost ptr`로 생성된 포인터 변수는 주소값을 바꾸는 것과 주소값이 가리키는 값을 변경하는 것 모두가 금지됩니다.