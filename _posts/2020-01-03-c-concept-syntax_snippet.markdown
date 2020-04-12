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

- ### declaration(선언)과 definition(정의) 차이
- ### char[]와 char*의 차이
- ### const char *와 char const *의 차이
- ### 문자 상수 리스트
- ### 조건 연산자 (? : )
- ### 쉬프트 연산자 (<<, >>)
- ### printf 출력 관련
- ### scanf 입력 관련
- ### 변수 주소 구조
- ### sizeof 관련

<br>

## **declaration(선언)과 definition(정의) 차이**

<br>

- C/C++ 언어 관련 글을 읽다가 보면 `declaration`과 `definition`이라는 용어가 나오는데 그 차이점에 대하여 알아보겠습니다.
- 먼저 `declaration(선언)`에 대하여 알아보도록 하겠습니다.
- 변수 선언은 컴파일러에 변수명, 변수 타입 그리고 초깃값이 정의되어 있다면 초깃값 까지 전달하는 역할을 합니다. 즉, `변수 선언`은 **컴파일러에 어떤 변수에 대한 상세한 정보를 주는 역할**을 합니다.
- 그러면 `definition(정의)`는 무슨 역할을 할까요?
- 정의는 선언한 변수가 어디에 저장되는지를 나타냅니다. 즉, **변수가 메모리 영역에 할당되는 순간**을 `정의`라고 합니다.

<br>

- 보통 C언어에서는 선언과 정의가 동시에 발생합니다. 그래서 전혀 차이가 없어 보입니다. 
- 예를 들면 선언과 정의가 동시에 발생하는 것이 다음과 같습니다.

<br>

```c
int a; //선언과 정의가 동시에 적용
```

<br>

- C언어의 변수에서는 선언과 정의가 보통 동시에 발생하기 때문에 차이점을 잘 모를 수 있으나 함수에서는 차이점이 확연히 나타납니다.
- 함수의 선언 또한 컴파일러에게 함수의 상세한 정보를 알려줍니다. 즉,함수명, 리턴타입, 매개변수 등등을 컴파일러에 전달해줍니다.

<br>

```c
int add(int, int) // 함수의 선언
```

<br>

- 위 함수의 선언은 함수명, 리턴 타입 그리고 2개의 매개변수가 있다는 것을 컴파일러에 알려줍니다.
- 하지만 이 단계에서는 아직 함수가 메모리에 할당되지는 않습니다.

<br>

```c
// 함수의 정의
int add(int a, int b)
{
    return (a+b);
}
```

<br>

- 정의 단계에서는 위 함수가 **메모리 영역에 할당**됩니다. 

<br>

- 정리해 보겠습니다.
- `Declaration(선언)`: 변수나 함수는 여러번 선언될 수 있고 이 단계에서는 메모리에 할당되지 않습니다. 이 단계에서는 컴파일러에 변수나 함수의 정보만 알려줍니다.
- `Definition(정의)`: 변수나 함수는 딱 한번 정의되고 이 단계에서 메모리에 할당됩니다. 


<br>

## **char[]와 char*의 차이**

<br>

- 문자열을 정의하는 방법에는 두가지 방법인 `char[]`와 `char*`가 있습니다.
- 두 방법의 차이는 `char str[] = "abc";`에서 str은 문자열 변수이고 `char* s = "abc";`는 문자열 상수입니다. 즉 문자열 변수는 변경이 가능하지만 문자열 상수는 변경이 불가능 합니다.

<br>

## **const char *와 char const *의 차이**

<br>

- `const char *`와 `char const *`의 차이점에 대하여 알아보도록 하겠습니다.

<br>

### const char * : pointer to constant

<br>

- **pointer to constant**는 포인터가 가리키는 값이 변경될 수는 없고 포인터가 변경될 수는 있습니다.

<br>
<img src="../assets/img/c/concept/const_char_pointer/1.png" alt="Drawing" style="width: 600px;"/>
<br>

- 즉 위 그림과 같이 포인터가 가리키는 주소 자체를 주소1 에서 주소2로 변경하는 것은 가능하지만 만약 주소1의 값을 값1'로 바꾸려고 하면 에러가 납니다.
- 따라서 **pointer to constant**에서는 **포인터가 가리키는 값**이 `상수`가 됩니다.
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

### char const * : constant pointer

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

- 정리하면 `const char*` 즉, **pointer to constant**는 **포인터가 가리키는 대상(값)이 상수화** 됩니다. 따라서 포인터 변수가 가지는 주소값은 변경 가능하지만 주소가 가리키는 값은 변경 불가합니다.
- 반면 `char const *`즉, **constant pointer**는 **포인터 주소값이 상수화**가 됩니다. 따라서 포인터 변수가 가지고 있는 주소값은 변경 불가능 합니다. 하지만 주소값이 가리키는 값에 대한 제약은 없으므로 값 변경은 가능합니다.

<br>

### const char const * : constant pointer to constant

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

## **scanf 입력 관련**

<br>

- scanf의 서식에 빈칸을 두면 whitespace 문자들을 입력받습니다. 예를 들어 `scanf("%d %d", &a, &b);`라고 입력 받으면 변수 a와 b를 구분해서 입력 받을 때, 스페이스(` `), 탭(`\t`), 엔터(`\n`)를 통해 구분할 수 있습니다.
- whitespace 끼리는 동일하게 적용되므로 `scanf("%d\n%d", &a, &b);` 또는 `scanf("%d\t%d", &a, &b);`라고 정의하여도 스페이스(` `), 탭(`\t`), 엔터(`\n`)를 통해 구분할 수 있습니다. 물론 스페이스를 통해 구분하는 것이 입력이 편하고 보기에도 편하기 때문에 스페이스로 구분하는 것이 일반적입니다.

<br>

- scanf의 서식에 whitespace가 아닌 특정 문자가 들어오면 입력 시 반드시 그 문자를 입력해주어야 합니다. 왜냐하면 그 문자를 구분자로 삼아서 입력받기 때문입니다.
- 예를 들어 `scanf("%d,%d", &a, &b);`로 입력하면 반드시 `,`를 입력해 주어야 두 정수를 구분해서 입력받을 수 있습니다.
- 같은 원리로 `scanf("%dA%d", &a, &b);`라면 반드시 `A`를 입력해 주어야 두 정수를 구분해서 입력받을 수 있습니다.
- 여기서 조심할 것은 `%d,%d`와 같은 경우 `1 ,2`와 같이 입력하면 정상적으로 정수 2가 입력되지 않는데 그 이유는 `%d,%d`에는 공백문자가 없지만 입력할 때에는 `,` 앞에 공백 문자가 있기 때문입니다. 이런 점을 고려하여 문자를 입력해 주어야 문제가 없습니다.
- 만약에 어떤 구분 문자도 없이 `scanf("%d%d", &a, &b);`로 입력된다면 whitespace 문자를 기준으로 구분하게 됩니다.

<br>

- scanf에서 입력 받은 문자를 무시할 때에는 `*`를 이용합니다.
- 예를 들어 `scanf("%*d %d", &a);`를 실행한 후 `1 2`를 입력하면 첫번째 입력 받은 1은 무시되고 변수 a에 2가 입력됩니다.
- 이 방법 또한 `scanf("%*d,%d", &a);`와 같이 구분자로 특정 문자들과 섞어서 사용할 수 있습니다.

<br>

- `문자열`을 입력 받을 때, 특정 문자를 무시하면서 받고싶을 수 있습니다. 예를 들어 `Hello world`를 한번에 입력받아야 하는데 중간에 공백 문자가 있으면 구분되어 받기 때문입니다.
- 이 때, 사용할 수 있는 방법이 `[^문자]` 입니다. 이 방법은 `^`뒤에 오는 문자들이 입력될 때 까지 계속 받는다는 뜻입니다.
- 예를 들어 `scanf("%[^\n]s", str);` 라고 입력 하면 `\n`이 입력 될 때 까지 계속 입력 받고 `\n`이 입력되면 구분자로 삼아서 입력을 마칩니다. 이 방법으로 `Hello world`를 입력하고 마지막에 엔터를 누르면 문자열 전체가 str에 입력 됩니다. 이 방법은 문자열을 입력 받을 때에만 유효합니다.
- `[^문자]`에서 문자에는 여러 문자들이 들어갈 수 있습니다. 예를 들어 `scanf("%[^,-]s", str);` 라고 입력하면 `,`와 `-`가 들어오면 구분자로 입력하게 됩니다. 즉, 이 때에는 기존에 사용하던 whitespace 문자들은 구분자가 되지 않습니다.

<br>

## **변수 주소 구조**

<br>

- 포인터를 이용하여 변수의 시작 주소와 끝 주소를 입력하면 변수의 크기를 알 수 있습니다.
- 다음 코드는 int형 (4바이트), char형 (1바이트), 구조체(선언된 자료형 크기들의 총합)의 크기를 알 수 있습니다.
- 특히 포인터 변수에 증감 연산자를 이용하면 **데이터형의 크기만큼 증감**된다는 것을 확인할 수 있습니다. 일반 변수에 대한 증감 연산은 크기가 1씩 증감하는 것과 차이가 있습니다.

<br>

<iframe height="800px" width="100%" src="https://repl.it/@gaussian37/addresscheck?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>

<br>

## **sizeof 관련**

<br>

- `sizeof`를 이용할 때, 정적 배열과 동적 배열에서의 사용 시 고려해야 할 점에 대하여 확인해 보겠습니다.
- 정적 배열의 주소에 sizeof를 이용하면 배열 전체의 크기가 반환되는 반면, 동적 배열의 주소에 sizeof를 이용하면 포인터의 크기만 반환됩니다.
- 정적 배열 주소의 크기를 구하는 경우 비록 주소의 크기이지만 **배열 전체의 크기가 반환**되는 이유는 정적 배열의 경우 compile 단계에서 배열의 크기를 알 수 있기 때문에 배열의 대표 주소의 크기를 구하면 배열의 크기가 반환 되도록 설정되었습니다. 반면 동적 배열의 경우 runtime 단계에서 배열의 크기를 알 수 있기 때문에 배열의 대표 주소의 크기는 단순히 포인터 변수의 크기만 반환하도록 되어있습니다.
- 또한 함수의 파라미터로 정적 배열을 입력 받은 경우에는 주소의 크기가 포인터 변수의 크기만 반환하도록 되어 있습니다. 파라미터로 넘겨 받을 때에는 `call by pointer` 방식으로 넘겨 받기 때문에 포인터의 크기 만큼만 함수로 전달됩니다.

<br>

<iframe height="800px" width="100%" src="https://repl.it/@gaussian37/sizeofcheck?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>

<br>

- 위에서 설명한 이유를 이해하여 문자열 배열의 크기와 문자열의 길이의 차이점에 대하여 명확하게 이해할 수 있습니다.

<br>

<iframe height="800px" width="100%" src="https://repl.it/@gaussian37/sizeofandstrlen?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>

<br>

- 위 코드의 `str1`은 배열의 크기는 100이지만 문자열의 길이는 5가 됩니다. 문자열의 길이에서 NULL 문자는 제외됩니다.
- `str2`의 경우 배열의 크기는 NULL 문자를 포함하여 5이지만 문자열의 길이는 5가 됩니다.
- `str3`, `str4`도 동일한 원리로 이해할 수 있습니다.
- 마지막으로 `str5`는 동적 할당을 통하여 문자열의 공간을 생성한 것입니다. 이 때에는 sizeof의 결과가 배열의 크기가 아니라 문자열 포인터의 크기가 반환됩니다. 반면 strlen을 이용한 문자열 길이는 앞선 예와 똑같이 5가 출력됩니다.