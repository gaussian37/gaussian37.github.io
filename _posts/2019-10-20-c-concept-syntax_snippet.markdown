---
layout: post
title: C 언어 문법 snippet
date: 2019-10-20 00:00:00
img: c/c.png
categories: [c-concept] 
tags: [c, c language, c 언어] # add tag
---

<br>

## **목차**

<br>

- ### C언어 컴파일, 링크, 빌드의 의미
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
- ### int main(int argc, char* argv[])
- ### 중복 선언 방지 팁

<br>

## **C언어 컴파일, 링크, 빌드의 의미**

<br>

- 출처 : https://opentutorials.org/module/1594/9734
- 초창기의 컴퓨터는 기계어로 프로그래밍을 했습니다. 그러나 기계어는 사람이 이해하기 아주 어려워서, 이를 보다 편하게 사용하기 위해 다음과 같은 방법을 생각하였습니다.
- 기계어의 집합을 더 간단하게 표현하는 텍스트 문서를 만드는 방법입니다. 예를 들어 C는 긴 코드를 간단하게 표현하기 위해 함수나 매크로(macro)를 사용할 수 있는데, 기계어로 10줄짜리의 코드를 매크로 A로 정의하고 문서에는 A만 써넣는 경우를 생각해보겠습니다.

<br>

- 먼저 이 텍스트 문서를 기계어로 자동 번역하는 프로그램 A를 만듭니다. 
- 텍스트 문서를 프로그램 A를 이용하여 기계어로 자동 번역한다. 이 프로그램을 실행하면 위에서 예시로 작성한 문서의 A가 기계어 10줄로 번역됩니다.
- 이렇게 하면 필요할 때 마다 텍스트 문서만 수정하여 프로그램을 간단하게 만들 수 있습니다. 위 내용을 순서에 맞게 다시 설명해 보겠습니다.

<br>

- ① 일정한 형식으로 작성된 문서를 기계어로 자동 번역하는 프로그램 A를 먼저 만듭니다.
- ② 프로그램을 만들 때마다 A가 번역할 수 있도록 일정한 형식으로 문서를 작성합니다.
- ③ 문서 작성이 완료되면 프로그램 A를 실행하고 파일을 넘겨서, A가 자동으로 번역해준 기계어 파일을 얻습니다.

<br>

- 바로 여기서 사용되는 프로그램 A를 `컴파일러(compiler)`라고 하고, 이 때 작성한 일정한 형식의 컴퓨터 명령을 `소스 코드(source code)`, 소스 코드가 저장된 텍스트 파일을 `소스 코드 파일(source code file)` 또는 간단히 `소스 파일(source file)`, 그리고 이를 번역하는 행위를 `컴파일(compile)`이라고 합니다.

<br>

- 컴퓨터가 발전하고 작성하는 소스 코드의 양이 늘어남에 따라, 한 파일에 모든 소스 코드를 작성하는 방식이 불편하다는 것을 깨닫게 되었습니다. 따라서 사람들은 소스 코드를 다른 파일에 분리하는 방법을 생각해 냅니다. - 원래 하나였던 파일을 분리했으므로, **프로그램을 완성하려면 분리했던 파일은 모두 연결**해야 합니다. 
- 이렇듯 분리된 파일을 모아 하나의 실행 가능한 파일을 만들면 이를 두고 파일들을 `링크(link)`했다고 하고, 이때 사용되는 프로그램을 `링커(linker)`라고 한다.

<br>

- 종합하면, 우리는 기계어를 이용하지 않고 실행 파일을 생성하기 위해 다음의 순서를 거치게 됩니다.
- ① 소스 코드를 작성하고 파일로 저장합니다.
- ② 저장한 소스 파일을 컴파일러를 이용하여 `컴파일` 합니다. 이 때, `오브젝트 파일`이 생성됩니다.
- ③ 컴파일러가 생성한 오브젝트 파일들을 링커를 이용하여 `링크` 합니다. 이 떄, 실행 가능한 오브젝트 파일이 생성됩니다.
- ④ 링커는 `실행 가능한 오브젝트 파일`을 생성합니다. 컴파일러가 생성하는 파일과 링커가 생성하는 파일의 차이는 생성한 목적 파일이 실행 가능 하느냐에 있습니다.
- ⑤ `컴파일`과 `링크` 과정을 합쳐 `빌드(build)`라고 하고, 이때 사용되는 프로그램을 `빌더(builder)`라고 합니다.

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


- `printf`의 `conversion specifier(변환(형식) 지정자)`는 `%[flags][width][.precision][length]specifier` 형태를 갖습니다.
- 예를 들면 아래와 같습니다.

<br>

```c
printf("%10.5hi", 256);
```

<br>

- 먼저 위 형식 중 `flag` 부터 알아보도록 하겠습니다.
- `-` : -를 flag로 사용하면 출력할 때, width를 고려하여 왼쪽으로 붙여서 출력합니다. 물론 기본 값은 width를 고려하여 오른쪽으로 붙여서 출력하는 방법입니다.
- `+` : +를 flag로 사용하면 출력할 때, 양의 값에도 +를 붙여서 출력합니다. 물론 기본값은 양의 값에는 부호를 붙이지 않습니다.
- ` ` (space) : flag로 빈칸을 사용하면 
- `#` : 
- `0` : width를 지정하였을 때, 남는 자리를 0으로 채웁니다.

<br>

- 그 다음 `width`를 알아보도록 하겠습니다.
- `숫자` : width로 숫자를 넣으면 출력할 최소 문자 갯수가 됩니다. 출력할 문자의 갯수가 width 보다 작으면 기본적으로 빈칸으로 출력되고 출력할 문자의 갯수가 width 보다 많으면 width와 상관 없이 출력할 모든 문자열을 출력합니다.
- `*` : 

<br>

- 그 다음 `.precision`에 대하여 알아보겠습니다.
- `.number` : 
- `*` : 

<br>

- `conversion specifier`의 형식들을 하나씩 살펴보도록 하겠습니다.
- `%a` : 부동 소수점 수, 16진수, p 표기법
- `%A` : 부동 소수점 수, 16진수, P 표기법 
- `%c` : 한 글자
- `%d` 또는 `%i` : 부호가 있는 10진(decimal) 정수(integer)
- `%e` : 부동 소수점 수, e 표기법
- `%E` : 부동 소수점 수, E 표기법
- `%f` : 부동 소수점 수, 10진수 표기. double의 경우 `%lf` 사용
- `%g` : 값에 따라서 `%e` 또는 `%f`를 자동으로 적용. 소수점 아래 자리수가 4자리 보다 많을 때 또는 값이 6자리 수 보다 클 때, `%e` 적용됨
- `%G` : 값에 따라서 `%E` 또는 `%f`를 자동으로 적용. 소수점 아래 자리수가 4자리 보다 많을 때 또는 값이 6자리 수 보다 클 때, `%E` 적용됨
- `%o` : 부호가 없는 8진(octal) 정수
- `%p` : 포인터 주소값
- `%s` : 문자열
- `%u` : 부호가 없는 10진 정수
- `%x` : 부호가 없는 16진 정수, 소문자 알파벳 사용
- `%X` : 부호가 없는 16진 정수, 대문자 알파벳 사용
- `%%` : 퍼센트 기호 출력

<br>

- `printf("%9d\n", 12345);` : `    12345` 출력됨. 9자리 포맷을 맞추되 부족한 자릿수는 빈칸으로 둡니다.
- `printf("%09d\n", 12345);` : `000012345` 출력됨. 9자리 포맷을 맞추되 부족한 자릿수는 0으로 채웁니다.
- `printf("%.2f\n", 3,141592);` : `3.14` 출력됨. 소숫점 2자리 까지 출력하고 그 아래는 반올림합니다.

<br>

- printf 함수의 **return 값**은 출력한 문자의 갯수입니다. 예를 들어 `n = printf("hello")`이면 n에 5가 저장됩니다. 5는 출력된 문자열인 hello의 길이 입니다.




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

<br>

## **int main(int argc, char* argv[])**

<br>

- 컴파일을 통하여 실행 파일을 만들었을 때, 그 실행 파일에 `parameter`를 전달하려면 `main` 함수를 다음과 같이 선언해야 합니다.

<br>

```c
int main(int argc, char* argv[]){

	if(argc > 2){
		// argv[1] 부터 이용한다.
	}

}
```

<br>

- `argc`는 파라미터의 갯수를 저장하고 `argv`는 각 파라미터를 문자열 형태로 저장하게 됩니다.
- 컴파일 후 생성된 실행 파일의 이름이 `main` 이라고하면 실행할 때, `./main` 와 같이 실행할 수 있다고 가정하겠습니다.
- 이 때, 기본적으로 `argc`는 1이고 `argv[0]`은 `./main`이 저장됩니다. 즉, 실행한 명령어가 저장됩니다. 만약 실행할 때, 절대 경로를 이용하여 실행하였다면 현재 경로도 이용할 수 있으니, 응용을 잘 하면 유용하게 사용할 수 있습니다.
- 만약 `./main aaa`와 같이 실행하면 뒤에 `aaa`라는 파라미터가 `argv[1]`로 전달됩니다. 즉 문자열로 전달됩니다. `./main aaa bbb`와 같이 실행하면 `argv[2]`에는 `bbb`가 전달됩니다. 즉, **공백 문자를 통하여 파라미터를 구분**합니다.

<br>

## **중복 선언 방지 팁**

<br>

- 한 프로젝트에서 많은 코드를 다루다 보면 같은 헤더를 중복 참조하여 그 헤더의 변수나 함수들을 중복으로 선언하는 에러가 발생하곤 합니다.
- 이 때, 다음 팁들을 차례대로 이용하여 문제를 해결해 나아갈 수 있습니다.

- ① `#ifndef` 를 사용하는 방법 입니다. 예를 들어 다음과 같습니다.

<br>

```c
// sample.h

#ifndef __SAMPLE_H__
#define __SAMPLE_H__

// 코드 //

#endif
```

<br>

- 위 코드를 설명하면 `#ifndef __SAMPLE_H__` 즉, __SAMPLE_H__ 이란 매크로가 선언되지 않았으면 아래 코드를 실행하는 것입니다.
- 이 때 `#ifndef` 다음에 바로 __SAMPLE_H__ 을 선언하는 영역이 있습니다. 따라서 매크로를 이 시점에 선언하기 때문에 다음에 이 헤더 파일을 접근하면 `#ifndef` 라인을 통과할 수 없어서 중복 선언하지 않게 됩니다.

<br>

- ② 어떤 헤더 파일에 어떤 목적으로 변수를 선언하여 여러 파일에서 접근하여 사용하려고 할 때가 있습니다. 이 때에도 중복 선언 문제가 발생할 수 있습니다.
- 이 문제를 쉽게 해결 할 수 있는 방법은 `static` 키워드를 사용하여 헤더 파일에 있는 변수를 선언하는 것입니다.
- 이 방법이 효과가 있는 이유는 `static` 키워드를 이용하여 선언된 변수는 전체 프로그램에서 딱 한번 메모리에 영역이 할당되기 때문입니다.
