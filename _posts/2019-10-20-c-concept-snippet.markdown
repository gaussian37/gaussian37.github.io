---
layout: post
title: C 언어 코드 snippet
date: 2019-10-20 00:00:00
img: c/c.png
categories: [c-concept] 
tags: [C] # add tag
---

<br>

## **목차**

<br>

- ### 변수명 출력
- ### 숫자 → 문자열
- ### 문자열 → 숫자
- ### 텍스트 입/출력 

<br>

## **변수명 출력**

<br>

```cpp
#include <stdio.h>

 #define GET_VARIABLE_NAME(variable, holder) sprintf(holder, "%s", #variable)

int main() {
    char variable_name[100]; // 출력할 변수명을 저장할 문자열
    GET_VARIABLE_NAME(print_this_variable, variable_name);
    puts(variable_name); // print_this_variable이 출력됩니다.
    return 0;
}
```

<br>

## **숫자 → 문자열**

<br>

- 숫자를 문자로 변경할 때에는 `sprintf` 함수를 사용합니다.

<br>

```c
#include <stdio.h>
#include <stdlib.h>

int main(){

    int a=123;
    int b=0;
    char buf[10];
    
    sprintf(buf, "%d", a);
    printf("%s\n",buf); 
```

<br>

## **문자열 → 숫자**

<br>

- 문자열을 숫자로 바꿀 때에는 아래 함수를 사용하여 바꿀 수 있습니다.

```c

int StringToInteger(char *str) {
    int ret = 0;
    while (*str) {
        ret = ret * 10 + (int)(*str - '0');
        str++;
    }
    return ret;
}

```

<br>

## **텍스트 입/출력**

<br>

- FILE 포인터와 fprintf를 이용하여 텍스트 파일에 출력하는 방법입니다.
- 아래 코드와 같이 사용 시 fprintf 함수의 파일 포인터에 해당하는 파일에는 fprintf를 통해 출력한 값만 적혀집니다.
- 물론 printf를 통하여 출력한 출력은 콘솔창에 기존 그대로 찍히게 됩니다.

<br>

```c
#include <stdio.h>

int main(){
    // "a" 적용 시 파일 끝에서 부터 계속 이어서 쓴다.
    // "w" 적용 시 파을을 덮어쓰기 한다.
    FILE* fp = fopen("output.txt", "w"); 
    fprintf(fp, "%d %lf\n", 1, 2.0);
    
    return 0;
}
```

<br>

- 이번에는 반대로 FILE 포인터와 fscanf를 이용하여 텍스트 파일을 읽어서 입력하는 방법입니다.
- 기본적인 사용법은 scanf와 유사하며 파라미터로 File Pointer를 넘겨주는 것에 차이가 있습니다.

<br>

```c
#include <stdio.h>

int main(){
    // "a" 적용 시 파일 끝에서 부터 계속 이어서 쓴다.
    // "w" 적용 시 파을을 덮어쓰기 한다.
    int a;
    double b;
    FILE* fp = fopen("input.txt", "r"); 
    fscanf(fp, "%d %lf\n", &a, &b);
    
    return 0;
}
```

<br>

- `fscanf` 함수가 더 이상 읽을 것이 없으면 -1을 반환합니다.
- 아래 코드와 같이 `fscanf`가 -1을 반환할 때 까지 파일을 읽으면 됩니다.

<br>

```c
#include <stdio.h>
#include <stdlib.h>

int main() {

	FILE* fp = fopen("input.txt", "r");
	int a;
	int ret = 1;
	while (1) {
		ret = fscanf(fp, "%d", &a);		
		if (ret < 0) {
			break;
		}
		printf("%d\n", a);
	}
}
```
