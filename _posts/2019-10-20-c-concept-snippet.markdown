---
layout: post
title: C 언어에서 필요한 snippet
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

