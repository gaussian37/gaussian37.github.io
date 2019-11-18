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
