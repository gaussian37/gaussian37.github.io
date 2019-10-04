---
layout: post
title: C++에서 문제 풀 때 좋은 문법 및 템플릿
date: 2019-06-02 00:00:00
img: interview/ps/ps.png
categories: [interview-ps] 
tags: [ps, c++] # add tag
---

+ C++의 cin, cout을 scanf, print와 같이 빠르게 사용하고 싶다면 다음 코드를 입력해 줍니다.
    + `ios_base::sync_with_stdio(false);`
    + `cin.tie(NULL);`
	+ `cout.tie(NULL);`
    + 대신에 cin, cout을 쓰면서 절대로 scanf와 printf를 같이 써서는 안됩니다. (동기화가 끊겨서 값이 뒤죽박죽 됩니다.)
    
<br>

+ 입력을 EOF까지 입력을 받는 경우 다음과 같이 받으면 됩니다.
    + C 스타일 : `while(scanf("%d %d", &a, &b) == 2)`
    + C++ 스타일 : `while(cin>>a>>b))`

<br>

+ 한 줄을 입력 받고 싶으면 다음과 같이 입력 받습니다.
    + C 스타일 : `scanf("%[^\n]\n", s);`
        + scanf 안의 `%[^\n]`의 뜻을 살펴보면 **^**뒤의 문자만 빼고 입력을 받겠다는 뜻입니다. 
        + 즉, 개행 문자는 빼고 문자를 받는 다는 뜻이고 대괄호 밖의 마지막 문자 개행문자의 뜻은 입력 받는 마지막의 문자는 개행 문자여야 한다는 뜻입니다.
        + 따라서 입력 받는 문자열에서 마지막 문자는 개행문자이어야 하고 그 이전에 받는 개행 문자는 무시한다는 뜻입니다.
    + C++ 스타일 : `getline(cin, s);`
    
