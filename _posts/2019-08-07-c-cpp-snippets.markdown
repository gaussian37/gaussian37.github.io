---
layout: post
title: C++ 관련 유용한 코드 snippets
date: 2019-08-07 00:00:00
img: cpp/cpp.png
categories: [c-cpp] 
tags: [cpp, c++, 객체 지향, oop, object oriented] # add tag
---

<br>

- 이번 글에서는 C++ 관련 코드 사용 시 유용하게 사용하였던 코드들을 모아서 정리해 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### nlohmann-json을 이용한 Json 파일을 읽기

<br>

## **nlohmann-json을 이용한 Json 파일을 읽기**

<br>

- 관련 깃 페이지 : https://github.com/nlohmann/json
- 참조 : https://snowdeer.github.io/c++/2022/01/11/cpp-nlohmann-json-example/
- 리눅스 설치 방법 : `sudo port install nlohmann-json`

<br>

- `C++`에서는 간단한 `json` 파일을 읽는 것도 파이썬과 다르게 다소 까다롭습니다. 이번 글에서는 C++을 사용할 때 `nlohmann-json`을 이용하여 `json` 파일을 읽어서 간단히 다루는 예제를 살펴보도록 하겠습니다.

<br>