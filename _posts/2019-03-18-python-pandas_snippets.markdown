---
layout: post
title: Pandas 기본 문법 및 코드 snippets
date: 2019-03-18 00:00:00
img: python/basic/pandas.png
categories: [python-basic] 
tags: [pandas, python, python 기본] # add tag
---

- 이 글에서는 `Pandas`를 사용하면서 필요하다고 느끼는 `Pandas 기본 문법 및 코드`들을 정리해 보겠습니다.

<br>

## **목차**

<br>

- ### read_csv(excel) 함수를 통하여 파일 읽을 때 팁


<br>

## **read_csv(excel) 함수를 통하여 파일 읽을 때 팁**

<br>

- ① 파일을 읽을 때, 첫 열의 인덱스(0, 1, 2, ...)를 만들고 싶지 않다면 다음 옵션을 준다.
    - `index_col = False`
- ② 파일을 읽을 때, 첫 행에 헤더를 만들고 싶지 않다면 다음 옵션을 줍니다.
    - `header = None`
- ③ 파일을 읽을 때, 구분자의 기준을 주고 싶다면 다음 옵션을 줍니다.
    - `sep = ','`, 특정 문자를 입력하면 됩니다.

