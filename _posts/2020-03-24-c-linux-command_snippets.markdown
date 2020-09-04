---
layout: post
title: Linux 명령어 snippets
date: 2020-02-01 00:00:00
img: c/linux/linux.jpg
categories: [c-linux] 
tags: [linux, 리눅스, 명령어, command] # add tag
---

<br>

이 글에서는 Linux에서 사용하였던 명령어들 중 유용하게 사용하였던 명령어들을 정리해보도록 하겠습니다.

<br>

## **목차**

<br>

- ### Could not get lock /var/lib/dpkg/lock - open 오류 수정
- ### 텍스트파일 내의 행의 갯수를 알고 싶은 경우
- ### 리눅스 특정 폴더 이하 파일 개수 세기
- ### 텍스트 파일 행 단위 정렬

<br>

## **Could not get lock /var/lib/dpkg/lock - open 오류 수정**

<br>

- 리눅스에서 `sudo apt-get install`을 이용하여 패키지를 설치하는 도중 오류가 발생하면 이후에 더이상 패키지 설치가 안되는 경우가 발생합니다.
- 이것은 패키지가 설치되는 부분에 `lock`이 발생한 것으로 그 부분에 해당하는 `lock`을 아래 명령어를 통하여 제거하면 됩니다.

<br>

```
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo rm /var/lib/dpkg/lock
```

<br>

## **텍스트파일 내의 행의 갯수를 알고 싶은 경우**

<br>

- `ls | wc -l` 명령어를 이용하면 new line의 갯수를 카운트 할 수 있습니다.

<br>

## **리눅스 특정 폴더 이하 파일 개수 세기**

<br>

- 어떤 특정 경로 이하에 존재하는 파일의 갯수를 셀 때, `find` 명령어와 `wc -l`을 이용하여 확인할 수 있습니다.

<br>

```python
find /폴더/경로 -type f | wc -l
```

<br>

- 만약 현재 경로 이하의 모든 파일의 갯수를 확인하고 싶으면 경로에 `.`을 입력하면 됩니다.

<br>

```python
find . -type f | wc -l
```

<br>

## **텍스트 파일 행 단위 정렬**

<br>

- `sort`는 텍스트로 된 파일의 `행`단위 정렬을 할 때 사용하는 명령어 입니다.
- 행단위 오름차순 정렬 : `sort 텍스트파일`
- 행단위 내림차순 정렬 : `sort -r 텍스트파일`

<br>

