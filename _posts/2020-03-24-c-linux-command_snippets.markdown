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
