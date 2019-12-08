---
layout: post
title: Pyinstaller 사용 방법
date: 2019-12-09 00:00:00
img: python/etc/pyinstaller/pyinstaller.jpg
categories: [python-etc] 
tags: [python, pyinstaller] # add tag
---

<br>

- 이번 글에서는 작성한 python 프로그램을 실행 파일로 변환하는 방법에 대하여 알아보도록 하겠습니다.
- 전체 내용은 공심 홈페이지에 자세히 나와있습니다 : https://www.pyinstaller.org/

<br>

- 먼저 설치 방법 및 사용방법은 상당히 간단합니다.

<br>

- 설치 방법 : `pip install pyinstaller`
- 실행 방법 : `pyinstaller main.py`

<br>

- 실행을 하면 `build`, `dist` 폴더가 생성됩니다. 실행할 때에는 `dist` 폴더의 python 파일이 실행됩니다.
- pyinstaller을 실행할 때 다양한 옵션을 줄 수 있습니다. https://pyinstaller.readthedocs.io/en/stable/usage.html 링크를 참조하시기 바랍니다.