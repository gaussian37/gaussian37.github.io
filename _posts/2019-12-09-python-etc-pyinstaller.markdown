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
- 여기서 중요한 것은 현재 사용하고 있는 파이썬의 환경에 따라서 용량이 달라질 수 있다는 것입니다.
- 예를 들어 아나콘다 환경에서 실행파일을 만들면 꽤나 큰 용량의 실행파일이 생성됩니다. 왜냐하면 아나콘다에 설치된 패키지들을 모두 포함하기 때문입니다.
- 따라서 필요한 패키지만 설치된 가상 환경에서 실행하는 것을 추천드립니다.

<br>

- 개인적으로 사용하는 옵션들은 아래와 같습니다.
- `--onefile` : dist 폴더 아래에 한 개의 파일로 합쳐서 생성합니다. 이 파일 하나로 뭉쳐있기에 관리가 편하지만 용량이 꽤 큰 편입니다.
- `--windowed` : 실행 파일을 실행하였을 때, 콘솔 창이 실행되지 않습니다. **콘솔창에 print를 하거나 에러 로그 확인이 필요하면 이 옵션은 사용하지 않아야** 하고, 어떤 로그도 확인할 필요가 없거나 콘솔창 이외의 환경에 출력한다면 이 옵션을 사용하여 콘솔 창을 열지 않도록 합니다.
- `--icon=아이콘 경로` : 아이콘 경로를 지정하면 그 아이콘 이미지를 이용하여 실행 파일을 생성합니다. 
    - 예시 : `--icon=..\icon.co`