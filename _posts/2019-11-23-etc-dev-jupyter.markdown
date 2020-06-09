---
layout: post
title: jupyter notebook, jupyter lab 사용 관련 snippets
date: 2019-11-23 00:00:00
img: etc/dev/jupyter/0.png
categories: [etc-dev] 
tags: [jupyter, notebook, lab] # add tag
---

<br>

- 이 글에서는 `jupyter notebook` 또는 `jupyter lab`을 사용하면서 필요한 것 들을 정리해 보겠습니다.

<br>

## **목차**

<br>

- ### jupyter 기본 경로 변경

<br>

## **jupyter 기본 경로 변경**

<br>

- jupyter를 실행하였을 때, working directory 기본 경로를 수정하는 방법을 다루어 보겠습니다. 아래 방법의 기준은 windows를 기준으로 작성하였습니다.
- 먼저 command를 실행합니다. 그 다음 jupyter를 실행할 수 있도록 환경을 준비합니다. (가상 환경 또는 conda)
- `jupyter notebook --generate-config`를 실행합니다. 그러면 다음 경로에 `.jupyter` 라는 폴더가 생성됩니다. 경로는 사용자 컴퓨터에 따라 다를 수 있으니 아래 경로는 참조만 하시고 실제 경로는 찾아가 보셔야 합니다.
    - `.jupyter` 폴더 생성 경로 : C:\Users\사용자이름
- `.jupyter` 폴더 안에 생성되어 있는 `jupyter_notebook_config.py` 를 편집기로 열어보겠습니다.
- 그 다음 `#c.NotebookApp.notebook_dir = ''` 가 적힌 행을 찾습니다. (179 번째 line 정도에 있습니다.)
- 이 행의 주석(`#`)을 제거하고 `''` 란 안에 원하는 폴더의 절대 경로 삽입합니다. 디렉토리 구분자는 `\`가 아닌 `/`를 사용합니다. 예를 들면 `c:/temp`와 같이 사용해야 합니다.
- 저장 후 jupyter를 다시 실행하면 지정한 경로가 working directory로 잡혀있습니다.

<br>