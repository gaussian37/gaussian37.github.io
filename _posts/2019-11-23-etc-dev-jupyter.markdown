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

- ### jupyter lab 단축키
- ### jupyter 기본 경로 변경
- ### jupyter lab font 변경
- ### (필수) ipywidgets 설치
- ### (extension) variable inspector
- ### (extension) spreadsheet
- ### (extension) debugger 

<br>

## **jupyter lab 단축키**

<br>

- jupyter lab 내부에서 Tab 전환하는 방법
    - `next 탭 전환` : ctrl + shift + [
    - `prev 탭 전환` : ctrl + shift + ]

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

## **jupyter lab font 변경**

<br>

- jupyter lab을 실행한 다음 상단의 `Settings` 탭 → `Advanced settings editors`로 들어갑니다.
- 그 다음 `Notebook`과 `Texteditor` 탭을 각각 클릭해 보면 왼쪽에는 `System defaults`가 있고 오른쪽에는 `User preferences`가 있습니다.
- `System defaults`의 내용을 복사해서 `User preferences` 에 붙여쓰고 내가 고치고 싶은 부분만 고치면 즉시 반영됩니다. (재실행 필요 없음)

<br>

## **(필수) ipywidgets 설치**

<br>

- `ipywidgets`은 jupyter lab에서 widget들이 구동되기 위한 기본적인 셋팅으로 jupyter lab을 온전하게 사용하려면 반드시 설치하셔야 합니다. 설치 방법은 다음 링크를 참조하시기 바랍니다.
    - 링크 : https://ipywidgets.readthedocs.io/en/stable/user_install.html
- 만약 저와 같이 `virtualenv` 환경에서 사용하신 다면 다음 순서를 통해 설치하시면 됩니다.
- 1) 가상 환경 실행
- 2) cmd 창에서 실행 : `pip install ipywidgets`
- 3) cmd 창에서 실행 : `jupyter nbextension enable --py widgetsnbextension --sys-prefix`
- 4) node.js 설치 : https://nodejs.org/en/
- 5) cmd 창에서 실행 : `jupyter labextension install @jupyter-widgets/jupyterlab-manager`

<br>

## **(extension) variable inspector**

<br>

- jupyter lab의 좋은 extension 중 하나인 variable inspector를 소개합니다.
- jupyter lab을 실행 후 왼쪽 탭의 extension manager를 클릭 후 `variable inspector`로 검색합니다.
- 그러면 `@lckr/jupyterlab_variableinspector` 라는 항목이 나옵니다. 이 extension 입니다.
    - 링크 : https://github.com/lckr/jupyterlab-variableInspector
- 이 extension은 `node.js`를 기반으로 돌아가게 되므로 다음 링크에서 먼저 node.js를 설치를 해주어야 합니다.
    - node.js 링크 : https://nodejs.org/en/
- node.js를 설치한 후에 extension manager 탭에서 `variable inspector`를 설치합니다.
- 설치한 후에 `ipynb`에서 오른쪽을 클릭하면 `open variable inspector`라는 항목이 생깁니다. 이것을 클릭하면 새로운 탭에 variable 들을 모아서 보여줍니다.

<br>
<center><img src="../assets/img/etc/dev/jupyter/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>

