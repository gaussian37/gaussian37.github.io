---
layout: post
title: Python 초기 세팅 및 설치
date: 2019-12-09 00:00:00
img: python/basic/python.jpg
categories: [python-basic] 
tags: [python, 세팅, 설치] # add tag
---

<br>

- 이번 글은 제가 사용하는 파이썬 환경을 정리하기 위하여 적었습니다.
- 아나콘다를 사용하시는 분은 사실 아나콘다를 사용하는 것이 훨씬 편합니다. 다만 필요한 것만 설치해야 하는 상황이거나 필요한 것만 설치하는 것을 원하는 분들은 이 글을 참조하시면 되겠습니다.
- 아래 환경 세팅은 윈도우 및 파이썬3를 기준으로 작성하였습니다.
- 그리고 파이썬3를 설치 할 때, `path`에 파이썬을 등록한 경우이여야 합니다.

<br>

## **목차**

- ### 가상 환경 세팅
- ### reqirement 파일 생성
- ### 내가 사용하는 라이브러리들

<br>

## **가상 환경 세팅**

<br>

- 가상 환경을 세팅하는 이유는 작업하는 사용자 및 작업 환경에 따라서 독립적인 환경을 만들기 위해서 입니다.
- 공용 컴퓨터를 사용하는 경우라면 나만의 공간을 위해서 독립 환경을 만드는 것이 중요하고 개인 컴퓨터를 사용하더라도 각 라이브러리의 버전에 관한 dependency가 있다면 독립을 해주는 것이 좋습니다.
- 무엇보다 제가 사용하는 가장 큰 이유는 [실행 파일을 만들 때](https://gaussian37.github.io/python-etc-pyinstaller/), 필요한 라이브러리만 설치가 되어 있어야 실행 파일의 용량이 작아지기 때문입니다.
- 참고로 아나콘다로 실행파일을 만들면 용량이 장난아니게 큽니다. 700MB 이상이었던 것으로 기억합니다.
- 그러면 가상 환경 세팅에 대하여 알아보겠습니다.

<br>

### 가상 환경 설치

<br>

- 리눅스에서 환경 설정은 아래와 같습니다. 
- 다음과 같이 `pip`를 먼저 설치하고 python3의 버전과 `virtualenv`를 `sudo apt install`을 이용하여 설치합니다.
- 다음으로 `virtualenv` 명령어를 통하여 가상 환경을 구성합니다.

<br>

```
sudo apt install python3-pip
sudo apt install python3.7
sudo apt install virtualenv
virtualenv --python=python3.7 myvenv
```
<br>

- 윈도우에서는 먼저 파이썬을 원하는 버전에 맞게 설치 파일을 받은 다음에 설치를 한 후

<br>

```
pip install virtualenv
virtualenv myenv
```

<br>

- 위 명령어를 통하여 `virtualenv`를 설치하고 리눅스 방식과 동일하게 가상 환경을 만듭니다.

<br>

- 또는 다음 방법도 있습니다.
- `python -m venv 가상환경이름` 으로 가상 환경을 만듭니다. `vevn`는 virtual environment의 줄임말입니다.
    - `python -m venv myvenv` 라고 하면 가상환경 이름을 myvenv로 구성한 것입니다.
- 윈도우 기준으로 가상 환경은 `C:\Users\사용자이름` 폴더에 저장 됩니다.
- 가상 환경을 만들었으면, 가상 환경을 실행한 상태에서 라이브러리 등을 설치해야 합니다.

<br>

### 가상 환경 실행

<br>

- 윈도우에서 가상 환경을 실행하는 방법은 다음과 같습니다.
- `cmd`를 실행합니다. 일반적으로 `cmd`를 실행하면 `C:\Users\사용자이름` 디렉토리에서 시작합니다.
- 그 다음 `가상환경이름\Scripts\activate`를 실행합니다. 그러면 콘솔 창의 각 라인의 가장 아래에 `(가상환경이름)`이 붙습니다.

<br>

## **reqirement 파일 생성**

<br>

- 필요한 라이브러리를 설치 할 때, `pip install 라이브러리_이름`을 이용하여 설치할 수 있습니다.
- 하지만 이미 설치해야 할 목록이 있다면 자동으로 필요한 목록을 다 설치할 수 있습니다.
- 그 목록이 바로 `requirements.txt`입니다.

<br>

```
numpy
matplotlib==1.3.1 
argparse==1.2.1 
```

<br>

- 위와 같이 `requirements.txt`에 필요한 라이브러리 입력하면 됩니다. 필요한 특정 버전이 있으면 위 처럼 `==` 를 이용하여 지정하면 되고 지정하지 않으면 최신 버전이 설치됩니다.
- 그리고 나서 `pip install requirements.txt `를 입력하면 차례대로 설치됩니다.

<br>

## **내가 사용하는 라이브러리들**

<br>

- **영상 처리 용도의 가상 환경 라이브러리**
- 설치 파일 간의 dependency가 있기 때문에 아래 순서로 설치하길 추천 드립니다.

```
cmake
opencv-python
dlib
pyqt5
```

