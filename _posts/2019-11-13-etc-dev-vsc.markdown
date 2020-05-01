---
layout: post
title: Visual Studio Code 기능
date: 2019-11-13 00:00:00
img: etc\dev\vsc\vsc.png
categories: [etc-dev] 
tags: [비주얼 스튜디오 코드 기능, visual studio code] # add tag
---

<br>

## **목차**

- ### Visual Studio Code 단축키
- ### Visual Studio Code에서 Python interpretor 설정
- ### Visual Studio Code에서 C/C++ 실행하기

<br>

## **Visual Studio Code 단축키**

<br>

- `ctrl + d` : 동일한 단어를 모두 한번에 잡아주고 동시에 수정이 가능하도록 해줍니다.
- `ctrl + shift + l` : 현재 선택한 문자열과 동일한 문자열을 동시에 선택합니다. (`ctrl + d` 를 여러번 누른것과 비슷함)

<br>

- `ctrl + [` : 내어쓰기 (커서가 라인의 어디에 위치하든 상관없이 그 라인 전체를 내어쓴다)
- `ctrl + ]`: 들여쓰기 (커서가 라인의 어디에 위치하든 상관없이 그 라인 전체를 들여쓴다)

<br>

- `ctrl + alt + up/down` : 커서를 여러 줄에 동시에 생성하기
- `alt + 클릭` : 클릭한 지점에 모두 커서를 생성시킵니다. 동일한 입력이 필요한 곳에 모두 커서를 생성한 다음에 한번에 입력할 수 있습니다. 
- `alt + up/down` : 현재 커서의 줄을 위 또는 아래로 옮깁니다. 따라서 copy & paste를 할 필요가 없어집니다.
- `alt + shift + up/down` : 현재 커서의 줄을 위 또는 아래로 복사합니다.

<br>

- `ctrl + /` : 라인 기준으로 주석 처리 (커서가 라인 중간에 있더라도 커서 앞부분 포함하여 그 라인 전체가 주석 처리됨)
- `shift + alt + a` : 커서 위치 기준으로 주석 처리

<br>

- `alt + shift + i` : 블록을 씌운 라인 전체에 커서를 생성시킵니다. 블록 씌운 영역에 동일한 작업을 하기에 용이합니다. 
- `alt + shift + drag` : 마우스 커서가 블록을 씌우는 곳에 모두 커서가 생깁니다. 
- ctrl + ` : 터미널 띄우기 숨기기
- ctrl + shift + ` :  새 터미널 
- `ctrl + shift + k` : 행 삭제
- `ctrl + x` : 행 잘라내기
- `ctrl + c` : 행 복사하기

<br>

## **Visual Studio Code에서 Python interpretor 설정**

<br>

- 파이썬을 사용할 때 가상 환경을 사용하면 VSC에서 그 환경에 맞는 파이썬 인터프리터를 지정해 주어야 합니다.
- VSC에서 왼쪽 아래에 `Python.3.XXX.64 -bit`라고 적혀있는 곳을 클릭하면 파이썬 인터프리터를 선택할 수 있는 창이 뜹니다.
- 여기서 현재 사용하려고 하는 가상 환경의 파이썬 실행 파일을 등록해주어야 VSC에서 원하는 가상 환경을 사용할 수 있습니다.
- 파이썬 실행 파일을 등록하는 방법은 다음과 같습니다.

<br>

1) `File` → `Preferences` → `Settings`를 선택합니다. <br>
2-1) 검색창에 `Files:Association`을 입력합니다. <br>

<br>

```
Files: Associations
Configure file associations to languages (e.g. "*.extension": "html"). These have precedence over the default associations of the languages installed.

Edit in settings.json
```

<br>

2-2) 위와 같은 검색 결과에서 `Edit in settings.json`을 클릭합니다. <br>
3) 가장 아래에 `python.pythonPath": "Your_venv_path/Scripts/python.exe`를 추가합니다. 이 때, 가상환경의 python.exe를 등록해야 합니다. 가상환경의 디렉토리를 찾고 싶으면 다음 [링크](https://gaussian37.github.io/python-concept-initial_setting/)를 클릭하시면 확인할 수 있습니다. <br>
4) VSC를 재시작하면 새로 등록된 python interpretor를 볼 수 있습니다. 

<br>

## **Visual Studio Code에서 C/C++ 실행하기**

<br>

- 먼저 리눅스 환경의 visual studio code 에서 C를 실행하는 방법에 대하여 알아보겠습니다.
- 비주얼 스튜디오 코드를 설치한 후에 빌드 관련 프로그램을 설치합니다.

<br>

```python
sudo apt install build-essential
```

<br>

- 그 다음 플러그인 2개를 설치합니다.
    - `C/C++` 플러그인과 `Code Runner`

<br>

- 리눅스에는 기본적으로 C/C++ 컴파일러가 설치되어 있기 떄문에 위 작업만 하면 C 코드 실행이 가능합니다.