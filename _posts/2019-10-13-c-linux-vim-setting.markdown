---
layout: post
title: vim 기능 및 세팅 정리
date: 2019-02-09 00:00:00
img: c/linux/vim/vim.png
categories: [c-linux] 
tags: [vim, vim 세팅] # add tag
---

- 이 글에서는 리눅스의 `vim`관련 경험이나 기능등을 한 곳에 정리하기 위해 만든 글입니다.

<br>

## **목차**

<br>

- ### .vimrc 세팅값
- ### vim에서 사용되는 명령어들
- ### vim 자동 완성 기능
- ### 명령어 취소
- ### vim으로 다중 파일 열기

<br>

## **.vimrc 세팅값**

<br>

```
" configure expanding of tabs for various file types
au BufRead,BufNewFile *.py set expandtab
au BufRead,BufNewFile *.c set noexpandtab
au BufRead,BufNewFile *.h set noexpandtab
au BufRead,BufNewFile Makefile* set noexpandtab

set number            " line 표시를 해줍니다.
set ai                    " auto indent
set si                    " smart indent
set cindent            " c style indent
set shiftwidth=4      " shift를 4칸으로 ( >, >>, <, << 등의 명령어)
set tabstop=4         " tab을 4칸으로
set ignorecase      " 검색시 대소문자 구별하지않음
set hlsearch         " 검색시 하이라이트(색상 강조)
set expandtab       " tab 대신 띄어쓰기로
set smartindent
set smarttab
set background=dark  " 검정배경을 사용할 때, (이 색상에 맞춰 문법 하이라이트 색상이 달라집니다.)
set nocompatible   " 방향키로 이동가능
set fileencodings=utf-8,euc-kr    " 파일인코딩 형식 지정
set bs=indent,eol,start    " backspace 키 사용 가능
set history=1000    " 명령어에 대한 히스토리를 1000개까지
set ruler              " 상태표시줄에 커서의 위치 표시
set nobackup      " 백업파일을 만들지 않음
set title               " 제목을 표시
set showmatch    " 매칭되는 괄호를 보여줌
set nowrap         " 자동 줄바꿈 하지 않음
set wmnu           " tab 자동완성시 가능한 목록을 보여줌
set showcmd        " show (partial) command in status line
syntax on        " 문법 하이라이트 킴"
# set paste
```

<br>

- 위 값들은 vim을 좀 더 편리하게 쓰기 위하여 기본적으로 설정한 값입니다.

<br>

## **vim에서 사용되는 명령어들**

<br>

- `문자열 검색` : `/` 키를 눌러 검색 입력 활성화 후 검색할 단어를 입력합니다. `n` 키로 다음 검색 결과를 찾고 `shift + n` 키로 이전 검색 결과로 커서를 이동합니다.
- `문자열 바꾸기` : `%s/찾을문자열/바꿀문자열/옵션`형태로 사용합니다. 예를 들어 `%s/apple/banana/g` 라고 하면 문서 전체의 apple을 banana로 바꾸는 것입니다. 대표적인 옵션은 3가지로 아래와 같습니다.
    - `g` : 매칭되는 문자열을 물어보지 않고 변경
    - `i` : 대소문자를 구분하지 않고 변경 (ignore cases)
    - `c` : 매칭되는 문자열마다 바꿀 것인지 사용자에게 물어보고 변경

<br>

- `1 줄 복사` : `y`를 눌러서 (yank) 그 줄을 복사하고 `p`를 눌러서 원하는 위치에 붙여 넣기 할 수 있습니다.
- `여러 줄 복사` : `v`를 누르면 복사할 영역을 선택할 수 있도록 활성화 되고 복사할 영역을 선택한 뒤 `y`를 누르면 복사할 영역이 선택되어집니다. `p`를 눌러서 원하는 위치에 붙여 넣기 할 수 있습니다.

<br>

## **vim 자동 완성 기능**

<br>

- vim에서 사용할 수 있는 자동 완성 기능은 크게 **① vim에서 자체 지원하는 자동 완성 기능**과 **② 추가 플러그인 설치 방법**이 있습니다. 이번 글에서는 간단한 자체 기능과 추가 플러그인 중 `YouCompleteMe`에 대하여 알아보도록 하겠습니다.

<br>

- **① vim에서 자체 지원하는 자동 완성 기능**
- 사용 방법 : `Ctrl + N` 또는 `Ctrl + P` 을 입력
- 사용 범위 :  `complete` 옵션에서 지정한 위치의 키워드를 기반으로 자동 완성을 해줍니다. 보통 IDE의 자동 완성과는 다르게, 주석이나 문자열 안에 있는 단어들도 모두 찾아줍니다. 기본적으로 적용되어 있는 자동 완성 기능 옵션은 `complete=.,w,b,u,t,i`입니다.
    - `.` : 현재 편집중인 버퍼의 모든 단어를 자동완성 소스로 사용합니다.
    - `w` : vim에 현재 열려 있는 window들의 모든 단어를 사용합니다.
    - `b` : 버퍼 리스트에 있고 로드된 버퍼들의 모든 단어를 사용합니다.
    - `u` : 버퍼 리스트에 있고 로드되지 않은 버퍼들의 모든 단어를 사용합니다.
    - `t` : tag completion을 사용합니다.
    - `i` : 현재 파일과 include된 파일의 단어를 사용합니다.

<br>

- **② 추가 플러그인 설치 방법**
- `YouCompleteMe` : https://github.com/ycm-core/YouCompleteMe

<br>

## **명령어 취소**

<br>

- vim의 명령어 모드에서 `u`를 누르면 직전에 편집한 작업을 취소할 수 있습니다.
- 만약, 현재 줄에서 수정한 내용을 취소하고 싶으면 `shift + u`를 누르면 됩니다. 이 작업은 **현재 커서가 위치하고 있는 줄에서 했던 모든 작업이 취소**됩니다.
- `u`나 `shift + u`를 통하여 취소한 작업을 다시 원복하고 싶을 때, `ctrl + r`를 통하여 **redo** 할 수 있습니다. 즉, 취소를 취소하는 작업입니다.

<br>

## **vim으로 다중 파일 열기**

<br>

- vim에서 한번에 여러 개의 파일을 열 수 있는 옵션이 있습니다. 대표적으로 수평 분할, 수직 분할, 탭 분할이 있으며 각 옵션은 `-o`, `-O`, `-p`에 해당합니다.
- 사용 방법 : `vim -O file1.txt file2.txt`
- `-o` : 여러 개의 파일을 **수평 분할**로 보여줍니다. (분할된 화면 간 이동 : `ctrl + w`)
- `-O` : 여러 개의 파일을 **수직 분할**로 보여줍니다. (분할된 화면 간 이동 : `ctrl + w`)
- `-p` : **윈도우 모드**로 보여줍니다. (분할된 탭 간 이동 : 명령어 모드에서 다음 명령어를 직접 입력해야 합니다. 이전 탭(`:tabp`), 다음 탭(`:tabn`), n번째 탭(`:tabm n`)

<br>

- 만약 vim을 이용하여 어떤 문서를 연 상태에서 추가적으로 vim 내부에서 다른 문서를 추가해서 보려면 명령어 모드에서 새로운 문서를 추가할 수 있습니다.
- `split 경로/문서명` : **수평 분할**로 새로운 문서가 추가됩니다.
- `vsplit 경로/문서명` : **수직 분할**로 새로운 문서가 추가됩니다.
- `tabedit 경로/문서명` : **윈도우 모드**로 새로운 문서가 추가됩니다.


