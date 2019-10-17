---
layout: post
title: vim 기능 및 세팅 정리
date: 2019-02-09 00:00:00
img: nd/linux/vim/vim.png
categories: [etc-linux] 
tags: [vim, vim 세팅] # add tag
---

- 이 글에서는 리눅스의 `vim`관련 경험이나 기능등을 한 곳에 정리하기 위해 만든 글입니다.

<br>

## **목차**

<br>

- ### .vimrc 세팅값
- ### NERDTree 설치

<br>

## **.vimrc 세팅값**

<br>

```
set number            " line 표시를 해줍니다.
set ai                    " auto indent
set si                    " smart indent
set cindent            " c style indent
set shiftwidth=4      " shift를 4칸으로 ( >, >>, <, << 등의 명령어)
set tabstop=4         " tab을 4칸으로
set ignorecase      " 검색시 대소문자 구별하지않음
set hlsearch         " 검색시 하이라이트(색상 강조)
set expandtab       " tab 대신 띄어쓰기로
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
syntax on        " 문법 하이라이트 킴"
```

<br>

- 위 값들은 vim을 좀 더 편리하게 쓰기 위하여 기본적으로 설정한 값입니다.

<br>

## **vundle 설치하기**

<br>

- `vundle`은 `vim`의 플러그인을 쉽게 설치하기 위해 관리하는 기능입니다.
- 관련 github은 아래 링크를 참조하시기 바랍니다.
    - https://github.com/VundleVim/Vundle.vim

- 먼저 설치하려면 아래 코드를 커맨드 라인에서 다음을 입력해야 합니다.

<br>

`
git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
`

<br>

- 설치를 한 다음에 `vim .vimrc`를 입력하여 아래 값들을 삽입해 줍니다.

<br>

```
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
```

<br>

- 다음으로 `vim`을 실행하고 `:PluginInstall`을 입력하거나 커맨드라인에 바로 `vim +PluginInstall +qall`을 입력하여 설치합니다.

<br>

## **NERDTree 설치하기**

<br>

- `NERDTree`는 `vim`의 왼쪽에 디렉토리 형태를 보여주는 기능을 말합니다.
- `NERDTree`를 설치하기 전에 먼저 위에서 `vundle`을 설치해 주어야 합니다. 그러면 `vundle`을 설치하였다는 가정 하에서 진행해 보겠습니다.
- 먼저 `.vimrc`을 열어서 아래 코드를 마지막에 삽입해 줍니다.

<br>

```
" Keep Plugin commands between vundle#begin/end.
 
Plugin 'vim-airline/vim-airline'
Plugin 'vim-airline/vim-airline-themes'
Plugin 'The-NERD-Tree' 
```

<br>

- 새로운 플러그인을 등록하였으므로 vim을 실행하고 `:PlugInstall`을 입력하면 플러그인이 설치됩니다.
- 이후 vim을 재시작 한 뒤 `:NERDTree`를 입력하면 파일시스템을 볼 수 있는 창이 생깁니다.
- 분할된 창 사이를 이동하려면 Ctrl+W을 입력하고 왼쪽 트리에서 커서를 이동하려면H(왼쪽),J(위),K(아래),L(오른쪾)을 입력하면 됩니다.