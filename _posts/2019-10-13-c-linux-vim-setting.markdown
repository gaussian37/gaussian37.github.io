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
- ### vundle 설치
- ### NERDTree 설치
- ### neocomplete 설치
- ### vim에서 사용되는 명령어들

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
- 분할된 창 사이를 이동하려면 `Ctrl+W`을 입력하고 왼쪽 트리에서 커서를 이동하려면 `H(왼쪽),J(위),K(아래),L(오른쪽)`을 입력하면 됩니다.

<br>

## **neocomplete 설치**

<br>

- 먼저 아래 명령어를 통하여 레퍼지토리를 다운로드 합니다.
    - `git clone https://github.com/Shougo/neocomplete.vim`
- 다운한 파일들을 `~/.vim` 디렉토리로 붙여넣기 합니다. `~/.vim` 안에 파일 및 폴더들을 바로 붙어넣어야 합니다.
- `vim ~/.vimrc`를 통하여 `vimrc`를 열고 아래 명령어를 끝에 추가합니다.
    - `let g:neocomplcache_enable_at_startup = 1`
    - 이 명령어는 수동이 아닌 자동으로 자동 옵션을 켜주는 것입니다.
- 레퍼지토리에서 설명한 예시 옵션입니다. `./vimrc`에 위 방법과 동일한 방법으로 명령어를 추가해 주시면 됩니다.

<br>

```
"Note: This option must be set in .vimrc(_vimrc).  NOT IN .gvimrc(_gvimrc)!
" Disable AutoComplPop.
let g:acp_enableAtStartup = 0
" Use neocomplete.
let g:neocomplete#enable_at_startup = 1
" Use smartcase.
let g:neocomplete#enable_smart_case = 1
" Set minimum syntax keyword length.
let g:neocomplete#sources#syntax#min_keyword_length = 3

" Define dictionary.
let g:neocomplete#sources#dictionary#dictionaries = {
    \ 'default' : '',
    \ 'vimshell' : $HOME.'/.vimshell_hist',
    \ 'scheme' : $HOME.'/.gosh_completions'
        \ }

" Define keyword.
if !exists('g:neocomplete#keyword_patterns')
    let g:neocomplete#keyword_patterns = {}
endif
let g:neocomplete#keyword_patterns['default'] = '\h\w*'

" Plugin key-mappings.
inoremap <expr><C-g>     neocomplete#undo_completion()
inoremap <expr><C-l>     neocomplete#complete_common_string()

" Recommended key-mappings.
" <CR>: close popup and save indent.
inoremap <silent> <CR> <C-r>=<SID>my_cr_function()<CR>
function! s:my_cr_function()
  return (pumvisible() ? "\<C-y>" : "" ) . "\<CR>"
  " For no inserting <CR> key.
  "return pumvisible() ? "\<C-y>" : "\<CR>"
endfunction
" <TAB>: completion.
inoremap <expr><TAB>  pumvisible() ? "\<C-n>" : "\<TAB>"
" <C-h>, <BS>: close popup and delete backword char.
inoremap <expr><C-h> neocomplete#smart_close_popup()."\<C-h>"
inoremap <expr><BS> neocomplete#smart_close_popup()."\<C-h>"
" Close popup by <Space>.
"inoremap <expr><Space> pumvisible() ? "\<C-y>" : "\<Space>"

" AutoComplPop like behavior.
"let g:neocomplete#enable_auto_select = 1

" Shell like behavior(not recommended).
"set completeopt+=longest
"let g:neocomplete#enable_auto_select = 1
"let g:neocomplete#disable_auto_complete = 1
"inoremap <expr><TAB>  pumvisible() ? "\<Down>" : "\<C-x>\<C-u>"

" Enable omni completion.
autocmd FileType css setlocal omnifunc=csscomplete#CompleteCSS
autocmd FileType html,markdown setlocal omnifunc=htmlcomplete#CompleteTags
autocmd FileType javascript setlocal omnifunc=javascriptcomplete#CompleteJS
autocmd FileType python setlocal omnifunc=pythoncomplete#Complete
autocmd FileType xml setlocal omnifunc=xmlcomplete#CompleteTags

" Enable heavy omni completion.
if !exists('g:neocomplete#sources#omni#input_patterns')
  let g:neocomplete#sources#omni#input_patterns = {}
endif
"let g:neocomplete#sources#omni#input_patterns.php = '[^. \t]->\h\w*\|\h\w*::'
"let g:neocomplete#sources#omni#input_patterns.c = '[^.[:digit:] *\t]\%(\.\|->\)'
"let g:neocomplete#sources#omni#input_patterns.cpp = '[^.[:digit:] *\t]\%(\.\|->\)\|\h\w*::'

" For perlomni.vim setting.
" https://github.com/c9s/perlomni.vim
let g:neocomplete#sources#omni#input_patterns.perl = '\h\w*->\h\w*\|\h\w*::'
```

<br>

## **vim에서 사용되는 명령어들**

<br>


