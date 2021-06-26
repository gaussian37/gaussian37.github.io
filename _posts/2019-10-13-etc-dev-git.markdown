---
layout: post
title: git 관련 내용 정리
date: 2019-02-09 00:00:00
img: etc/dev/git/0.png
categories: [etc-dev] 
tags: [git, git 셋팅] # add tag
---

<br>

- 이 글은 `git`과 관련된 기능 또는 유용한 팁 등을 정리한 글입니다.

<br>

## **목차**

<br>

- ### [git 관련 셋팅](#git-관련-셋팅-1)
- ### [자주 사용하는 명령어](#자주-사용하는-명령어-1)
- ### [버전 컨트롤](#버전-컨트롤-1)
- ### [.git](#git-1)
- ### [git status](#git-status-1)
- ### [git diff](#git-diff-1)
- ### [git ignore](#git-ignore-1)
- ### [git 작업 취소 관련](#git-작업-취소-관련-1)

<br>

## **git 관련 셋팅**

<br>

- 링크 : https://medium.com/@thucnc/how-to-show-current-git-branch-with-colors-in-bash-prompt-380d05a24745
- 링크 : https://wjs890204.tistory.com/886
- 우분투 터미널에서 git을 사용할 때, 어떤 branch에 접속한 상태인 지 확인이 필요하거나 color로 터미널에서 git의 상태가 구분되었으면 좋을 때 다음 2가지 방법을 사용할 수 있습니다.
- **잘못된 git branch를 사용하는 실수를 방지하기 위하여** 아래 셋팅은 반드시 하길 권장 드립니다.
- 1) 아래 코드를 `.bashrc`에 입력해 놓는 방법입니다. 단, 우분투 터미널의 기본적인 글자 색도 바뀌니 그 점은 유의하시기 바랍니다.
- `.bashrc` 파일 열기 : `gedit ~/.bashrc`

<br>

```
parse_git_branch() {
     git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "
```

<br>

- 2) `oh-my-zsh` 을 이용하는 방법입니다. 아래 3줄의 설치 과정을 따라하시면 됩니다.
- `sudo apt-get install zsh`
- `sudo apt-get install git wget curl`
- `curl -L https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | sh`
- 설치가 완료되면 터미널에서 `zsh` 명령어를 통하여 `oh-my-zsh`를 사용할 수 있습니다. `oh-my-zsh` 환경에서 git이 연관된 폴더에 들어가면 command line에 branch 이름이 나타납니다.

<br>

## **자주 사용하는 명령어**

<br>

- 먼저 git을 사용할 때, 아래 명령어 리스트를 이용하여 깃 서버에 업데이트 가능합니다. 

<br>

```
git checkout feature
git add .
git commit -m "message"
git push origin feature
```

<br>

- 원하는 branch로 이동하는 방법 : `git checkout feature이름`
- 원하는 branch에 push 하는 방법 : 
    - 1) 변경할 파일 등록 : `git add . ` 또는 `git add 파일명1 파일명2` 
    - 2) 변경 내용 commit : `git commit 파일명1 파일명2 -m "변경내용"`
    - 3) 변경 내용 push : `git push origin feature이름`

<br>

- 위 4가지 명령어는 레퍼지토리를 업데이트 하기 위한 가장 기본적인 명령어 입니다. 그 이외의 필수 기능들에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/etc/dev/git/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 출처 : https://web.facebook.com/bigdatastudy/photos/a.110625387093056/272139340941659/

<br>


- 설명 추가 예정



<br>

## **버전 컨트롤** 

<br>

- 버전 제어 시스템은 프로젝트의 파일 및 디렉터리에 대한 변경 사항을 관리하는 도구입니다. 
- 많은 버전 제어 시스템이 존재하지만 `git`는 여러가지 장점이 있습니다.
- 먼저 `git`에 저장된 결과는 손실되지 않으므로 항상 프로그램의 각 버전에서 생성된 결과를 확인할 수 있습니다.
- 다른 사용자의 작업과 충돌할 때 `git`은 자동으로 사용자에게 알려주기 때문에 실수로 작업을 덮어쓰는 것을 방지하는 데 도움을 줍니다.
- `git`은 다른 컴퓨터에서 다른 사람이 수행한 작업을 동기화 할 수 있으므로 팀원끼리 사용하는 데에도 도움을 줍니다.
- 또한 버전 관리는 단순히 소프트웨어에 국한된 것이 아닙니다. 책, 종이, 매개변수 세트 및 시간이 지남에 따라 변경되거나 공유되어야 하는 모든 것은 `git`를 이용하여 저장하고 공유할 수 있습니다.
- 정리하면 `git`으로 할 수 있는 대표적인 기능은 다음과 같습니다.
    - **파일의 변화를 추적**할 수 있습니다.
    - **같은 파일을 서로 다른 사람들이 변경**을 하였을 때, 발생할 수 있는 **충돌 등을 확인**할 수 있습니다.
    - 어떤 파일의 변화가 발생하였을 때, **서로 다른 컴퓨터에서 동기화** 할 수 있습니다.

<br>

## **.git**

<br>

- 각 git 프로젝트에는 두가지 부분이 있으며 **이 두 가지를 결합한 것을 repository**라고 합니다.
    - 직접 생성하고 편집하는 파일 및 디렉터리
    - git에서 프로젝트의 기록에 대해 기록하는 추가 정보
- git은 repository의 루트 디렉터리에 있는 `.git`라는 디렉터리에 모든 추가 정보를 저장합니다.
- git은 이 정보가 매우 정확한 방법으로 제공될 것으로 예상하므로 `.git`의 어떤 것도 편집하거나 삭제해서는 안 됩니다.

<br>

## **git status**

<br>

- Git을 사용하는 경우 repository의 상태를 확인하는 경우가 많습니다. 
- 이렇게 하려면 명령어 `git status`를 실행하여 마지막으로 변경 내용을 저장한 이후 수정된 파일 목록을 통해 확인할 수 있습니다.

<br>
<center><img src="../assets/img/etc/dev/git/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>


<br>

- git에는 아직 저장되지 않은 변경 사항과 함께 파일을 저장하는 `스테이징 영역`이 있습니다. 
- 파일을 staging 영역에 넣는 것은 상자에 물건을 넣는 것과 같으며, 이러한 변경 사항을 `committing`하는 것은 우편함에 상자를 넣는 것과 같습니다. 
- 원하는 만큼 더 많은 항목을 추가하거나 자주 꺼내 놓을 수 있지만, 우편함에 파일을 넣은 후에는 더 이상 변경할 수 없습니다.

<br>
<center><img src="../assets/img/etc/dev/git/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `git status`는 이 stating 영역에 있는 파일과 아직 배치되지 않은 변경 사항이 있는 파일을 표시합니다. 
- 현재 파일을 마지막으로 저장한 파일과 비교하기 위해 `git diff 파일이름`을 사용할 수 있습니다. 
- 파일 이름 없이 `git diff`는 저장소의 모든 변경 사항을 표시하는 반면 `git diff 디렉토리`는 일부 디렉토리의 파일에 대한 변경 내용을 표시합니다.

<br>

## **git diff**

<br>

- diff는 두 파일 간의 차이를 표시합니다. 예를 들어 다음과 같은 `diff` 내용을 확인해 보겠습니다.

<br>

```
diff --git a/report.txt b/report.txt
index e713b17..4c0742a 100644
--- a/report.txt
+++ b/report.txt
@@ -1,4 +1,5 @@
-# Seasonal Dental Surgeries 2017-18
+# Seasonal Dental Surgeries (2017) 2017-18
+# TODO: write new summary 
```

<br>

- 1번줄: 출력을 생성하는 데 사용되는 명령입니다(이 경우 diff --git). 여기서 `a`와 `b`는 "첫 번째 버전"과 "두 번째 버전"을 의미하는 자리 표시자입니다.
- 2번줄: Git의 내부 변경 데이터베이스에 키를 표시하는 인덱스 라인입니다.
- 3,4번줄: `--- a/report.txt` and `+++ b/report.txt`에서 `-`로 시작하는 것은 제거된 항목이고 `+`로 시작하는 것은 추가된 항목입니다.
- 5번줄: @@로 시작하는 행은 변경되는 위치를 알려줍니다. 숫자 쌍은 시작선과 줄 수입니다.

<br>

## **git ignore**

<br>

- git을 사용할 때, 어떤 파일이나 폴더는 git 관리 대상에서 제외하고 싶은 경우가 발생합니다. 예를 들면 어떤 파일을 실행 하였을 때, 부산물로 생기는 임시 리소스 들이나 다른 개발자들과 공유할 필요 없는 파일 그리고 로그인 암호나 키 등 개인 정보가 있는 파일들이 있습니다.
- 이러한 파일들은 git이 관리하지 않고 무시하도록 만들기 위하여 `.gitignore`라는 파일을 최상위에 만들어서 그 하부 디렉토리에 있는 특정 파일이나 폴더를 관리하지 않게 만들 수 있습니다.
- 이와 같이 처리 되었을 때, **git에 의해 변경 내역이 추적되지 않고**, `git status`나 `git diff`와 같은 명령어 수행시에 검출되지 않고 `git add .`와 같은 명령어 수행시에도 stage 영역에 추가되지 않습니다.
- `.gitignore` 문서 작성 방법은 다음과 같습니다.

<br>

- `#` : #으로 시작하는 줄은 주석 처리합니다. 주석은 반드시 한 줄 자체를 주석으로 사용해야 하며 ignore 규칙과 같은 줄에 작성될 수 없습니다.
- `file_name.ext` : file_name.ext라는 이름의 파일 및 확장자에 해당하는 파일을 ignore 합니다. 파일명만 규칙으로 두면 모든 하위 디렉토리에 모두 적용됩니다.
- `path/to/.../the/file_name.ext` : 특정 디렉토리 밑에 파일 이름을 지정하면 그 특정 경로의 파일만 ignore 규칙으로 적용됩니다.
- `path/to/.../the/folder/` : 마지막에 `/`를 붙이면 그 디렉토리 전체와 디렉토리 내의 모든 내용들이 ignore 처리됩니다.
- `path/to/.../the/[fF]older` : folder/, Folder/ 모두 ignore 되도록 복수 규칙을 적용할 수 있습니다.
- `path/to/.../the/folder/*.ext` : 특정 폴더의 특정 확장자를 모두 무시할 수 있습니다.
- `*.ext` : 특정 확장자를 모두 무시할 수 있습니다.
- `**/folder/` : folder 라는 이름의 디렉토리가 저장소 내 어떤 위치에 존재하더라도 모두 ignore 처리됩니다. 마지막에 반드시 `/`를 붙여야 합니다. 이 경우 folder 라는 이름의 하위 디렉토리 및 파일 보두 ignore 처리됩니다.
- `folder_A/**/folder_B/` : folder_A 내부에 있는 folder_B 하부의 모든 파일 및 디렉토리가 모두 ignore가 됩니다. folder_A와 folder_B 사이는 대상에서 제외됩니다.

<br>

## **git 작업 취소 관련**

<br>

- git에서 작업을 하다가 작업한 내용을 취소해야 하는 경우가 발생합니다. 글 도입부의 flow를 살펴보면 git에는 4가지 단계가 있습니다.
- `working directory` : 현재 작업 공간을 뜻하며 git에 어떠한 반영도 하지 않은 상태입니다.
- `stage area` : git add를 통하여 변경된 내용을 commit 하기 위한 준비 상태로 둡니다.
- `local repository` : git commit을 마친 상태로 실제 업데이트 하기 위한 대상이 된 상태입니다.
- `remote repository` : 최종적으로 사용되고 있는 git 서버에서의 상태입니다.

<br>

- 위에서 살펴본 4가지 단계에서 현재 작업한 내용을 취소하고 싶을 때, 각 단계에서 다음과 같이 취소 가능합니다.
- **git add 명령 이전 (stage에 올리지 않은 상태)** : 
    - repository 내 모든 수정 되돌리기 : `git checkout .`
    - 특정 폴더 아래의 모든 수정 되돌리기 : `git checkout {dir}`
    - 특정 파일의 수정 되돌리기 : `git checkout {file_name}`
- **git add 명령으로 stage에 올린 경우** : `git reset`
- **git commit을 한 경우** : 
    - commit 내용을 없애고 이전 상태로 원복 : `git reset --hard HEAD^` (브랜치의 마지막 커밋을 가리키던 HEAD를 그 이전으로 이동시켜서 commit 내용을 없앰)
    - commit은 취소하고 commit 했던 내용은 남기고 unstaged 상태로 만들기 : `git reset HEAD^`
    -  commit은 취소하고 commit 했던 내용은 남기고 staged 상태로 만들기 : `git reset --soft HEAD^`
- **모든 untracked 파일들을 지우기** : `git clean -fdx`
- **git push를 한 경우 remote repository도 이전으로 되돌리기**
    - ① `git reset HEAD^` : local repository에서 commit을 하나 되돌림
    - ② `git commit -m "revert reason comments"` : local repository의 직전 상태로 되돌린 것으로 commit
    - ③ `git push origin branch명` : local repository의 직전 상태를 다시 push 함으로써 remote repository를 강제로 revert 시킴