---
layout: post
title: WSL(Windows Subsystem for Linux) 사용 관련 snippets
date: 2019-11-23 00:00:00
img: etc/dev/wsl/0.png
categories: [etc-dev] 
tags: [window10, 윈도우 10, wsl, wsl2] # add tag
---

<br>

- 이 글에서는 `WSL(Windows Subsystems for Linux)`을 사용하면서 필요한 것 또는 발생한 문제들을 정리하였습니다.
- 엔지니어라면 가성비를 중요시 생각해야 하므로 (사실 MAC 살 돈이 없어서...) 윈도우 또는 리눅스를 쓰려고 하는데, 윈도우에서 리눅스를 VMware 같은 방식이 아닌 진짜 subsystem 형식으로 사용할 수 있게 되어 너무나 기분이 좋습니다.
- 이제는 가성비 좋은 노트북에 윈도우를 설치하고 `WSL` 까지 설치하면 윈도우와 리눅스 2가지를 이용할 수 있으니 특히 개발자들이 매우 좋을듯 합니다.
- 하지만, 아직 까지는 과도기 이므로 안되는 기능이 많이 있습니다. 언젠가 이 점들은 해결될 거라 생각하면 계속 이 글에서 Trend를 따라가 보겠습니다.

<br>

## **목차**

<br>

- ### sudo apt install 이 안될 때
- ### wsl에서 사용하면 좋은 export
- ### GUI 방식으로 실행 시키고 싶을 때
- ### wsl2 셋팅 방법

<br><br>

## **sudo apt install 이 안될 때**

<br>

- `sudo apt install`을 하였을 때, `Unable to locate package` 라는 에러가 발생하는 경우가 있습니다. 그 경우는 Ubuntu의 Repository 접근 권한이 허용이 안된 경우 입니다. 이 경우 아래 명령어를 통해 enable 시켜주면 됩니다. 또는 이와 유사하게 repository 접근이 안되는 경우 아래 명령어를 통하여 해결할 수도 있습니다.

<br>

```
sudo add-apt-repository universe
```

<br>

## **wsl에서 사용하면 좋은 export**

<br>

- wsl을 사용할 때, 불편할 것들을 개선하기 위하여 제가 사용하는 `export`등을 적어 놓아봅니다.
- 특히 `export` 할 것은 기본 경로(`/home/{USER}` 또는 `~/`)의 `.bashrc` 파일에 적어놓으면 wsl을 매번 켤 때 마다 자동으로 호출됩니다.
- 즉, `vi ~/.bashrc` 또는 `vim ~/.bashrc`를 입력하여 `.bashrc` 파일을 열고 가장 아래에 다음 export 를 추가하면 됩니다.

<br>

```
export LINUX_USER=리눅스_사용자명
export WIN_USER=윈도우_사용자명
export WIN_HOME=/mnt/c/Users/$WIN_USER
```

<br>

- 참고로 `./bashrc`에 명령어들을 추가한 뒤 적용하고 싶으면 리눅스 재부팅 또는 `. ~/.bashrc` 명령어를 실행시키면 됩니다.

<br>

## **GUI 방식으로 실행 시키고 싶을 때**

<br>

- **2020.05.01 기준으로 wsl1에서만 GUI가 됩니다. wsl2에서는 can't display 오류가 발생하오니 주의 바랍니다.**
- `wsl`에서 실행한 프로그램이 CLI 환경에서만 돌아간다면 실제 사용하는 리눅스에 비해 너무나 불편합니다.
- 이번에 살펴볼 내용은 wsl에서 GUI 기반의 프로그램을 실행시키고 실제로 GUI(eclipse, gedit 등등)를 실행시켜 보겠습니다.
- 참고로 **visual studio code**는 MS에서 특별히 신경을 쓴 덕분인지, 따로 셋팅을 하지 않아도 `code`란 명령어를 통해 GUI 환경을 실행시킬 수 있습니다.

<br>

- ① `./bashrc`에 `export DISPLAY=:0.0`를 추가합니다. vi/vim을 열어서 마지막에 추가해도 되고 아래 명령어를 이용하면 자동 추가 됩니다.
    - echo "export DISPLAY=:0.0" >> ~/.bashrc
- ② `. ~/.bashrc` 명령어 실행
- ③ 다음 sourceforge 링크를 접속하여 `VcXsrv`를 설치합니다. 이것은 윈도우 용 `X server`로 window에서 `wsl`에 접근할 때 사용한다고 생각하시면 됩니다. ①에서 설정한 DISPLAY 주소인 localhost:0.0 으로 X server가 접근하여 Graphic을 출력합니다.
    - 링크 : https://sourceforge.net/projects/vcxsrv/
- ④ 설치가 완료되면 `XLaunch`란 것이 생기는 데 실행합니다. 실행하면 셋팅값들이 나오는데 default 값을 사용해도 상관없으니 계속 다음을 누르시면 됩니다.
- ⑤ wsl에서 graphic 기반의 프로그램을 아무거나 실행해 보시면 됩니다. 아래는 `gedit`을 한번 실행해 본 결과입니다.

<br>
<center><img src="../assets/img/etc/dev/wsl/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 만약 실제 Ubuntu와 같이 GUI 환경에서 리눅스를 사용하고 싶으면 여러가지 방법이 있겠지만 이 글에서는 


<br>

## **wsl2 셋팅 방법**

<br>

- 참조 : https://docs.microsoft.com/en-us/windows/wsl/wsl2-install

<br>

- 윈도우 10에서 부터는 `wsl(windows subsystem for linux)`라는 꽤 괜찮은 기능을 제공합니다.
- 이 기능은 윈도우에서 리눅스 우분투 터미널을 사용할 수 있도록 제공하여 윈도우에서 개발에 불편함을 겪는 분들에게 나름의 단비 같은 존재로 거듭나고 있습니다. 물론 아직 까진 개선할 점이 많긴 하지만요...
- 어찌되었든 간에 윈도우에서 리눅스를 같이 쓸 수 있다면 사실상 원가절감할 수 있는 아주 좋은 포인트라고 생각됩니다.

<br>

- 현재 (2020년 초 기준) `Microsoft store`에서 `Ubuntu`를 설치하게 되면 `wsl` 1버전이 기본적으로 설정되는데, 여기서 설정만 2버전으로 바꾸어 주면 wsl 2를 사용할 수 있습니다.

<br>

### **wsl 2 란 무엇인가?**

<br>

- 먼저 설치 방법에 앞서서 `wsl 2`가 무엇인지 부터 알아보겠습니다. 아래는 MS 홈페이지의 내용을 발췌하였습니다.

<br>

```
※ About WSL 2
WSL 2는 Windows 용 Linux 하위 시스템이 Windows에서 ELF64 Linux 바이너리를 실행하도록하는 새로운 버전의 아키텍처입니다. 주요 목표는 파일 시스템 성능을 높이고 전체 시스템 호출 호환성을 추가하는 것입니다. 이 새로운 아키텍처는 이러한 Linux 바이너리가 Windows 및 컴퓨터의 하드웨어와 상호 작용하는 방식을 변경하지만 WSL 1 (현재 널리 사용되는 버전)과 동일한 사용자 환경을 제공합니다. 개별 Linux 배포판은 WSL 1 배포판 또는 WSL 2 배포판으로 실행하거나 언제든지 업그레이드 또는 다운 그레이드 할 수 있으며 WSL 1 및 WSL 2 배포판을 나란히 실행할 수 있습니다. WSL 2는 실제 Linux 커널을 사용하는 완전히 새로운 아키텍처를 사용합니다.

※ Linux kernel in WSL 2
WSL 2의 Linux 커널은 kernel.org에서 사용할 수있는 소스를 기반으로 구축되었습니다. 이 커널은 WSL 2에 맞게 특별히 조정되었습니다. 크기와 성능에 최적화되어 Windows에서 놀라운 Linux 환경을 제공하고 Windows 업데이트를 통해 서비스를 제공하므로 관리 할 필요없이 최신 보안 수정 사항과 커널 개선 사항을 얻을 수 있습니다.
또한이 커널은 오픈 소스입니다. Linux 커널의 전체 소스 코드는 여기에서 찾을 수 있습니다. 이 커널에 대한 자세한 내용을 보려면 해당 팀이 작성한이 블로그 게시물을 확인하십시오.

※ Brief overview of the WSL 2 architecture
WSL 2는 최신 가상화 기술을 사용하여 경량 유틸리티 가상 머신 (VM) 내에서 Linux 커널을 실행합니다. 그러나 WSL 2는 일반적인 VM 환경이 아닙니다. 기존 VM 환경은 부팅 속도가 느리고 격리되고 많은 리소스를 소비하며 이를 관리하는 데 시간이 걸립니다. WSL 2에는 이러한 속성이 없습니다. 여전히 WSL 1의 놀라운 이점을 제공합니다. Windows와 Linux 간의 높은 수준의 통합, 매우 빠른 부팅 시간, 적은 리소스 풋프린트 및 무엇보다도 VM 구성이나 관리가 필요하지 않습니다. WSL 2는 VM을 사용하지만 WSL 1과 동일한 사용자 환경을 유지하면서 가상 환경에서 관리 및 실행됩니다.

※Increased file IO performance
git clone, npm install, apt update, apt upgrade 등과 같은 파일 집약적 인 작업은 모두 훨씬 빠릅니다. 실제 속도 증가는 실행중인 앱과 파일 시스템과의 상호 작용 방식에 따라 다릅니다. 압축 된 tarball의 압축을 풀 때 WSL 2의 초기 버전은 WSL 1에 비해 최대 20 배 더 빠르며, 다양한 프로젝트에서 git clone, npm install 및 cmake를 사용할 때는 약 2-5 배 더 빠릅니다.

※ Full System Call Compatibility
Linux 바이너리는 시스템 호출을 사용하여 파일 액세스, 메모리 요청, 프로세스 작성 등과 같은 많은 기능을 수행합니다. WSL 1은 WSL 팀이 구축 한 변환 계층을 사용했지만 WSL 2에는 전체 시스템 호출 호환성을 갖춘 자체 Linux 커널이 포함되어 있습니다. Docker 등 WSL 내부에서 실행할 수있는 완전히 새로운 앱 세트를 소개합니다. 또한 Linux 커널에 대한 모든 업데이트는 WSL 팀이 변경 사항을 구현 한 다음 추가하도록 기다리지 않고 즉시 컴퓨터에 추가 할 수 있습니다.
```

<br>

- 길게 적어놓았지만 결과적으로 wsl1보다 wsl2가 더 개선한 버전이다 라는 것이고 커뮤니티의 사람들 의견을 보았을 때, 개발자가 사용하기 위해서는 wsl2는 필수적이다 라는 의견 또한 있었습니다.

<br>

### **wsl 2 설치 방법**

<br>

- 설치를 하기 전에 다음 2가지 세팅은 먼저 해주시길 바랍니다. 특히 **② 은 반드시 하셔야 합니다.**
- ① **언어 설정은 영어**로 하길 매우 권장합니다. 한글로 하다가 powershell 부분에서 막힌 것도 있고해서... 영어 기준으로 블로그 내용을 따라하시길 추천드립니다.
- ② **windows 10을 업데이트** 하시기 바랍니다. command 창에서 `winver`을 입력하면 윈도우 버전으 정보가 나타나는데 이때 `OS Build`의 버전이 `18917` 보다 낮으면 `wsl 2`를 실행할 수 없습니다. 따라서 숫자가 18917 보다 낮으면 windows update가 필요합니다.

<br>
<center><img src="../assets/img/etc/dev/wsl/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 `OS Build`가 18917 보다 커야합니다.

<br>

- 다음 링크를 통하여 윈도우 10을 최신 업데이트 합니다. 홈페이지의 update now를 누르면 됩니다.
    - 링크 : https://aka.ms/downloadwindows10
- 다운 받은 파일을 실행 시켜서 윈도우를 최신 버전으로 업데이트 합니다.
- `시작` 키를 눌러서 검색 창에 `windows insider program settings` 또는 `windows 참가자 프로그램`을 입력하여 세팅 창을 실행시킵니다.
- 세팅 창에서 시작을 눌러서 계정을 등록 후 세팅을 계속 진행합니다. 마지막에 재부팅 시작이 나오면 `나중에 재부팅` 버튼을 눌러주세요.

<br>

- 그 다음 시작 키를 눌러서 검색 창에 `turn windows features on or off` 또는 `windows 기능 켜기/끄기`를 검색합니다.
- 아래 두가지를 체크를 합니다.
    - Windows Subsystem for Linux (또는 Linux용 윈도우 하위 시스템)
    - Virtual Machine Platform (또는 가상 머신 플랫폼)
- `재부팅`을 합니다.

<br>

- Microsoft store에서 ubuntu를 검색하여 설치합니다. 현재 시점으로 ubuntu 18.04 LTS 버전을 설치하였는데 향후 미래에도 가장 최신의 LTS 버전을 설치하시길 권장드립니다.
- ubuntu 설치가 완료되고 `windows insider program settings` 또는 `windows 참가자 프로그램`에서 작업 중인 것이 없으면 `재부팅`을 한번 더 하겠습니다.

<br>

- 이제 설치는 끝났습니다. 한번 실행해 보겠습니다.
- `Windows PowerShell`을 실행합니다.
- PowerShell에서 `wsl -l`을 입력하면 현재 사용중인 `wsl`의 정보가 나타납니다. 여기 까지 따라하셨다면 Default ubuntu가 설치되었다고 출력될 것입니다. 

<br>

### **wsl2 셋팅**

<br>

- `powershell`을 실행시킨 다음에 `wsl -l -v` 라는 명령어를 입력하면 다음과 같이 출력됩니다.

<br>

```
NAME            STATE           VERSION
Ubuntu-18.04    Stopped         1  
```

<br>

- 위 출력의 뜻은 현재 사용하고 있는 `wsl`의 버전이 우분투 18.04이고 버전은 1이라는 것입니다. 이제 버전을 2로 바꾸어 보겠습니다.
- `powershell`에 `wsl --set-version Ubuntu-18.04 2` 명령어를 입력합니다.
- 만약 `Conversion in progress, this may take a few minutes..` 라는 내용이 출력이 되지 않고 커널의 버전이 낮다는 내용이 출력되면 다음 링크에 들어가서 커널을 업데이트 한 다음 다시 시도해 봅니다. (이 작업은 금방 끝납니다.)보통 이 경우는 기존에 우분투를 설치한 상태에서 윈도우 업데이트를 하였다면 이전 버전의 리눅스 커널이 설치되었을 수 있습니다.
    - 링크 : https://docs.microsoft.com/en-us/windows/wsl/wsl2-kernel
- 만약 정상적으로 작업이 된다면 다음과 같은 출력이 powershell에 나타납니다.

<br>

```
Conversion in progress, this may take a few minutes..
For information on key differences with WSL 2 please visit https://aka.ms/wsl2
Conversion complete.
```

<br>

- 이제 `wsl 2`가 정상적으로 셋팅이 되었는 지 `wsl -l -v` 명령어를 입력해 보겠습니다.

<br>

```
NAME            STATE           VERSION
Ubuntu-18.04    Stopped         2  
```

<br>

- 위와 같이 VERSION이 2로 변경 되었다면 정상적으로 셋팅된 것입니다.

<br>


