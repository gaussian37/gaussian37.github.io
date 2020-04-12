---
layout: post
title: visual studio에서 opencv 세팅하는 방법
date: 2020-03-29 00:00:00
img: vision/opencv/opencv.png
categories: [vision-opencv] 
tags: [opencv, window, visual studio] # add tag
---

<br>

- 윈도우의 visual studio에서 opencv를 세팅하는 방법은 다른 OS의 다른 IDE에서 사용하는 것 보다 상당히 쉽습니다.
- 5단계의 Step만 차례대로 적용하면 아주 쉽게 사용할 수 있습니다.

<br>

## **① opencv 윈도우 버전 다운 받기**

<br>

- 첫번째 단계는 사용할 opencv 설치 파일을 받는 것입니다. 이 때, 원하는 버전을 사전에 확인한 후 찾아서 받아야 합니다.
    - 링크 : https://opencv.org/releases/
- 링크에 접속하여 다운 받을 버전의 `Windows` 버튼을 눌려서 설치 파일(exe)을 받습니다.
- 이 때 확인해야 할 것은 설치 파일의 끝부분에 어떤 visual studio와 호환이 되는 지 보여주는 것이 있는데 이 부분을 확인해야 합니다.
    - 예를 들어 visual studio 2015/2017은 vc15와 호환됩니다.
- 설치 파일을 원하는 경로 (아무 경로나 상관없음)에 설치합니다. 이 글에서는 편의상 **C:\opencv**에 설치되었다고 가정하겠습니다.    
    
<br>

## **② 환경 변수 등록**

<br>

- 그 다음으로 환경 변수 등록을 해야 합니다.
- 환경 변수 등록은 어떤 경로에서라도 실행 파일에 접근 하여 실행 할 수 있도록 세팅하는 작업입니다.
- 윈도우 검색에 `고급 시스템 설정 보기`를 검색하여 실행한 후 오른쪽 하단의 `환경 변수`를 클릭합니다.
- `시스템 변수` 항목에서 `Path`를 더블 클릭합니다.
- 윈도우 10에서는 새로 만들기를 클릭하고, 윈도우 7에서는 마지막 칸에 `;`을 입력한 다음, 사용하는 visual studio의 설정에 맞게 경로를 셋팅해줍니다.
    - 아래 셋팅은 `x64`를 사용하요 visual studio 2017을 사용할 때의 세팅 예시입니다.
        - `C:\opencv\build\x64\vc15\bin`
    - 위 예시에 해당하는 경로를 직접 찾아 들어가면 `exe` 파일 및 `dll` 파일이 있으며 visual studio에서 opencv관련 코드를 수행 시 사용되게 됩니다.
- 환경 변수 등록을 완료하였다면 **컴퓨터를 재부팅** 하여 실제로 변수가 등록 되도록 해줍니다.

<br>

## **③ C/C++ - Additional Include Directories**

<br>

- visual studio 프로젝트를 생성한 다음 프로젝트의 속성(property) 창을 엽니다.
- `구성 속성(Configuration Properties)` → `C/C++` → `일반(General)` → `추가 포함 디렉토리(Additional Include Directories)`에 다음과 같이 입력해 줍니다.
    - `C:\opencv\build\include` (이 경로는 작업 환경에 따라 다를 수 있으며 참조만 하시기 바랍니다.)
- 이 셋팅은 헤더 파일들을 모아 둔 경로를 설정해주는 작업입니다. 위 경로를 찾아 들어가면 코드에서 사용할 헤더 파일(.h)들이 있습니다.

<br>

## **④ Linker - Additional Library Directories**

<br>

- `구성 속성(Configuration Properties)` → `링커(Linker)` → `일반(General)` → `추가 라이브러리 디렉토리(Additional Library Directories)`에 다음과 같이 입력합니다.
    - `C:\opencv\build\x64\vc15\lib` (이 경로는 작업 환경에 따라 다를 수 있으며 참조만 하시기 바랍니다.)
- 이 셋팅은 opencv에서 사용할 라이브러리 경로를 지정해 주는 작업입니다. 단지 여기에서는 폴더의 경로를 지정한 것이고 ⑤ 작업에서 실제 사용할 라이브러리 파일을 입력해 줍니다.

<br>

## **⑤ Linker - Additional Dependencies**

<br>

- `구성 속성(Configuration Properties)` → `링커(Linker)` → `입력(Input)` → `추가 종속성(Additional Dependencies)`에 다음과 같이 입력합니다.
    - `opencv_world347.lib`
    - `opencv_world347d.lib`
- 위 예제는 opencv의 3.4.7 버전을 받았을 경우 파일 명이 위와 같으면 끝에 d가 붙은 것은 디버깅 모드, d가 없는 것은 release 모드에 해당합니다.
- 참고로 opencv 2.X.X 버전에서는 필요한 라이브러리들의 목록이 다 펼쳐져 있어서 하나 하나 입력해 주어야 했지만 3버전에서는 opencv_world 라는 라이브러리 하나만 입력하면 되도록 바뀌어서 상당히 편해졌습니다.

<br>

- 여기 까지가 끝입니다. 예제 코드를 이용하여 코드가 정상적으로 동작하는 지 확인해 보시면 됩니다. 
- 아래 코드는 test.png 이미지를 입력으로 받아서 show 하는 코드입니다.

<br>

```c
#include <stdio.h>
#include "opencv/highgui.h"

int main() {

	IplImage* img = cvLoadImage("test.png");
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);
	cvShowImage("test", img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow("test");
}
```