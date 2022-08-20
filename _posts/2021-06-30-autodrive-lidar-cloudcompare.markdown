---
layout: post
title: 포인트 클라우드 처리를 위한 CloudCompare 사용법 정리
date: 2021-06-30 00:00:00
img: autodrive/lidar/cloudcompare/0.png
categories: [autodrive-lidar] 
tags: [라이다, 클라우드 컴패어, CloudCompare] # add tag
---

<br>

- CloudCompare 매뉴얼 : https://www.danielgm.net/cc/doc/qCC/Documentation_CloudCompare_version_2_1_eng.pdf

<br>

- 이번 글에서는 라이다 포인트를 읽고 처리하는 툴인 `CloudCompare`에 대하여 다루어 보도록 하겠습니다.
- `CloudCompare`는 코드를 다룰 필요가 없고 단순히 툴을 다루면 되기 때문에 필요한 기능들을 위주로 본 글에서 다루도록 하겠습니다.

<br>

## **목차**

<br>

- ### 설치 방법
- ### 입력 파일 형식
- ### 자주 사용하는 기능

<br>

## **설치 방법**

<br>

- CloudCompare는 아래 링크에서 운영 체제에 맞게 다운 받으면 됩니다.
    - 링크 : http://www.danielgm.net/cc/release/
- 윈도우와 macOS의 경우 설치 파일을 받으면 되고 리눅스의 경우 간단하게 명령어로 설치 가능합니다.
    - `snap install cloudcompare`
- 오프라인 환경에서는 git에서 source를 받아서 사용하면 됩니다. 단, 아래 명령어를 통하여 `recursive`로 받아야 필요한 파일을 모두 받을 수 있습니다.
    - git source : https://github.com/CloudCompare/CloudCompare
    - `git clone --recursive https://github.com/cloudcompare/CloudCompare.git`
- git source를 이용하여 설치하는 방법은 repository의 `BUILD.md` 파일에 자세하게 나와있습니다.
    - https://github.com/CloudCompare/CloudCompare/blob/master/BUILD.md
- git source를 받아서 설치하는 방법은 요약하면 다음과 같습니다. 리눅스 기준으로 설명 드리겠습니다.
    - ① 리눅스 패키지 설치 : `sudo apt install libqt5svg5-dev libqt5opengl5-dev qt5-default qttools5-dev qttools5-dev-tools libqt5websockets5-dev`
    - ② git source 다운로드 : `git clone --recursive https://github.com/cloudcompare/CloudCompare.git`
    - ③ cmake configuration : 
        - `mkdir build && cd build`
        - `cmake ..`
    - ④ build : `cmake --build .` (build 폴더에서 입력) 
    - ⑤ install : `cmake --install .` (build 폴더에서 입력)
- 여기까지 에러 없이 정상 동작 하였다면 설치가 완료되었습니다. `build` 내부의 `qCC` 폴더에 `CloudCompare`라는 실행파일이 있습니다. 이 파일을 실행하면 CloudCompare가 실행 됩니다.

<br>

## **입력 파일 형식**

<br>

- 참조 : https://www.cloudcompare.org/doc/wiki/index.php/FILE_I/O
- 위 링크를 보면 CloudCompare에서 사용하는 입력 포맷을 확인할 수 있습니다. 입력 가능한 포맷의 종류는 굉장히 많이 있으며 위 링크에서 포맷을 꼭 참조하시기 바랍니다.

<br>
<center><img src="../assets/img/autodrive/lidar/cloudcompare/1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 링크의 내용을 참조하면 가장 마지막 열에 `Features`를 통해 필요로 하는 성분을 확인하고 사용하시면 됩니다.
- 본 글에서는 `ASCII` 타입을 사용하도록 하겠습니다. `ASCII` 타입은 용량을 제외하면 큰 단점이 없고 좌표 정보 이외에 `Normals, colors (RGB), scalar fields (all)`와 같은 다양한 정보를 사용할 수 있고 편집기로 바로 읽고 쓸 수 있어서 편리하다는 장점이 있습니다. `scalar fiedls`는 `Intensity, time, confidence, temperature, etc.`와 같은 다양한 형태를 지원합니다. 
- 본 글에서는 `X Y Z Red Green Blue`에 대한 포인트 클라우드의 정보를 행 별로 저장한 것을 기준으로 글을 작성하겠습니다. `ASCII` 양식은 다음과 같습니다.
    - `X Y Z R G B`
    - Ex) `120.32 10.23 56.12 100 100 20`
    - 단위는 m 이며 소숫점 2자리 까지 반영하여 1cm로 구분가능하도록 정밀도를 구성하였습니다. RGB 각각의 값은 0 ~ 255의 범위를 가집니다. 이와 같이 구성하면 각 포인트에 대하여 위치 정보와 색 정보를 CloudCompare에서 표현할 수 있어서 보기 편합니다.