---
layout: post
title: 리눅스에서 이클립스로 opencv 사용하기
date: 2020-03-08 00:00:00
img: vision/opencv/linux_eclipse_c_opencv/0.png
categories: [vision-opencv] 
tags: [리눅스, 이클립스, opencv] # add tag
---

<br>

- 참조1 : https://agiantmind.tistory.com/182
- 참조2 : https://webnautes.tistory.com/1186
- 이번 글에서는 리눅스에서 이클립스를 이용하여 OpenCV C++ 버전을 사용하는 방법에 대하여 정리하겠습니다.

<br>

## **이클립스 설치**

<br>

- 먼저 이클립스를 설치하도록 하겠습니다. 다른 IDE도 많이 있지만 이클립스 만의 장점이 있으니 사용해 보시기를 추천 드립니다.
- 이클립스를 이용하면 환경 세팅하는 것이 편하고 디버깅 모드도 쉽게 동작시킬 수 있습니다.
- 먼저 이클립스는 자바 환경에서 동작하기 때문에 `JRE`(Java Runtime Environment) 또는 `JDK`(Java Development Kit)을 설치하여 자바 환경이 동작할 수 있도록 만들어 주어야 합니다.
- 이 글에서는 `JRE`를 설치해 보도록 하겠습니다.

<br>

`sudo apt install default-jre`

<br>

- 그 다음으로 C/C++ 개발 관련 패키지 설치를 해보겠습니다. 다음 명령어를 통하여 설치를 완료합니다.

<br>

`sudo apt-get install build-essential`

<br>

- 이제 이클립스가 구동될 수 있는 환경이 만들어 졌으므로 이클립스를 설치해 보도록 하겠습니다.
- 이클립스 버전 중에 `CDT`라는 버전을 설치해야 C/C++이 지원이 됩니다. (Eclipse IDE for C/C++ Developers) 아래 링크를 이용해서 최신 버전을 깔아보시기 바랍니다.
    - https://www.eclipse.org/downloads/packages/release/2019-12/r/eclipse-ide-cc-developers?
- 설치할 때에는 설치할 리눅스 버전에 맞게 설치하면 됩니다. 제가 사용하는 리눅스 버전은 64bit linux이므로 64bit linux용 이클립스를 설치하면 됩니다.
- 받은 파일의 형태가 `tar.gz` 이면 `tar -xvzf 압축파일`을 이용하여 압축을 해제하면 됩니다.
- 압축을 풀고 나면 `eclipse`라는 폴더가 생기게 됩니다. 이 폴더를 root 아래의 /opt로 이동시키겠습니다.

<br>

`sudo mv eclipse /opt`

<br>

- 터미널 실행 설정에 대한 세팅을 입력 합니다.
- CUI 환경에서 입력하려면 다음 명령어를 입력합니다.
    - `sudo vi /usr/bin/eclipse`
- GUI 환경에서 입력하려면 다음 명령어를 입력합니다.
    - `sudo gedit /usr/bin/eclipse`

<br>

- 열린 vi 또는 gedit에서 다음을 입력합니다.

<br>

```
#! /bin/sh
export ECLIPSE_HOME=/opt/eclipse
$ECLIPSE_HOME/eclipse $*
```

<br>

- 바로가기 설정에 대한 권한 설정을 해줍니다.

<br>

`sudo chmod 755 /usr/bin/eclipse`

<br>

- 만약 X윈도우에서 바로가기 설정이 필요하다면 아래처럼 입력해 주면 됩니다.
- vi 사용 : `sudo vi /usr/share/applications/eclipse.desktop`
- gedit 사용 : `sudo gedit /usr/share/applications/eclipse.desktop`

<br>

```
[Desktop Entry]

Encoding=UTF-8
Name=Eclipse
Comment=Eclipse IDE
Exec=eclipse
Icon=/opt/eclipse/icon.xpm
Terminal=false
Type=Application
Categories=Development
StartupNotif=true
```

<br>

- 이제 기본적인 이클립스 세팅은 끝났습니다. 이제 이클립스를 한번 실행해 보도록 하겠습니다.
- 이클립스를 실행하면 CDT 관련 설치 목록이 뜰 수도 있고 아닐 수도 있습니다.
- 만약 CDT 관련 설치 목록이 생긴다면 관련 목록을 설치해 주면 됩니다. 만약 따로 설치 목록이 뜨지 않는다면 다음 경로로 들어가서 설치해 보도록 하겠습니다.
    - 이클립스 우측 상단의 `help - Install New Software`를 클릭합니다.
    - Work with 란에 `CDT`라고 입력하면 `CDT Main Features`와 `CDT Optional Features`가 뜹니다. 
    - 이 두 항목을 클릭한 다음에 Next를 눌러서 설치를 진행합니다.
- 이제 준비는 끝났습니다. C 또는 C++ 프로젝트를 생성해서 코드를 돌려봐서 정상 설치 되었는 지 확인하면 됩니다.
- 프로젝트를 생성할 때, `Toolchains`로 선택해야 하는 항목이 있는데 `Linux GCC`를 사용하면 됩니다.
- 프로젝트에서 왼쪽 상단의 망치 모양이 build입니다. build 후 run 하면 코드를 실행할 수 있습니다.

<br>
<center><img src="../assets/img/vision/opencv/linux_eclipse_c_opencv/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 프로젝트에서 위 그림처럼 stop 심볼 오른쪽에 `Run`과 `Debug`가 정상적으로 세팅이 되어 있는지 확인 후 build 및 run을 하시기 바랍니다.
- 정상적으로 코드가 실행되었다면 이제 OpenCV를 설치하도록 해보겠습니다.

<br>

## **OpenCV 설치**

<br>

- OpenCV를 설치 하기 전에 현재 컴퓨터에 OpenCV가 설치되어 있는 지 먼저 확인해 보도록 하겠습니다.
- 기존에 설치되어 있는 OpenCV가 있으면 새로 설치하는 버전이 정상적으로 동작하지 않을 수 있기 때문입니다.
- 다음 명령어를 입력하여 설치 되어 있는 OpenCV 버전을 확인해 보시기 바랍니다.

<br>

`pkg-config --modversion opencv`

<br>

- 만약 커맨드 결과로 특정 버전이 나온다면 기존에 특정 버전이 설치된 것으로 보면 됩니다.
    - 예를 들어 `3.4.2` 처럼 설치된 버전이 출력되게 됩니다.
- 반면 설치되어 있는 버전이 없으면 다음의 출력이 터미널에 나오게 됩니다.

<br>

```
Package opencv was not found in the pkg-config search path.
Perhaps you should add the directory containing `opencv.pc'
to the PKG_CONFIG_PATH environment variable
No package 'opencv' found
```

<br>

- 만약 현재 설치된 opencv 파일 및 설정등을 삭제하려면 다음 명령어를 차례대로 입력 합니다.

<br>

- `sudo apt-get purge  libopencv* python-opencv`
- `sudo apt-get autoremove`
- `sudo find /usr/local/ -name "*opencv*" -exec rm -i {} \;`
 
<br>

- 이제 초기화된 상태에서 OpenCV를 설치해 보도록 하겠습니다.
- 먼저 기존에 설치된 패키지들의 새로운 버전이 저장소에 있다면 리스트를 업데이트 하기 위해 다음 명령어를 입력합니다.
    - `sudo apt-get update`

<br>

- 기존에 설치된 패키지의 새로운 버전이 있으면 업그레이드를 진행합니다.
    - `sudo apt-get upgrade`

<br>

- 지금부터 설치하는 것들은 OpenCV를 컴파일 하기 전에 필요한 패키지를 설치하는 단계입니다.
- 패키지 내용을 하나하나 살펴보려면 아래 글을 쭉 살펴보면서 설치하시고 그렇지 않고 그냥 한번에 설치하려면 다음 커맨드를 입력하시기 바랍니다.

<br>

`sudo apt-get install cmake pkg-config libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev libv4l-dev v4l-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk2.0-dev mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev libatlas-base-dev gfortran libeigen3-dev python2.7-dev python3-dev python-numpy python3-numpy`

<br>

- 먼저 `cmake`를 설치하도록 하겠습니다. `cmake`는 컴파일 옵션이나 빌드된 라이브러리에 포함시킬 OpenCV 모듈 설정등을 위해 필요합니다. 
    - `sudo apt-get install cmake`

<br>

- 그 다음 `pkg-config`를 설치하도록 하겠습니다.
- pkg-config는 프로그램 컴파일 및 링크시 필요한 라이브러리에 대한 정보를 메타파일(확장자가 .pc 인 파일)로부터 가져오는데 사용됩니다. 터미널에서 특정 라이브러리를 사용한 소스코드를 컴파일시 필요한 컴파일러 및 링커 플래그를 추가하는데 도움이 됩니다.
    - `sudo apt-get install pkg-config`

<br>

- 다음은 특정 포맷의 이미지 파일을 불러오거나 기록하기 위해 필요한 패키지들입니다.
    - `sudo apt-get install libjpeg-dev libtiff5-dev libpng-dev`

<br>

- 다음은 특정 코덱의 비디오 파일을 읽어오거나 기록하기 위해 필요한 패키지들입니다.
    - `sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev`

<br>

- Video4Linux 패키지는 리눅스에서 비디오 캡처를 지원하기 위한 디바이스 드라이버와 API를 포함하고 있습니다. 
    - `sudo apt-get install libv4l-dev v4l-utils`

<br>

- GStreamer는 비디오 스트리밍을 위한 라이브러리입니다. 
    - ` sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev`

<br>

- OpenCV에서는 highgui 모듈을 사용하여 자체적으로 윈도우 생성하여 이미지나 비디오를 보여줄 수 있습니다. 윈도우 생성 등의 GUI를 위해 gtk 또는 qt를 선택해서 사용가능합니다. 여기서는 gtk2를 지정해주었습니다.
    - `sudo apt-get install libgtk2.0-dev`

<br>

- OpenGL 지원하기 위해 필요한 라이브러리입니다.
    - `sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev`

<br>

- OpenCV 최적화를 위해 사용되는 라이브러리들입니다.
    - `sudo apt-get install libatlas-base-dev gfortran libeigen3-dev`

<br>

- python2.7-dev와 python3-dev 패키지는 OpenCV-Python 바인딩을 위해 필요한 패키지들입니다. Numpy는 매트릭스 연산등을 빠르게 처리할 수 있어서 OpenCV에서 사용됩니다. 
    - `sudo apt-get install python2.7-dev python3-dev python-numpy python3-numpy`

<br>
<br>

- 이제 OpenCV 설정과 컴파일 및 설치하는 방법에 대하여 알아보도록 하겠습니다.
- 먼저 OpenCV 소스 코드를 저장할 임시 디렉토리를 생성하겠습니다.
    - `mkdir opencv`
    - `cd opencv`
- opencv 폴더에 opencv 소스 코드를 다운 받아서 저장하고 압축을 풉니다. (버전은 선택)
    - 버전 4 예시 : `wget -O opencv.zip https://github.com/opencv/opencv/archive/4.0.1.zip`
    - 버전 3 예시 : `wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.2.zip`
    - `unzip opencv.zip`
- opencv_contrib(extra modules) 소스코드를 다운로드 받아 압축을 풀어줍니다. (버전은 선택)
    - 버전 4 예시 : `wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.0.1.zip`
    - 버전 3 예시 : `wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.2.zip`
    - `unzip opencv_contrib.zip`
- 앞으로 예제는 버전 4를 예시로 진행하겠습니다. 그러면 다음 두 개의 디렉토리가 생성됩니다.
    - opencv 폴더 안에서 `ls -d */`를 실행하면 `opencv-4.0.1/  opencv_contrib-4.0.1/`가 있는 것을 확인할 수 있습니다.
- 그 다음 opencv-4.0.1 디렉토리로 이동하여 build 디렉토리를 생성하고 build 디렉토리로 이동합니다. 컴파일은 build 디렉토리에서 이루어집니다.
    - `cd opencv-4.0.1/`
    - `mkdir build`
    - `cd build`

<br>

- build 디렉토리 안에서 cmake를 사용하여 `OpenCV 컴파일 설정`을 해줍니다.

<br>

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=OFF \
-D WITH_IPP=OFF \
-D WITH_1394=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_DOCS=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D WITH_QT=OFF \
-D WITH_GTK=ON \
-D WITH_OPENGL=ON \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.0.1/modules \
-D WITH_V4L=ON  \
-D WITH_FFMPEG=ON \
-D WITH_XINE=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON ../
```
    
<br>

- 만약 파이썬에 대한 경로가 없으면 아래 명령어도 추가하시면 됩니다. 단, 아래 명령어는 실제 컴퓨터에서의 경로와 일치하는 것이 있는 지 확인하시기 바랍니다.

<br>

```
-D PYTHON2_INCLUDE_DIR=/usr/include/python2.7 \
-D PYTHON2_NUMPY_INCLUDE_DIRS=/usr/lib/python2.7/dist-packages/numpy/core/include/ \
-D PYTHON2_PACKAGES_PATH=/usr/lib/python2.7/dist-packages \
-D PYTHON2_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so \
-D PYTHON3_INCLUDE_DIR=/usr/include/python3.6m \
-D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/  \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
```

<br>

- 위 cmake 명령어가 성공적으로 동작하면 다음 출력을 볼 수 있습니다.

<br>

```
-- Configuring done
-- Generating done
-- Build files have been written to: /home/webnautes/opencv/opencv-4.0.1/build
```

<br>

- 앞에서 cmake를 이용하여 컴파일 설정을 완료하였으면 컴파일을 해야합니다.
- 컴파일을 시작하기 전에 사용 중인 컴퓨터의 CPU 코어수를 확인합니다.
    - `cat /proc/cpuinfo | grep processor | wc -l`
    - 결과로 4가 나오면 사용중인 CPU의 코어수가 4입니다.
- **opencv/opencv-4.0.1/build** 경로에서 `make` 명령을 사용하여 컴파일을 시작합니다. -j 다음에 위에서 확인한 숫자를 붙여서 실행해줍니다. 앞에 time을 붙여서 실행하면 컴파일 완료 후 걸리는 경과를 알려줍니다.
    -`time make -j4`
- 이제 컴파일을 완료하였으므로 **컴파일 결과물을 설치**합니다.
    - `sudo make install`
- 그 다음 `cat /etc/ld.so.conf.d/*` 명령어를 터미널에서 입력하여 출력물에서 **/usr/local/lib** 를 포함하는 설정 파일이 있는 지 확인합니다.
- **/usr/local/lib**이 출력되지 않았다면 다음 명령을 추가로 실행합니다.
    - `sudo sh -c 'echo '/usr/local/lib' > `
- 마지막으로 컴파일 시 opencv 라이브러리를 찾을 수 있도록 다음 명령을 실행합니다.
    - `sudo ldconfig`

<br>

## **OpenCV 설치 결과 확인**

<br>

- 이클립스에서 실제로 코드를 돌려보기 이전에 코드가 잘 설치되었는 지 확인이 필요합니다.
- 다음 명령어를 이용하여 예제를 컴파일 해보도록 하겠습니다.
    - 아래 명령어를 그대로 실행하지 말고 실제 경로에 맞추셔야 합니다. 예를 들어 **/usr/local/share/opencv4/**에서 opencv4가 아니라 버전에 따라 다른 이름이 있을 수도 있습니다.
    - 두번째로 `pkg-config`에서의 opencv4는 버전에 맞게 지정해주시면 됩니다. opencv-4.x.x 이면 opencv4를 하면 되고 opencv-3.x.x이면 opencv3을 하면 됩니다.
    - `g++ -o facedetect /usr/local/share/opencv4/samples/cpp/facedetect.cpp $(pkg-config opencv4 --libs --cflags)`
- 컴파일이 완료 되었으면 다음 코드를 실행해 보시기 바랍니다. 컴파일 할 때와 동일하게 경로 지정을 조금 변경하신 후 실행해 보시기 바랍니다.
    - `./facedetect --cascade="/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml" --nested-cascade="/usr/local/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml" --scale=1.3`
- 마지막으로 컴파일에 사용하였던 opencv 소스코드 디렉토리를 삭제하셔도 됩니다.

<br>

## 이클리스에서 OpenCV 연동하기

<br>

- 앞에서 이클립스도 설치하였고 OpenCV도 설치하였으므로 이제 이클립스와 OpenCV를 연동해 보도록 하겠습니다.
- 이클립스에서 임의의 프로젝트를 열고 `Project Explorer`를 오른쪽 클릭하여 `Properties`를 들어가서
- `C/C++ Build`의 `Settings`를 클릭합니다. 그 다음 `Tool Settings` 탭을 보겠습니다.
- 탭 안에 보면 크게 `GCC C++ Compiler`, `GCC C Compiler` 그리고 `GCC C++ Linker`가 있습니다.
- 먼저 `GCC C++ Compiler` 부분을 살펴보도록 하겠습니다.
    - `Dialect` → `Language standard` → `ISO C++ 11`
    - `Include` → `Include paths`에 opencv가 설치 된 폴더의 위치를 입력 합니다. 보통 `/usr/local/include/`가 됩니다.
    - 원활한 동작을 위하여 `Optimization` → `optimization level` → `-o2`로 입력합니다.
- 다음으로 `GCC CCompiler` 부분도 C++과 유사하게 설정해 주면 됩니다.
    - `Dialect` → `Language standard` → `ISO C99`
    - `Include` → `Include paths`에 opencv가 설치 된 폴더의 위치를 입력 합니다. 보통 `/usr/local/include/`가 됩니다.
    - 원활한 동작을 위하여 `Optimization` → `optimization level` → `-o2`로 입력합니다.
- 마지막으로 `GCC C++ Linker`를 추가해 주겠습니다. `Libraries` → `Libraries`에 아래 내용들을 추가하면 됩니다.
    - opencv_core
    - opencv_imgcodecs
    - opencv_highgui
    - opencv_imgproc 