---
layout: post
title: (OpenCV-Python) 코너점(Corner) 검출  
date: 2018-08-01 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, corner, 코너,] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/crop%20image

** 오늘 하루도 즐거운 하루 되길 바라며, 도움이 되셨다면 광고 한번 클릭 부탁 드립니다. 꾸벅 ^^ **

OpenCV를 이용하여 영상에서의 코너점을 검출하는 방법에 대하여 알아보도록 하겠습니다.
`코너점`은 단일 채널의 입력 영상의 `미분 연산자에 의한 에지 방향`을 이용하여 검출 합니다.

## 코너점 검출 함수

코너점을 검출할 수 있는 OpenCV 함수는 크게 6가지가 있습니다. 각각에 대하여 알아보고 어떻게 구현하며 되는지 알아보겠습니다.

코너 검출에는 다음 이미지를 사용하겠습니다.

![test](../assets/img/vision/opencv/corner-detection/corner-test.png)

+ dst = cv2.preCornerDetect(src, ksize)
    + 코너 검출 방법 : local maxima, minima
    + 영상 src에서 코너점 검출을 위한 특징맵 dst를 Sobel 미분 연산자를 이용하여 계산
    + ksize는 Sobel 연산자의 마스크 크기
    + 코너점은 dst에서 local maxima/minima 에서 검출
    + 이 때, $$ dst(x, y) = I^{2}_{x}I_{yy} + I^{2}_{y}I_{xx} - 2I_{x}I_{y}I_{xy} $$
        + 1) $$ I_{x} = \frac{\partial I(x,y)}{\partial x} $$
        + 2) $$ I_{y} = \frac{\partial I(x,y)}{\partial y} $$
        + 3) $$ I_{xx} = \frac{\partial^{2} I(x,y)}{\partial^{2} x} $$
        + 4) $$ I_{yy} = \frac{\partial^{2} I(x,y)}{\partial^{2} x} $$
        + 5) $$ I_{xy} = \frac{\partial^{2} I(x,y)}{\partial x \partial y} $$
        
        
... 작성중 ...    
        
                    
