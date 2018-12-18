---
layout: post
title: (OpenCV-Python) 코너점(Corner) 검출  
date: 2018-03-01 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, corner, 코너,] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/crop%20image

OpenCV를 이용하여 영상에서의 코너점을 검출하는 방법에 대하여 알아보도록 하겠습니다.
`코너점`은 단일 채널의 입력 영상의 `미분 연산자에 의한 에지 방향`을 이용하여 검출 합니다.

## 코너점 검출 함수

+ dst = cv2.preCornerDetect(src, ksize)
    + 코너 검출 방법 : local maxima, minima
    + 영상 src에서 코너점 검출을 위한 특징맵 dst를 Sobel 미분 연산자를 이용하여 계산
    + ksize는 Sobel 연산자의 마스크 크기
    + 코너점은 dst에서 local maxima/minima 에서 검출
    + 이 때, $$ dst(x, y) = I^{2}_{x}I_{yy} + I^{2}_{y}I_{xx} - 2I_{x}I_{y}I_{xy} $$
        +  $$ I_{x} = \frac{\partial I(x,y)}{\partial x} $$
        + dd            
