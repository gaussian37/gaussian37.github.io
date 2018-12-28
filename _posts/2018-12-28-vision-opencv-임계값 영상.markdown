---
layout: post
title: 코너점(Corner) 검출 - cornerEigenValsAndVecs  
date: 2018-08-01 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, corner, 코너, cornerEigenValsAndVecs] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : 

이미지에서 행과 열의 순서로 img(y, x)에 해당하는 각각의 픽셀이 있다고 가정합시다.
포인트 프로세싱이란 입력 영상의 특정 픽셀 src(y, x)를 변환 함수를 통하여 dst(y, x) 로 만드는 것을 뜻합니다.

![1](../assets/img/vision/opencv/pointprocessing/point_processing.png)

<br>

이번 글에서는 포인트 프로세싱 중 가장 간단한 `threshold`에 대하여 알아보도록 하겠습니다.

+ 아래 두 함수 즉, 임계값 관련 함수는 영상을 분할하는 가장 간단한 방법입니다.
+ `cv2.threshold()`는 주어진 임계값에 따라 threshold image를 출력합니다.
+ `cv2.adaptiveThreshold()`는 화소 마다 다른 임계값을 적용하는 adaptive threshold 이미지를 계산합니다.

## threshold

+ ret, dst = cv2.threshold(src, thresh, max_val, type(, dst))
+ src : 1-채널의 np.uint8 또는 np.float32 입력 영상
+ dst : 1-채널의 np.uint8 또는 np.float32 입력 영상 & src와 같은 크기 영상
+ thresh : 임계값
+ type : 임계값의 종류
    + type에 cv2.THRESH_OTSU를 추가하면 thresh 값에 상관없이 Otsu 알고리즘으로 최적 임계값을 계산함
    + cv2.THRESH_BINARY
        + $$ dst(x, y) = 
            \left\{ 
            \begin{array}{c}
            max\ val\quad  if \quad  src(x, y) > thresh\\ 
            q\qquad0\cdot w
            \end{array}
            \right. 
          $$
    



