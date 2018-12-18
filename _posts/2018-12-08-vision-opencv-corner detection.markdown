---
layout: post
title: (OpenCV-Python) 코너점(Corner) 검출  
date: 2018-08-01 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, corner, 코너,] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/corner%20detection

** 오늘 하루도 즐거운 하루 되길 바라며, 도움이 되셨다면 광고 한번 클릭 부탁 드립니다. 꾸벅 ^^ **

OpenCV를 이용하여 영상에서의 코너점을 검출하는 방법에 대하여 알아보도록 하겠습니다.
`코너점`은 단일 채널의 입력 영상의 `미분 연산자에 의한 에지 방향`을 이용하여 검출 합니다.

## 코너점 검출 함수

코너점을 검출할 수 있는 OpenCV 함수는 크게 6가지가 있습니다. 각각에 대하여 알아보고 어떻게 구현하며 되는지 알아보겠습니다.

코너 검출에는 다음 이미지를 사용하겠습니다.

![test](../assets/img/vision/opencv/corner-detection/corner-test.png)

+ dst = cv2.preCornerDetect(src, ksize)
    + 코드 : [링크](https://github.com/gaussian37/Vision/blob/master/OpenCV/corner%20detection/preCornerDetect.py)
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
        
        
```python
import argparse
import cv2
import numpy as np

# findLocalMaxima()는 src에서 Morphology 연산(팽창 & 침식)으로 
# Local maxia의 좌표를 points 배열에 검출하여 반환함
def findLocalMaxima(src):
    kernel= cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = (11,11))
    
    # cv2.dilate()로 src에서 rectKernel의 이웃에서 최댓값을 dilate에 계산
    # kernel = None을 사용하면 기본적으로 3 x 3 커널이 사용됩니다.
    dilate = cv2.dilate(src, kernel)
    
    # src == dilate 연산으로 src에서 local maxima 위치를 localMax 배열에 계산합니다.
    localMax = (src == dilate)

    # cv2.erode()로 src에서 rectKernel의 이웃에서 최솟값을 erode에 계산합니다.
    erode = cv2.erode(src, kernel) # local min if kernel = None, 3x3
    
    # src > erode로 최솟값보다 큰 위치를 localMax2에 계산합니다.
    localMax2 = src > erode 
    
    # & 연산으로 local maxima 위치를 계산
    localMax &= localMax2
    
    # 행,열 순서로 localMax 위치를 찾습니다.
    points = np.argwhere(localMax == True)
    
    # 좌표 순서를 열(x), 행(y)으로 변경 합니다.
    points[:,[0,1]] = points[:,[1,0]] # switch x, y 
    return points

# argument parser를 구성해 주고 입력 받은 argument는 parse 합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

src = cv2.imread(args["image"])

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# grayscale의 gray에서 cv2.preCornerDetect()로 res를 계산합니다.
res = cv2.preCornerDetect(gray, ksize = 3)

# 극대값만을 찾기 위해서 np.abs(res)인 절대값 배열에서
# cv2.threshold()로 임계값 thresh = 0.1 보다 작은 값은 0으로 변경하여 res2에 저장합니다.
# res에서 임계값보다 작은 값을 제거 합니다.
# findLocalMaxima() 함수를 통해 res2에서 
# 지역 극값의 좌표를 코너점으로 찾아 corners에 저장합니다.
ret, res2 = cv2.threshold(np.abs(res), 0.1, 0, cv2.THRESH_TOZERO)

# corner를 저장합니다.
corners = findLocalMaxima(res2) 
print('corners.shape=', corners.shape)

# src를 dst에 복사하고, 코너점 배열 corners의 각 코너점 좌표에 cv2.circle()로
# dst에 반지름 5, 빨간색 원을 그립니다.
dst = src.copy()
for x, y in corners:
    cv2.circle(dst, (x, y), 5,(0, 0, 255), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
```
        
... 작성중 ...    
        
                    
