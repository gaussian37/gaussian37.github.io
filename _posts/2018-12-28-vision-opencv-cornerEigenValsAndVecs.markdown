---
layout: post
title: 코너점(Corner) 검출 - cornerEigenValsAndVecs  
date: 2018-08-01 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, corner, 코너, cornerEigenValsAndVecs] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/corner%20detection

OpenCV를 이용하여 영상에서의 코너점을 검출하는 방법에 대하여 알아보도록 하겠습니다.
`코너점`은 단일 채널의 입력 영상의 `미분 연산자에 의한 에지 방향`을 이용하여 검출 합니다.

## 코너점 검출 함수

코너점을 검출할 수 있는 OpenCV의 대표적인 6가지 방법중에 `cornerEigenValsAndVecs`에 대하여 알아보도록 하겠습니다.

코너 검출에는 다음 이미지를 사용하겠습니다.

<img src="../assets/img/vision/opencv/corner-detection/corner-test.png" alt="Drawing" style="width: 300px;"/>

사용할 함수는 `cv2.cornerEigenValsAndVecs()` 입니다.

+ dst = cv2.cornerEigenValsAndVecs(src, blockSize, ksize)
+ 코너점 검출 방법 : $$ \lambda_{1}, \lambda_{2} $$ 
+ 상세 내용
    + 입력 영상 src에서 각 픽셀의 고유값과 고유벡터를 6 채널 dst에 계산합니다.
    + 영상의 모든 픽셀에 대하여 blockSize x blockSize의 이웃에 있는 미분값을 이용하여
    2 x 2 크기의 gradient를 이용한 covariance matrix M을 계산하고, M의 eigenvalue $$ \lambda_{1}, \lambda_{2} $$,
    eigenvector $$ (x_{1}, y_{1})과 (x_{2}, y_{2}) $$를 계산하여 dst에 저장합니다.
        + eigenvalue $$ \lambda_{1}, \lambda_{2} $$ 가 `모두 작은 곳`은 **평평**한 영역에 있는 점
        + eigenvalue $$ \lambda_{1}, \lambda_{2} $$ 둘 중 하나는 크고 하나는 작으면 **에지**
        +  eigenvalue $$ \lambda_{1}, \lambda_{2} $$ 두 값이 `모드 큰` 경우 **코너점**
        
Covariant Matrix X 은 다음과 같습니다.        

$$ M = \left[
\begin{array}{cc}
  \sum_{Nbd(x,y)}I_{x}^{2}&\sum_{Nbd(x,y)}I_{x}I_{y}\\
  \sum_{Nbd(x,y)}I_{y}I_{x}&\sum_{Nbd(x,y)}I_{y}^{2}
\end{array}
\right] $$
        
이 때, $$ I_{x} = \frac{\partial I(x,y)}{\partial x}, I_{y} = \frac{\partial I(x,y)}{\partial y} $$ 입니다.

