---
layout: post
title: 히스토그램 계산 - calcHist  
date: 2018-12-28 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, corner, 임계값, threshold, adaptiveThreshold] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/

+ 히스토그램은 관찰 데이터의 `빈도수`를 막대그래프로 표시한 것
    + 히스토그램을 이용하면 `데이터의 확률분포`를 추정할 수 있습니다.
+ 히스토그램은 이미지의 픽셀의 분포에 대하여 매우 중요한 정보를 제공합니다.
+ 이미지의 픽셀 분포를 이용하여 즉, 히스토그램을 이용하여 아래 작업을 할 수 있습니다.
    + 화질 개선 with 히스토그램 이퀄라이제이션
    + 히스토그램 비교
    + 히스토그램 역투영 
    
+ opencv에서 제공하는 히스토그램 함수는 opencv 내의 모든 함수 중에서도 좀 복잡한 편에 속합니다. 파라미터가 많기 때문입니다.

```python
hist = cv2.calcHist(images, channels, mask, histSize, ranges(, hist(, accumulate)))
```

<br>

아래는 calcHist의 파라미터 입니다. `배열`이라고 표시한 부분은 반드시 `리스트`로 입력해야 합니다.

+ images : 히스토그램을 계산할 영상의 `배열`입니다. 영상은 같은 사이트, 깊이의 8bit unsigned 정수 또는 32bit 실수형 입니다.
+ channels : 히스토그램을 계산할 channel의 `배열`. (**배열 형태로 입력 필요함**)
+ mask : images\[i\]와 같은 크기의 8bit 이미지로, mask(x, y)가 0이 아닌 경우에만 image\[i\](x,y)을 히스토그램 계산에 사용합니다.
    + mask = None이면 마스크를 사용하지 않고, 모든 화소에서 히스토그램을 계산합니다.
+ histSize : 히스토그램 hist (return 값)의 각 빈(bin) 크기에 대한 정수 `배열` 입니다.
+ ranges : 히스토그램 각 빈의 경계값에 대한 `배열`입니다. opencv는 기본적으로 등간격 히스토그램을 제공합니다.
+ accumulate : True 이면 calcHist() 함수를 수행할 때, 히스토그램을 초기화 하지 않고, 이전 값을 계속 누적합니다.
+ hist : 히스토그램 리턴값

```python
import cv2

import numpy as np

src = np.array([ [0, 0, 0, 0],
                 [1, 2, 3, 4],
                 [3, 4, 4, 1],
                 [2, 5, 5, 3],
               ], dtype = np.uint8)

hist1 = cv2.calcHist(images = [src], 
                     channels = [0], 
                     mask = None, 
                     histSize = [4], 
                     ranges = [0, 8])

>> print("hist1", hist1)

hist1 [[6.]
 [5.]
 [5.]
 [0.]]

hist2 = cv2.calcHist(images = [src],
                     channels = [0],
                     mask = None,
                     histSize = [4],
                     ranges = [0, 4])

>> print("hist2", hist2)

hist2 [[4.]
 [2.]
 [2.]
 [3.]]
```

<br>

위에서 calcHist를 이용하여 간단하게 히스토그램을 구해보았습니다. 입력되는 파라미터를 보면 전부다 `배열`형태 즉, 
리스트로 입력된 것을 볼 수 있습니다. 왜? 라고 물으신다면,,, opencv의 정책에 따른다고 밖에 말할 수 없겠네요.
개인적으로 calcHist가 python 버전, c++ 버전 모두 난해했던것 같습니다.

파이썬에서 사용하는 범위는 기본적으로 \[, ) 범위를 가집니다. `[` 는 `이상`을 뜻하고 `)`는 미만을 뜻합니다.
hist1에서는 histSize가 4이고 ranges가 0이상 8미만이므로, 각각의 범위는 아래와 같습니다.

+ [0, 2) : 0, 1
+ [2, 4) : 2, 3
+ [4, 5) : 3, 4
+ [6, 7) : 4, 5

이번에는 lenna 이미지를 그레이스케일로 받아서 히스토그램을 계산해 보겠습니다.







