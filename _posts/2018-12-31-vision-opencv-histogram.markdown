---
layout: post
title: 임계값 검출 - threshold, adaptiveThreshold  
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

+ images : 히스토그램을 계산할 영상의 배열입니다. 영상은 같은 사이트, 깊이의 8bit unsigned 정수 또는 32bit 실수형 입니다.
+ channels : 히스토그램을 계산할 channel
+ mask : images\[i\]와 같은 크기의 8bit 이미지로, mask(x, y)가 0이 아닌 경우에만 image\[i\](x,y)을 히스토그램 계산에 사용합니다.
    + mask = None이면 마스크를 사용하지 않고, 모든 화소에서 히스토그램을 계산합니다.
+ histSize : 히스토그램 hist (return 값)의 각 빈(bin) 크기에 대한 정수 배열 입니다.
+ ranges : 히스토그램 각 빈의 경계값에 대한 배열입니다. opencv는 기본적으로 등간격 히스토그램을 제공합니다.
+ accumulate : True 이면 calcHist() 함수를 수행할 때, 히스토그램을 초기화 하지 않고, 이전 값을 계속 누적합니다.
+ hist : 히스토그램 리턴값
 


