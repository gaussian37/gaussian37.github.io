---
layout: post
title: 히스토그램 평활화(equalization) - equalizeHist   
date: 2018-12-28 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, equalization, equalizeHist, 히스토그램, hist, histogram] # add tag
---

+ Reference : Python으로 배우는 OpenCV 프로그래밍
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/histogram

히스토그램 평활화 방법은 영상의 픽셀값들의 `누적분포함수` 를 이용하여 영상을 개선하는 방법입니다.
즉, 가장 간단한 `image enhancement` 방법 중 하나입니다.

+ 히스토그램 평활화는 화소값의 범위가 좁은 low contrast 입력 영상으로 화소값의 범위가 넓은 high contrast 출력 영상을 얻습니다.
    + 밝기 값이 몰려 있어서 어둡기만 한 영상 또는 밝기만 한 영상을 평활화하여 좀 더 `선명한 영상`을 얻습니다.
+ 그레이스케일 영상은 입력 영상에 바로 히스토그램 평활화를 적용하면 됩니다.
+ 컬러 영상은 HSV, YCrCb 등의 컬러 모델로 변환한 다음, 밝기값 채널 (V, Y)에 히스토그램 평활화를 적용 후 BGR 컬러로 변환합니다.

```python
dst = cv2.equalizeHist(src)
```

+ src : 1채널 8비트 입력 영상. dst는 src와 같은 크기, 같은 종류의 히스토그램 평활화된 출력

+ 평활화 과정

<img src="../assets/img/vision/opencv/pointprocessing/histogramEqualization/unequalHist.PNG" alt="Drawing" style="width: 500px;"/>

<br>

먼저 어떤 이미지의 히스토그램이 위와 같다고 생각해 보겠습니다. 상당히 한쪽으로 치우쳐 있습니다.
위와 같은 경우를 이미지에 대응시켜보면 상당히 밝은 이미지에 해당하게 됩니다. 너무 밝아서 선명하지가 않게 됩니다.

<img src="../assets/img/vision/opencv/pointprocessing/histogramEqualization/process.PNG" alt="Drawing" style="width: 500px;"/>

<br>

차례 차례 계산하는 방법을 알아보겠습니다.
+ 1열 Gray Level 은 영상의 픽셀 값이라고 볼 수 있습니다.
+ 2열 $$ n_{i} $$ 는 각 gray level의 빈도 수 입니다.
+ 3열 $$ \sum n_{i} $$ 는 2열을 누적합 하였습니다. 누적합의 최종 결과는 360 입니다.
+ 4열 gray level의 총 갯수는 15개 (일반적 이미지에서는 8bit를 사용하므로 255) 이고 누적합이 360 이므로 각 gray level 별 누적합에 $$ \frac{15}{360} $$ 을 곱해줍니다.
    + 이 방법을 통해 정규화를 해주고 나온 결과값은 반올림을 해줍니다.
+ 5열의 값은 4열 값을 반올림 해준 것이고 2열의 값과 비교하면 `균등하게 분포`되었음을 볼 수 있습니다.

<img src="../assets/img/vision/opencv/pointprocessing/histogramEqualization/equalHist.PNG" alt="Drawing" style="width: 500px;"/>

<br>

히스토그램이 상당히 균등하게 분포되었음을 볼 수 있습니다.

+ cv2.equalizeHist() 를 사용하면 8bit 이미지를 대상으로 하기 때문에 $$ \[0, 255 \] $$ 사이로 균등하게 분포시킵니다.

```python
import cv2
import numpy as np

src = np.array([ [0, 0, 0, 0],
                 [1, 1, 1, 1],
                 [200, 200, 200, 200],
                 [250, 250, 250, 250]], dtype = np.uint8)

dst = cv2.equalizeHist(src)

>> print(dst)
[[  0   0   0   0]
 [ 85  85  85  85]
 [170 170 170 170]
 [255 255 255 255]]
```     

<br>

결과를 보면 위에서 손으로 계산해 본 것과 같이 0 ~ 255 사이의 범위에서 잘 분포된 것을 보실 수 있습니다.
이번에는 grayscale 이미지에 히스토그램을 적용해 보겠습니다.

코드는 다음과 같습니다.

```python
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path to the directory")
args = vars(ap.parse_args())
path = args['image']
fname = os.path.splitext(path)[0]

# 입력 받은 이미지를 불러옵니다.
src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# 불러온 이미지에 histogram equalization을 적용합니다.
dst = cv2.equalizeHist(src)
srcHist = cv2.calcHist(images = [src],
                       channels = [0],
                       mask = None,
                       histSize = [256],
                       ranges = [0, 256])

dstHist = cv2.calcHist(images = [dst],
                       channels = [0],
                       mask = None,
                       histSize = [256],
                       ranges = [0, 256])

cv2.imshow('src', src)
cv2.imshow('dst', dst)
plt.title('Grayscale histogram of {} image'.format(fname), fontSize = 16)
plt.plot(srcHist, color = 'b', label = 'src hist')
plt.plot(dstHist, color = 'r', label = 'dst hist')
plt.legend(loc='best')
plt.show()


cv2.waitKey()
cv2.destroyAllWindows()

```

<br>

+ src 이미지

<img src="../assets/img/vision/opencv/pointprocessing/histogramEqualization/src1.PNG" alt="Drawing" style="width: 500px;"/>

+ dst 이미지

<img src="../assets/img/vision/opencv/pointprocessing/histogramEqualization/dst1.PNG" alt="Drawing" style="width: 500px;"/>

+ 히스토그램 비교

<img src="../assets/img/vision/opencv/pointprocessing/histogramEqualization/grayscalehistogram.PNG" alt="Drawing" style="width: 500px;"/>

dst 이미지의 히스토그램이 전체적으로 잘 분포되어 있고 좀 더 선명한 것을 보실 수 있습니다.


도움이 되셨다면 광고 클릭 한번이 저에게 큰 도움이 되겠습니다.^^