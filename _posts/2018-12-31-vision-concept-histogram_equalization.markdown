---
layout: post
title: Histogram Equalization (히스토그램 평활화) 알아보기
date: 2018-12-29 00:00:00
img: vision/coconcept/histogram_equalization/0.png
categories: [vision-concept] 
tags: [vision, 히스토그램 평활화, histogram equalization, opencv, equalization, equalizeHist, 히스토그램, hist, histogram] # add tag
---

<br>

- Reference : Python으로 배우는 OpenCV 프로그래밍
- Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/histogram

<br>

히스토그램 평활화 방법은 영상의 픽셀값들의 `누적분포함수` 를 이용하여 영상을 개선하는 방법입니다.
즉, 가장 간단한 `image enhancement` 방법 중 하나입니다.

<br>

- 히스토그램 평활화는 화소값의 범위가 좁은 low contrast 입력 영상을 이용하여 화소값의 범위가 넓은 high contrast 출력 영상을 얻습니다. 즉, 밝기 값이 몰려 있어서 어둡기만 한 영상 또는 밝기만 한 영상을 평활화하여 좀 더 `선명한 영상`을 얻습니다.
- 히스토그램 평활화를 적용하기 위하여 그레이스케일 영상은 입력 영상에 바로 히스토그램 평활화를 적용하면 됩니다. 반면 컬러 영상은 `HSV`, `YCrCb` 등의 컬러 모델로 변환한 다음, 밝기값 채널 (V, Y)에 히스토그램 평활화를 적용 후 BGR 컬러로 변환합니다.

<br>

```python
dst = cv2.equalizeHist(src)
```

<br>

- 히스토그램 평활화 과정을 정확하게 확인하기 위하여 아래 예제를 살펴보도록 하겠습니다.
- 아래 예제에서 사용되는 입력 `src`는 1채널, 8비트 입력 영상이며 출력값 `dst`는 `src`와 같은 크기, 같은 종류의 히스토그램 평활화된 출력을 가집니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/unequalHist.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 어떤 이미지의 히스토그램이 위와 같다고 생각해 보겠습니다. 상당히 한쪽으로 치우쳐 있습니다.
위와 같은 경우를 이미지에 대응시켜보면 상당히 밝은 이미지에 해당하게 됩니다. 즉, 너무 밝아서 선명하지가 않게 됩니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/process.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 히스토그램 평활화를 위하여 차례 차례 계산하는 방법을 알아보겠습니다.
- `1열` :  Gray Level 은 영상의 픽셀 값이라고 볼 수 있습니다.
- `2열` : $$ n_{i} $$ 는 각 gray level의 빈도 수 입니다.
- `3열` : $$ \sum n_{i} $$ 는 2열을 누적합 하였습니다. 누적합의 최종 결과는 360 입니다.
- `4열` : gray level의 총 갯수는 15개 (일반적 이미지에서는 8bit를 사용하므로 255) 이고 누적합이 360 이므로 각 gray level 별 누적합에 $$ \frac{15}{360} $$ 을 곱해줍니다. 이 방법을 통해 정규화를 해주고 나온 결과값은 반올림을 해줍니다.
- `5열` : 이 값은 4열 값을 반올림 해준 것이고 2열의 값과 비교하면 `균등하게 분포`되었음을 볼 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/equalHist.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 처음 이미지의 히스토그램과 비교하였을 때, 히스토그램 평활화를 거친 히스토그램이 상당히 균등하게 분포되었음을 볼 수 있습니다.
- 이 과정을 OpenCV의 `cv2.equalizeHist()`를 사용하면 쉽게 적용할 수 있습니다. 8bit 이미지를 대상으로 적용하기 때문에 $$ [0, 255 ] $$ 사이로 균등하게 분포시킵니다.

<br>

```python
import cv2
import numpy as np

src = np.array([ [0, 0, 0, 0],
                 [1, 1, 1, 1],
                 [200, 200, 200, 200],
                 [250, 250, 250, 250]], dtype = np.uint8)

dst = cv2.equalizeHist(src)

>> print(dst)
# [[  0   0   0   0]
#  [ 85  85  85  85]
#  [170 170 170 170]
#  [255 255 255 255]]
```     

<br>

- 코드를 실행한 결과를 보면 위에서 손으로 계산해 본 것과 같이 0 ~ 255 사이의 범위에서 잘 분포된 것을 보실 수 있습니다.
- 이번에는 grayscale 이미지에 히스토그램을 적용해 보겠습니다. 코드는 다음과 같습니다.

<br>

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

- `src` 이미지는 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/src1.PNG" alt="Drawing" style="width: 500px;"/></center>
<br>

- `dst` 이미지는 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/dst1.PNG" alt="Drawing" style="width: 500px;"/></center>
<br>

- `src`와 `dst`의 히스토그램을 비교하면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/grayscaleHistLenaImage.png" alt="Drawing" style="width: 500px;"/></center>
<br>

- `dst` 이미지의 히스토그램이 전체적으로 잘 분포되어 있고 좀 더 선명한 것을 확인할 수 있습니다.
이번에는 컬러 이미지에 equalizeHist 함수를 적용해 보겠습니다.
- 컬러영상에 히스토그램 평활화를 적용할 때에는 RGB 값에 바로 적용하면 색이 변할 수 있습니다.
먼저 RGB로 받은 이미지를 `HSV` 또는 `YCrCb` 형태의 이미지로 변경한 다음에 밝기값 채널을 변경해야 **색을 변경하지 않고 선명**하게 만들 수 있습니다. (그럼에도 불구하고 미세하게 나마 색이 변경될 수는 있습니다.) 코드는 다음과 같습니다.

<br>

```python
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Image path to the directory")
# args = vars(ap.parse_args())
# path = args['image']

path = "lena.png"

# 입력 받은 이미지를 불러옵니다.
src = cv2.imread(path)

# hsv 컬러 형태로 변형합니다.
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
# h, s, v로 컬러 영상을 분리 합니다. 
h, s, v = cv2.split(hsv)
# v값을 히스토그램 평활화를 합니다.
equalizedV = cv2.equalizeHist(v)
# h,s,equalizedV를 합쳐서 새로운 hsv 이미지를 만듭니다.
hsv2 = cv2.merge([h,s,equalizedV])
# 마지막으로 hsv2를 다시 BGR 형태로 변경합니다.
hsvDst = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

# YCrCb 컬러 형태로 변환합니다.
yCrCb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
# y, Cr, Cb로 컬러 영상을 분리 합니다.
y, Cr, Cb = cv2.split(yCrCb)
# y값을 히스토그램 평활화를 합니다.
equalizedY = cv2.equalizeHist(y)
# equalizedY, Cr, Cb를 합쳐서 새로운 yCrCb 이미지를 만듭니다.
yCrCb2 = cv2.merge([equalizedY, Cr, Cb])
# 마지막으로 yCrCb2를 다시 BGR 형태로 변경합니다.
yCrCbDst = cv2.cvtColor(yCrCb2, cv2.COLOR_YCrCb2BGR)

# src, hsv, YCrCb 각각을 출력합니다.
cv2.imshow('src', src)
cv2.imshow('hsv dst', hsvDst)
cv2.imshow('YCrCb dst', yCrCbDst)
cv2.waitKey()
cv2.destroyAllWindows()
```

<br>

- 원본 영상은 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/colorSrc.PNG" alt="Drawing" style="width: 500px;"/></center>
<br>

- `hsv` 형태에서 `밝기값`을 평활화 한 영상입니다.

<br>
<center><img src="../assets/img/vision/concept/histogram_equalization/hsvDst.PNG" alt="Drawing" style="width: 500px;"/></center>
<br>