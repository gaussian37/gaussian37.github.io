---
layout: post
title: (OpenCV-Python) 마우스 클릭으로 ROI 영역 추출 하기  
date: 2018-12-08 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, ROI 검출] # add tag
---

Reference : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

이번 포스트에서는 마우스 클릭으로 drag & drop을 하여 ROI(Region Of Interest)를
추출하는 방법에 대하여 알아보도록 하겠습니다.

이번 포스트를 통해 사진에서 ROI에 해당하는 `Box의 좌표`를 얻는 방법을 통하여
다양한 응용을 할 수 있습니다.

+ 먼저 필요한 패키지들을 import 합니다.

```python
import argparse
import cv2
```

<br>

+ Points 들을 저장할 리스트와 Crop이 실행 되었는지 저장할 Boolean 변수를 선언합니다.

```python
refPt = []
cropping = False
``` 

<br>

+ click 하여 crop 하는 함수를 만듭니다.

```python
def click_and_crop(event, x, y, flags, param):
    # refPt와 cropping 변수를 global로 만듭니다.
	global refPt, cropping
 
    def click_and_crop(event, x, y, flags, param):
	# refPt와 cropping 변수를 global로 만듭니다.
	global refPt, cropping

	# 왼쪽 마우스가 클릭되면 (x, y) 좌표 기록을 시작하고
	# cropping = True로 만들어 줍니다.
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# 왼쪽 마우스 버튼이 놓여지면 (x, y) 좌표 기록을 하고 cropping 작업을 끝냅니다.
	# 이 때 crop한 영역을 보여줍니다.
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
 
		# ROI 사각형을 이미지에 그립니다.
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
    
    # 왼쪽 마우스 버튼을 놓으면 그 시점에서의 좌표를 기록하고
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

```












