---
layout: post
title: (OpenCV-Python) 마우스 클릭으로 ROI 영역 추출 하기  
date: 2018-12-08 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, ROI 검출] # add tag
---

+ Reference : https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
+ Code : https://github.com/gaussian37/Vision/tree/master/OpenCV/crop%20image

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
```

<br>

+ 마우스 클릭 이벤트를 처리 하기 위하여 `click_and_crop` callback 함수를 선언합니다.
    + 마우스 클릭 이벤트가 발생할 때 마다 콜백 함수가 실행 됩니다.
    + event : 발생한 이벤트로 여기서는 마우스 왼쪽 버튼을 누르는 것과 떼는 것에 해당합니다.
    + x : 이벤트가 발생한 x 좌표 입니다.
    + y : 이벤트가 발생한 y 좌표 입니다.
    + flags : OpenCV에 의해 발생한 flags 입니다. 여기선 무의미 합니다.
    + param : 추가적인 parameter 입니다. 사용되지 않았습니다.
    
+ 위 코드가 콜백되면 전체적인 프로세스는 다음과 같습니다.
    + 마우스 왼쪽 클릭이 되면 좌표 저장
    + 마우스 왼쪽 클릭이 놓아지면 놓아진 지점 좌표 저장하고 이미지에 사각형 표시
    
    
```python
# argument parser를 구성해 주고 입력 받은 argument는 parse 합니다.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# 이미지를 load 합니다.
image = cv2.imread(args["image"])
# 원본 이미지를 clone 하여 복사해 둡니다.
clone = image.copy()
# 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)


'''
키보드에서 다음을 입력받아 수행합니다.
- q : 작업을 끝냅니다.
- r : 이미지를 초기화 합니다.
- c : ROI 사각형을 그리고 좌표를 출력합니다.
'''
while True:
	# 이미지를 출력하고 key 입력을 기다립니다.
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# 만약 r이 입력되면, crop 할 영열을 리셋합니다.
	if key == ord("r"):
		image = clone.copy()

 	# 만약 c가 입력되고 ROI 박스가 정확하게 입력되었다면
	# 박스의 좌표를 출력하고 crop한 영역을 출력합니다.
	elif key == ord("c"):
		if len(refPt) == 2:
			roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			print(refPt)
			cv2.imshow("ROI", roi)
			cv2.waitKey(0)
	# 만약 q가 입력되면 작업을 끝냅니다.
	elif key == ord("q"):
		break
 
# 모든 window를 종료합니다.
cv2.destroyAllWindows()
```
    
<br>

+ `--image` 파라미터를 통하여 이미지 파일의 경로를 받습니다.
+ 입력 받은 이미지는 `clone = image.copy()`을 통하여 원본을 따로 복사해 둡니다.
+ `cv2.setMouseCallback("image", click_and_crop)` 을 통해 선언한 콜백 함수를 세팅합니다.
+ 키보드에서 다음을 입력받아 수행합니다.
    - q : 작업을 끝냅니다.
    - r : 이미지를 초기화 합니다.
    - c : ROI 사각형을 그리고 좌표를 출력합니다.