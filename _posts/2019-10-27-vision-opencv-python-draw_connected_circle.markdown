---
layout: post
title: 마우스로 연속적으로 이미지에 점 찍기   
date: 2019-10-27 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv] # add tag
---

<br>

- 이번 글은 use case로 마우스로 연속적으로 이미지에 점을 찍는 방법입니다.
- 아래 기능은 이미지에서 마우스를 누른 채로 점을 찍으면 점을 이미지에 그리고 좌표를 출력합니다.  

<br>

```python
import cv2
import numpy as np

# 마우스 콜백 함수: 연속적인 원을 그리기 위한 콜백 함수
def DrawConnectedCircle(event, x, y, flags, param):
    global drawing
    
    # 마우스 왼쪽 버튼이 눌리면 드로윙을 시작함
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img,(x,y),2,(0,0,255),-1)
        print(x, y)
    
    # 마우스가 왼쪽 버튼으로 눌린 상태에서 마우스 포인트를 움직이면 
    # 움직인 자취를 따라서 마우스의 점들이 그려짐
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),2,(0,0,255),-1)
            print(x, y)
    
    # 마우스 왼쪽 버튼을 떼면 드로윙을 종료함
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

drawing = False # 마우스 왼쪽 버튼이 눌러지면 그리기 시작 
img = np.zeros((512, 512, 3), np.uint8) + 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', DrawConnectedCircle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break

cv2.destroyAllWindows()
```