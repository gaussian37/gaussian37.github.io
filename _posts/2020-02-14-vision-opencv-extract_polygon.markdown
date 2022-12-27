---
layout: post
title: 이미지에서 마우스 클릭하여 다각형 추출 및 마스크 생성
date: 2020-02-14 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, coordinate, extraction, 좌표, 좌표 검출, polygon] # add tag
---

<br>

[opencv 관련 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>

- 아래 코드에서는 마우스를 클릭하여 찍은 점을 이용하여 다각형을 만들고 그 결과를 저장합니다.
- 마우스 왼쪽 클릭을 하면 클릭한 지점의 점이 선택되고 오른쪽 클릭을 하면 클릭한 지점과 가장 가까운 모서리의 점이 선택됩니다.
- 스페이스바를 누르면 선택된 점들을 이용하여 다각형을 만든 결과가 저장되며 저장되는 양식은 총 3가지 입니다.
- ① `blended_image.png` : 다각형이 만들어진 결과가 저장되며 다각형은 빨간색으로 표시됩니다.
- ② `mask.png` : 전체 영역에서 다각형 영역만 0으로 저장되며 나머지 영역은 1로 저장됩니다. 마스킹을 하기 위한 목적으로 만들어 졌습니다.
- ③ `mask.npy` : numpy 배열을 그대로 저장하였습니다. shape은 (height, width, 1)의 크기입니다.

<br>

- 마스크는 실제 이미지와 곱하여 사용할 수 있습니다. 예를 들어 image : (height, width, 3) 이 있다고 하면 (image * mask) 를 통하여 `mask == 0` 인 부분을 제외해서 사용할 수 있습니다. 이런 목적으로 만들기 때문에 3차원인 (height, width, 1)로 만듭니다.
- 만약 `mask.png`를 읽어서 동일한 목적으로 사용하려면 다음과 같이 사용할 수 있습니다.
    - `img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)`
    - `img = np.expand_dims(img, -1)`

<br>

- 아래는 클릭을 통하여 다각형을 얻은 예시 입니다.

<br>
<center><img src="../assets/img/vision/opencv/extract_polygon/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>

```python
import os
import cv2
import argparse
import numpy as np

clicked_points = []
clone = None

def MouseClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])

		# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), thickness = -1)
        cv2.imshow("image", image)
        
    if event ==cv2.EVENT_RBUTTONDOWN:
        image = clone.copy()
        height, width = image.shape[0], image.shape[1]
        # 좌상단
        if x < width //2 and y < height//2:
            clicked_points.append([0, 0])
        # 우상단
        if x > width //2 and y < height//2:
            clicked_points.append([width-1, 0])
        # 좌하단
        if x < width //2 and y > height//2:
            clicked_points.append([0, height-1])
        # 우하단
        if x > width //2 and y > height//2:
            clicked_points.append([width-1, height-1])
            
        # 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), thickness = -1)
        cv2.imshow("image", image)
        

def GetArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Enter the image files path")
    args = vars(ap.parse_args())
    path = args['path']
    return path

def main():
    global clone, clicked_points
    print("- Clicking the left mouse button selects a pixel.")
    print("- Right-click selects the nearest image edge vertex.")
    print("- Pressing the spacebar saves the result of the points clicked.")
    print("- Press b to return to the previous step.")
    print("- Press r to reset.")
    print("- Press q to quit.")

    path = GetArgument()

    # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseClick)

    image = cv2.imread(path)
    clone = image.copy()
    flag = False

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(0)

        if key == 32: # spacebar
            # save mask
            mask = np.ones((image.shape[0], image.shape[1], 1))
            cv2.fillPoly(mask, pts=[np.array(clicked_points)], color=[0])            
            cv2.imwrite("mask.png", mask)
            '''
            To read "mask.png" and use:
                img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
                img = np.expand_dims(img, -1)
            '''
            
            # save npy file
            np.save("mask.npy", mask)
            
            # save blended image
            cv2.fillPoly(image, pts=[np.array(clicked_points)], color=(0, 0, 255))
            cv2.imwrite("blended_image.png", image)

        if key == ord('b'):
            if len(clicked_points) > 0:
                clicked_points.pop()
                image = clone.copy()
                for point in clicked_points:
                    cv2.circle(image, (point[0], point[1]), 5, (0, 0, 255), thickness = -1)
                cv2.imshow("image", image)
        
        if key == ord('r'):
            clicked_points = []
            image = clone.copy()
            cv2.imshow("image", image)            

        if key == ord('q'):
            # 프로그램 종료
            flag = True
            break

    # 모든 window를 종료합니다.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()```
```

<br>

[opencv 관련 글 목록](https://gaussian37.github.io/vision-opencv-table/)
