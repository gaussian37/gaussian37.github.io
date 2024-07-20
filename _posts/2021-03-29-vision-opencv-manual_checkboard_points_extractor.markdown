---
layout: post
title: 매뉴얼 체크보드 점 추출기
date: 2021-03-29 00:00:00
img: vision/opencv/manual_checkboard_points_extractor/0.png
categories: [vision-opencv] 
tags: [opencv, checkboard, 체크보드] # add tag
---

<br>

- 카메라 캘리브레이션을 할 때 주로 사용하는 체크보드 패턴의 경우 패턴이 일정하거나 패턴을 인식하기 좋은 상황이라면 체크보드를 인식하는 다양한 알고리즘을 통하여 쉽게 인식할 수 있습니다.

<br>
<center><img src="../assets/img/vision/opencv/manual_checkboard_points_extractor/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 일반적인 체크보트 패턴에서 코너점을 찾아서 교차점을 얻은 결과입니다.

<br>

- 하지만 상황에 따라서 위 그림과 같은 간단한 체크보드 패턴이 있지 않은 경우가 있을 수 있고 모든 체크보드 패턴이 붙어있지 않고 떨어져 있는 경우도 있어서 한번에 체크보드의 모든 교차점들을 인식하기 어려울 수 있습니다.
- 경우에 따라서는 점들의 위치를 수동으로 옮겨주고 싶은 경우도 발생할 수 있습니다.

<br>

- 이와 같은 작업을 지원하기 위하여 점들의 위치를 수동으로 입력하되 편리하게 작업할 수 있는 툴이 필요합니다. 이 때 다음과 같이 코드를 이용할 수 있습니다. 실행해야 할 전체 코드는 글 가장 아래 부분에 있습니다.

<br>

```python
python manual_checkboard_points_extractor.py --image_path=checkboard.png --save_path=./
```

<br>

- 툴 사용방법은 다음과 같습니다.
- ① `MOUSE LEFT CLICK`: 클릭한 위치의 점을 표시합니다.
- ② `CTRL + MOUSE LEFT CLICK` : 클릭한 위치와 가장 가까운 코너점을 표시합니다.
- ③ `w` : 가장 마지막에 표시된 점의 위치를 위로 한칸 옮깁니다. (v 좌표 -1)
- ④ `s` : 가장 마지막에 표시된 점의 위치를 아래로 한칸 옮깁니다. (v 좌표 +1)
- ⑤ `a` : 가장 마지막에 표시된 점의 위치를 왼쪽으로 한칸 옮깁니다. (u 좌표 -1)
- ⑥ `d` : 가장 마지막에 표시된 점의 위치를 오른쪽으로 한칸 옮깁니다. (u 좌표 +1)
- ⑦ `b` or `backspace` : 가장 마지막에 표시된 점을 지웁니다.
- ⑧ `space` : `save_path` 위치에 표시된 이미지 결과 이미지와 표시된 점들의 좌표를 csv 파일에 저장하고 종료합니다.
- ⑨ `q` : 작업을 취소하고 종료합니다.

<br>

- 작업 화면은 다음과 같습니다. 아래 체크보드는 중간에 벽이 꺽이는 부분이 있다는 점에서 일반적인 체크보드 패턴과 다릅니다. 아래 그림을 보면 5번째 점을 `w, a, s, d`를 이용하여 움직이는 것을 볼 수 있습니다. 마지막에는 `space`를 눌러서 작업을 종료합니다.
- 저장된 결과는 `save_path`로 지정된 위치에 이미지 파일과 csv 파일이 저장된 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/vision/opencv/manual_checkboard_points_extractor/2.gif" alt="Drawing" style="width: 800px;"/></center>
<br>

- 아래는 재미로 작업을 해본 것입니다. 아래와 같이 체크보드 패턴이 어려운 상황에서는 본 글의 코드를 이용하여 좌표 위치를 추출해 낼 수 있습니다.

<br>
<center><img src="../assets/img/vision/opencv/manual_checkboard_points_extractor/3.gif" alt="Drawing" style="width: 800px;"/></center>
<br>

- 실행할 코드는 다음과 같습니다.

<br>

```python
import os
from datetime import datetime
import cv2
import numpy as np
import csv
import argparse

harris_corners = []
clicked_points = []
clone = None

def draw_points(image):
    global clicked_points
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    color = (0, 0, 255)
    thickness = 1
    lineType = cv2.LINE_AA

    for index, point in enumerate(clicked_points):
        org = (point[1], point[0] - 10)
        cv2.putText(image, str(index+1).zfill(3), org, fontFace, fontScale, color, thickness, lineType)
        cv2.circle(image, (point[1], point[0]), 3, (255, 0, 0), thickness = 1)
        cv2.circle(image, (point[1], point[0]), 1, (0, 0, 255), thickness = -1)
    return image

def find_nearest_corner(centroids, given_point):
    # Convert to numpy array for easier manipulation
    centroids = np.array(centroids)
    given_point = np.array(given_point)

    # Calculate Euclidean distance from given_point to each corner point
    distances = np.linalg.norm(centroids - given_point, axis=1)

    # Find the index of the minimum distance
    nearest_index = np.argmin(distances)
    nearest_corner = centroids[nearest_index]

    return nearest_corner

def MouseEvents(event, x, y, flags, param):   
    
    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
        # Find the checkerboard corners
        nearest_corner = find_nearest_corner(harris_corners[:, :2], (x, y))
        nearest_corner = [int(nearest_corner[0]), int(nearest_corner[1])]
        clicked_points.append([nearest_corner[1], nearest_corner[0]])
        print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {nearest_corner[0]}, {nearest_corner[1]}")
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([y, x])
        print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")

    if event == cv2.EVENT_LBUTTONDOWN:
        image = clone.copy()
        image = draw_points(image)
        cv2.imshow("image", image)

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', required=True)
parser.add_argument('--save_path', default="./results")
args = parser.parse_args()

def main():
    global clone, clicked_points, harris_corners

    now = datetime.now()
    now_str = "%s%02d%02d_%02d%02d%02d" % (now.year - 2000, now.month, now.day, now.hour, now.minute, now.second)   

    image = cv2.imread(args.image_path)
    clone = image.copy()
    
    # Harris corner detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

    # Dilate corner image to enhance corner points
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # Find centroids of the corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    harris_corners = centroids

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseEvents)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(0)

        if key == 32: # space
            save_path = args.save_path.rstrip(os.sep) + os.sep + os.path.basename(args.image_path).rstrip(".png") + "_" + now_str
            os.makedirs(save_path, exist_ok=True)
            with open(save_path + os.sep + 'points.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                # Write the header
                writer.writerow(['no', 'u', 'v'])
                # Write the data rows
                for index, point in enumerate(clicked_points):
                    writer.writerow( (index+1, point[1], point[0]))
                    
            if len(clicked_points) > 0:
                clicked_points.pop()
                image = clone.copy()
                image = draw_points(image)
                cv2.imwrite(save_path + os.sep + "points.png", image)
            break

        if key == 8 or  key == ord('b'):  # backspace or b(back)
            if len(clicked_points) > 0:
                clicked_points.pop()
                image = clone.copy()
                image = draw_points(image)
                cv2.imshow("image", image)
        
        elif key == ord('w'):
            if len(clicked_points) > 0:
                if clicked_points[-1][0] > 0:
                    clicked_points[-1][0] -= 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)

        elif key == ord('s'):
            if len(clicked_points) > 0:
                if clicked_points[-1][0] < image.shape[0]-1:
                    clicked_points[-1][0] += 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)

        elif key == ord('a'):
            if len(clicked_points) > 0:
                if clicked_points[-1][1] > 0:
                    clicked_points[-1][1] -= 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)

        elif key == ord('d'):
            if len(clicked_points) > 0:
                if clicked_points[-1][1] < image.shape[1]-1:
                    clicked_points[-1][1] += 1
                    print(f">>> {str(len(clicked_points)).zfill(3)} points: (u, v) = {clicked_points[-1][1]}, {clicked_points[-1][0]}")
                    
                    image = clone.copy()
                    image = draw_points(image)
                    cv2.imshow("image", image)
        else:
            image = clone.copy()
            image = draw_points(image)
            cv2.imshow("image", image)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
```