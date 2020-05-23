---
layout: post
title: 이미지에서 마우스 클릭 좌표값 추출하기  
date: 2020-02-14 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv, coordinate, extraction, 좌표, 좌표 검출] # add tag
---

<br>

[opencv 관련 글 목록](https://gaussian37.github.io/vision-opencv-table/)

<br>

- 이번 글에서는 마우스를 클릭하여 좌표값을 추출하는 코드에 대하여 간략하게 알아보겠습니다.
- 아래 코드에서는 이미지에서 마우스를 클릭하는 시점의 점을 임시로 저장하였다고 각 이미지의 작업을 완료하였을 때, 텍스트에 모든 점의 결과를 저장합니다.
- 입력값1 : --path로 받으며 이미지들이 들어있는 디렉토리의 경로를 받습니다.
- 입력값2 : --sampling으로 sampling 할 단위 갯수를 입력 받습니다.

<br>

```python
import sys
import subprocess
import os
from datetime import datetime

# pip가 없으면 pip를 설치한다.
try:
    import pip
except ImportError:
    print("Install pip for python3")
    subprocess.call(['sudo', 'apt-get', 'install', 'python3-pip'])

# cv가 없으면 cv를 설치한다.
try:
    import cv2
except ModuleNotFoundError:
    print("Install opencv in python3")
    subprocess.call([sys.executable, "-m", "pip", "install", 'opencv-python'])
finally:
    import cv2

# argparse가 없으면 argparse를 설치한다.
try:
    import argparse
except ModuleNotFoundError:
    print("Install argparse in python3")
    subprocess.call([sys.executable, "-m", "pip", "install", 'argparse'])
finally:
    import argparse

# numpy가 없으면 numpy를 설치한다.
try:
    import numpy
except ModuleNotFoundError:
    print("Install numpy in python3")
    subprocess.call([sys.executable, "-m", "pip", "install", 'numpy'])
finally:
    import numpy as np

dir_del = None
clicked_points = []
clone = None

def MouseLeftClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((y, x))

		# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, (point[1], point[0]), 2, (0, 255, 255), thickness = -1)
        cv2.imshow("image", image)

def GetArgument():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Enter the image files path")
    ap.add_argument("--sampling", default=1, help="Enter the sampling number.(default = 1)")
    args = vars(ap.parse_args())
    path = args['path']
    sampling = int(args['sampling'])
    return path, sampling

def main():
    global clone, clicked_points
    
    print("\n")
    print("1. 입력한 파라미터인 이미지 경로(--path)에서 이미지들을 차례대로 읽어옵니다.")
    print("2. 키보드에서 'n'을 누르면(next 약자) 다음 이미지로 넘어갑니다. 이 때, 작업한 점의 좌표가 저장 됩니다.")
    print("3. 키보드에서 'b'를 누르면(back 약자) 직전에 입력한 좌표를 취소한다.")
    print("4. 이미지 경로에 존재하는 모든 이미지에 작업을 마친 경우 또는 'q'를 누르면(quit 약자) 프로그램이 종료됩니다.")
    print("\n")
    print("출력 포맷 : 이미지명,점의갯수,y1,x1,y2,x2,...")
    print("\n")

    # 이미지 디렉토리 경로를 입력 받는다.
    path, sampling = GetArgument()
    # path의 이미지명을 받는다.
    image_names = os.listdir(path)

    # path를 구분하는 delimiter를 구한다.
    if len(path.split('\\')) > 1:
        dir_del = '\\'
    else :
        dir_del = '/'

    # path에 입력된 마지막 폴더 명을 구한다.    
    folder_name = path.split(dir_del)[-1]

    # 결과 파일을 저장하기 위하여 현재 시각을 입력 받는다.
    now = datetime.now()
    now_str = "%s%02d%02d_%02d%02d%02d" % (now.year - 2000, now.month, now.day, now.hour, now.minute, now.second)   

    # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseLeftClick)

    for idx, image_name in enumerate(image_names):
        if (idx % sampling != 0):
            continue

        image_path = path + dir_del + image_name
        image = cv2.imread(image_path)

        clone = image.copy()

        flag = False

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(0)

            if key == ord('n'):
                # 텍스트 파일을 출력 하기 위한 stream을 open 합니다.
                # 중간에 프로그램이 꺼졌을 경우 작업한 것을 저장하기 위해 쓸 때 마다 파일을 연다.
                file_write = open('./' + now_str + '_' + folder_name + '.txt', 'a+')

                text_output = image_name
                text_output += "," + str(len(clicked_points))
                for points in clicked_points:
                    text_output += "," + str(points[0]) + "," + str(points[1])
                text_output += '\n'
                file_write.write(text_output)
                
                # 클릭한 점 초기화
                clicked_points = []

                # 파일 쓰기를 종료한다.
                file_write.close()

                break

            if key == ord('b'):
                if len(clicked_points) > 0:
                    clicked_points.pop()
                    image = clone.copy()
                    for point in clicked_points:
                        cv2.circle(image, (point[1], point[0]), 2, (0, 255, 255), thickness = -1)
                    cv2.imshow("image", image)

            if key == ord('q'):
                # 프로그램 종료
                flag = True
                break

        if flag:
            break

    # 모든 window를 종료합니다.
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
```

<br>

- 위 코드를 실행하면 매 이미지마다 최대 `num_points` 갯수 만큼 점을 클릭할 수 있고 클릭한 점의 좌표를 텍스트로 저장할 수 있습니다.
- 만약 `num_points` 갯수 이상의 점을 클릭하게 되면 가장 먼저 입력한 점은 무시 됩니다.

<br>

[opencv 관련 글 목록](https://gaussian37.github.io/vision-opencv-table/)
