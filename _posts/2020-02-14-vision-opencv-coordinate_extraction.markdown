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
- 입력값1 : -i 또는 --image_dir로 받으며 이미지들이 들어있는 디렉토리의 경로를 받습니다.
- 입력값2 : -r 또는 --result_dir로 받으며 좌표들이 저장된 텍스트 파일을 저장할 결로를 받습니다.
- 입력값3 : -p 또는 --points로 받으며 한 이미지 당 최대 저장할 좌표의 갯수를 받습니다.

<br>

```python
# 특정 디렉토리에서 이미지들을 읽고 점 2개만 찍어서 각도를 구한다.
# 점의 좌표 위치와 각도를 텍스트에 저장한다.
# 텍스트 저장 내용 : 이미지이름, x1, y1, x2, y2, ...

import argparse
import cv2
import numpy as np
import os
from datetime import datetime

num_points = 2
clicked_points = []
clone = None

def MouseLeftClick(event, x, y, flags, param):
	# 왼쪽 마우스가 클릭되면 (x, y) 좌표를 저장한다.
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # 만약 max_num_of_points의 갯수보다 더 많은 점을 클릭하면 처음 점은 삭제 한다.
        if len(clicked_points) > num_points:
            del clicked_points[0]

		# 원본 파일을 가져 와서 clicked_points에 있는 점들을 그린다.
        image = clone.copy()
        for point in clicked_points:
            cv2.circle(image, point, 2, (0, 255, 255), thickness = -1)
        cv2.imshow("image", image)

def main():
    global clone, clicked_points, num_points
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Enter the image files path")
    ap.add_argument("-r", "--result_dir", required=True, help="Enter the text file path of the coordinate values to be saved.")
    ap.add_argument("-p", "--points", required=True, help="Enter the number of points(coordinates) to be saved for each image")

    args = vars(ap.parse_args())
    image_dir_path = args['image_dir']
    result_path = args['result_dir']
    num_points = int(args['points'])    

    print("\n")
    print("1. 입력한 파라미터인 이미지 경로(-i 또는 --image_dir)에서 이미지들을 차례대로 읽어옵니다.")
    print("2. 입력한 파라미터인 점의 최대 갯수 만큼(-p 또는 --max_points)점을 찍습니다. 점의 최대 갯수를 초과하면 가장 먼저 찍은 점은 사라집니다.")
    print("3. 키보드에서 'n'을 누르면(next 약자) 다음 이미지로 넘어갑니다. 이 때, 작업한 점의 좌표가 저장 됩니다.")
    print("4. 이미지 경로에 존재하는 모든 이미지에 작업을 마친 경우 또는 'q'를 누르면(quit 약자) 프로그램이 종료됩니다.")
    print("\n")
    print("출력 포맷 : 이미지명\tx1좌표\ty1좌표\tx2좌표\ty2좌표...")
    print("\n")

    image_list = os.listdir(image_dir_path)
    now = datetime.now()
    now_str = "%s-%s-%s-%s-%s-%s" % ( now.year, now.month, now.day, now.hour, now.minute, now.second)

    # 텍스트 파일을 출력 하기 위한 stream을 open 합니다.
    file_write = open(result_path + '/' + now_str + '.txt', 'w')

    # 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", MouseLeftClick)

    for image_file in image_list:

        full_image_path = image_dir_path + "/" + image_file
        image = cv2.imread(full_image_path)
        clone = image.copy()

        flag = False

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(0)

            if key == ord('n'):
                text_output = image_file + " "
                for points in clicked_points:
                    text_output += str(points[0]) + " " + str(points[1]) + " "
                text_output += '\n'
                file_write.write(text_output)

                # 클릭한 점 초기화
                clicked_points = []
                break

            if key == ord('q'):
                # 프로그램 종료
                flag = True
                break

        if flag:
            break

    # 모든 window를 종료합니다.
    cv2.destroyAllWindows()
    # 파일 쓰기를 종료합니다.
    file_write.close()

if __name__ == "__main__":
    main()    
```

<br>

- 위 코드를 실행하면 매 이미지마다 최대 `num_points` 갯수 만큼 점을 클릭할 수 있고 클릭한 점의 좌표를 텍스트로 저장할 수 있습니다.
- 만약 `num_points` 갯수 이상의 점을 클릭하게 되면 가장 먼저 입력한 점은 무시 됩니다.

<br>

[opencv 관련 글 목록](https://gaussian37.github.io/vision-opencv-table/)
