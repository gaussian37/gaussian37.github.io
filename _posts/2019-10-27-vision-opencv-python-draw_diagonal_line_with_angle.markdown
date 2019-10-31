---
layout: post
title: 각도 기준으로 대각선 그리기   
date: 2019-10-27 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [vision, opencv] # add tag
---

<br>

- 이번 글은 use case로 이미지 내의 한 점이 있고, 그 점에서 `각도` 기준으로 대각선을 그리는 코드를 살펴보겠습니다.
- `스크린 좌표계`에서는 (0, 0) 이 왼쪽 상단에 있고 행 방향(↓)이 `y` 축이고 열 방향(→)이 `x` 축입니다.

<br>
<center><img src="../assets/img/vision/opencv/draw_diagonal_line_with_angle/degree.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같은 기준으로 회전을 한다고 가정하면 y좌표와 x좌표는 다음 기준으로 변경됩니다.

<br>
    
$$ y = y + sin( \frac{\pi}{180} \times \ angle) \times length $$

$$ x = x + cos( \frac{\pi}{180} \times \ angle) \times length $$

<br>

- 기준점과 특정 각도 만큼 회전한 점을 이은 직선을 아주 길게 그린 다음 이미지를 넘은 영역은 잘라 버리면 됩니다.

<br>

```python
import cv2
import numpy as np

image = np.zeros((512, 1024, 3), np.uint8) + 255
y_center = image.shape[0]//2
x_center = image.shape[1]//2

cv2.namedWindow('image')
cv2.circle(image, (x_center, y_center), 3, (255, 0, 0), -1)

infinity = 10000
image_height = image.shape[0]
image_width = image.shape[1]

for i in range(0, 100):

    angle = 0 + 45 * i
    if(angle >= 360):
        break
    # print(angle)

    # y, x 좌표를 각도 기준으로 회전
    # 기존 좌표와 회전한 좌표를 선으로 이었을 때, 이미지 끝 점과 교차점이 생기도록
    # 회전한 좌표는 이미지 밖에 두도록 합니다. 따라서 Length는 아주 큰 값 infinity로 둡니다.
    y_rotated = y_center + int(np.sin(np.pi / 180 * angle)*infinity)
    x_rotated = x_center + int(np.cos(np.pi / 180 * angle)*infinity)
    # print(y_rotated, x_rotated)
    
    # 기존 좌표와 이미지 끝에 교차된 점 까지를 이은 직선을 그립니다.
    ret, inner_point, clipped_point = cv2.clipLine((0, 0, image_width-1, image_height-1), (x_center, y_center), (x_rotated, y_rotated))
    cv2.line(image, (x_center, y_center), clipped_point, (255, 0, 0))

while(1):
    cv2.imshow('image',image)
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break

cv2.destroyAllWindows()
```

<br>

- 또다른 방법으로 `tan` 함수를 이용하여 구할 수도 있습니다.


<br>

- 추가적으로 이미지에서 생성된 직선의 모든 좌표값들을 찾고 싶으면 파란색 점 (255, 0, 0)에 해당하는 점들만 찾아서 저장하면 됩니다.

<br>

```python
points_set = set()

for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        this_image = image[i][j]
        if this_image[0] == 255 and this_image[1] == 0 and this_image[2] == 0:
            points_set.add( (i, j) )
```





