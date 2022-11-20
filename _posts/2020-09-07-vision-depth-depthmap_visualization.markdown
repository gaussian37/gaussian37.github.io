---
layout: post
title: depthmap 시각화 방법
date: 2020-09-07 00:00:00
img: vision/depth/depthmap_visualization/0.png
categories: [vision-depth] 
tags: [depthmap, visualization] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 이번 글에서는 이미지의 각 픽셀 별 (또는 일부 픽셀) 소수 단위 (ex. 10.34 m)의 깊이 정보를 나타내는 방법과 컬러로 시각화 하는 방법에 대하여 간단하게 살펴보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [정수 형태로 depthmap 저장하는 방법](#정수-형태로-depthmap-저장하는-방법-1)
- ### [컬러 형태로 depthmap 시각화 하는 방법](#컬러-형태로-depthmap-시각화-하는-방법-1)

<br>

## **정수 형태로 depthmap 저장하는 방법**

<br>

- 3D 공간에서의 거리 정보를 2D 이미지에 사영하였을 때, 이미지의 각 픽셀 별 값은 깊이 방향의 값을 저장하게 되고 이 데이터를 `depthmap`이라고 부릅니다.
- `depthmap`을 이미지 형태로 저장하여 관리하는데 깊이 값은 소수를 포함한 값이므로 소수값을 그대로 이미지 형태로 저장할 수 없습니다. 따라서 소수값을 정수로 변환한 다음 정수값을 이용하여 이미지로 저장해야 합니다.
- 일반적으로 사용하는 `uint8`을 이용하면 0 ~ 255 범위를 표현할 수 있기 때문에 소수값의 대부분의 정보를 잃어버립니다. 따라서 `uint16`을 이용하여 0 ~ 65,535의 범위를 이용하면 다양한 소수값도 표현할 수 있습니다.
- 표현하고 싶은 깊이 정보의 범위가 0 ~ 100 m 라고 가정하겠습니다. 그러면 0 m는 0의 값에 대응되고 100m는 65,535 값에 대응되도록 하면 됩니다. 따라서 `int(depth/max_range * 65535)`를 이용하여 모든 깊이 값을 표현하면 됩니다.
- 예를 들어 90.256 m 를 표현하고 싶으면 `int(90.256 / 100 * 65535) = 59149` 로 픽셀의 값을 저장하면 됩니다. 저장할 때에는 `img.astype(np.uint16)` 형태로 타입을 `uint16`으로 바꾸어서 `cv2.imwrite("파일명", img)`로 저장하면 `uint16`으로 저장됩니다.
- 저장된 `depthmap`을 읽을 때에는 `cv2.imread("파일명", cv2.IMREAD_UNCHANGED)`와 같은 방식으로 원본을 그대로 읽어오도록 하면 `uint16` 타입으로 읽어와집니다.
- 이 때 픽셀을 `depth * max_range / 65535`를 이용하면 원래 깊이 값을 복원할 수 있습니다. 예를 들어 위 예제의 59149 값과 max_range=100을 이용하면 `59149 * 100 / 65535 = 90.255588...` 가 됩니다.

<br>

- 위 과정을 정리하면 다음과 같습니다.
- ① depthmap의 float 타입의 깊이(depth) 값을 `int(depth/max_range * 65535)` 형태의 `uint16` 값으로 변환합니다.
- ② depthmap의 타입을 다음과 같이 `uint16`으로 변환합니다.
    - `img = img.asypte(np.uint16)`
- ③ depthmap을 다음과 같이 저장합니다. 
    - `cv2.imwrite("파일명", img)`
- ④ 실제 사용할 때에는 다음과 같이 불러옵니다.
    - `img = cv2.imread("파일명", cv2.IMREAD_UNCHANGED)`
- ⑤ 깊이 값을 복원할 때에는 저장할 때 사용한 방식을 역산하여 다음과 같이 사용합니다.
    - `depth * max_range / 65535`

<br>

- ①, ⑤에 사용한 깊이의 표현 방법은 단순히 `depth * 255`와 같은 형태로 사용하기도 합니다. (NYU v2 데이터에서도 사용)

<br>

## **컬러 형태로 depthmap 시각화 하는 방법**

<br>

- `dense depthmap`(모든 픽셀에 depth가 있음) 또는 `sparse depthmap`(일부 픽셀에 depth가 있음)은 `grayscale` 형태로 되어 있어 시각화 하였을 때, 음영의 차이가 있어서 시각적으로 확인은 가능하지만 색상으로 확인할 때 보다는 뚜렷하지 않습니다.
- 따라서 `depthmap`을 컬러 형태로 변환하는 방법에 대하여 살펴보겠습니다.
- 샘플 데이터 : https://drive.google.com/file/d/1d3AsyUkunvC5zOpwBAG4caSoidjvNgUx/view?usp=share_link

<br>

- 샘플 데이터로 아래 코드를 따라하면 다음과 같은 결과를 얻을 수 있습니다. 왼쪽은 원본이고 오른쪽은 컬러로 시각화한 결과입니다.

<br>
<center><img src="../assets/img/vision/depth/depthmap_visualization/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

depthmap_path = "../path/to/depthmap.png"
max_range = 255

depthmap = cv2.imread(depthmap_path, cv2.IMREAD_UNCHANGED)

# 실제 depthmap이 저장된 방식에 맞게 depth 복원하여 사용하면 됩니다.
# depthmap = depthmap.astype(np.float32) / 255.0
# depthmap = depthmap.astype(np.float32) * max_range / 65535

def get_color_depthmap(depthmap, max_range):
    # 256 단계의 color map을 생성합니다.
    cmap = plt.cm.get_cmap("jet", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # sparse depthmap인 경우 depth가 있는 곳만 추출합니다.
    depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 0)

    H, W = depthmap.shape
    color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
    for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
        depth = depthmap[depth_pixel_v, depth_pixel_u]
        color_index = int(255 * min(depth, max_range) / max_range)
        color = cmap[color_index, :]
        cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=-1)
    
    return color_depthmap

plt.imshow(color_depthmap)
plt.show()

```

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>
