---
layout: post
title: Fisheye Camera (어안 카메라) 관련 정리
date: 2022-03-29 00:00:00
img: vision/concept/fisheye_camera/0.png
categories: [vision-concept] 
tags: [fisheye camera, 어안 카메라, lens distortion, 카메라 모델, 렌즈 왜곡] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 사전 지식 : [카메라 모델 및 카메라 캘리브레이션의 이해와 Python 실습](https://gaussian37.github.io/vision-concept-calibration/)
- 사전 지식 : [카메라 모델과 렌즈 왜곡 (lens distortion)](https://gaussian37.github.io/vision-concept-lens_distortion/)

<br>

- 참조 : https://arxiv.org/abs/2205.13281
- 참조 : https://plaut.github.io/fisheye_tutorial/#pinhole-camera-distortion
- 참조 : https://kr.mathworks.com/help/vision/ug/camera-calibration.html
- 참조 : http://jinyongjeong.github.io/2020/06/19/SLAM-Opencv-Camera-model-%EC%A0%95%EB%A6%AC/
- 참조 : http://jinyongjeong.github.io/2020/06/15/Camera_and_distortion_model/

<br>

- 이번 글에서는 `Fisheye Camera`에 관한 전반적인 내용에 대하여 다루도록 하겠습니다. 글의 내용을 이해하려면 사전 지식에 해당하는 2개의 글을 반드시 읽고 오시길 권장 드립니다.
- 글의 전반적인 내용은 `Fisheye Camera`가 가지는 의미나 필요했던 내용 그리고 `Kumar`의 `survey` 논문에 대한 내용을 다룹니다.
- 본 글에서 다루는 데이터셋은 [woodscape](https://woodscape.valeo.com/dataset) 데이터셋 이거나 개인적으로 구매한 `Fisheye Camera`를 이용하여 촬영한 데이터셋입니다. 제가 자체적으로 취득한 데이터셋들은 아래 링크에 공개하오니 자유롭게 쓰셔도 됩니다.
    - `Woodscape` (Public Dataset) : https://woodscape.valeo.com/dataset
    - `Fisheye Mono Camera` (Custom Dataset) : https://drive.google.com/drive/u/0/folders/16kPNXPaBFMogi8xt3fm6jhKp4jO6wB-H
    - `Fisheye Stereo Camera` (Custom Dataset) : https://drive.google.com/drive/u/0/folders/1Q4f8bAD0lypEXgqvqehrgbAtEGzC6jOX

<br>

- Custom Dataset의 `Fisheye Camera`는 다음 링크에서 구매하였습니다.
    - `Fisheye Mono Camera` : https://ko.aliexpress.com/item/4000333240423.html
    - `Fisheye Stereo Camera` : https://astar.ai/collections/astar-products/products/stereo-camera

<br>

## **목차**

<br>

- ### Fisheye Camera의 특징과 Pinhole Camera와의 차이점
- ### Fisheye Camera의 Vignetting 영역 인식 방법
- ### Generic Camera 모델의 Fisheye Camera의 유효 영역 확인 방법
- ### Surround-view Fisheye Camera Perception for Automated Driving 리뷰
    - ### Abstract
    - ### 1. Introduction
    - ### 2. Fisheye Camera Models
    - ### 3. Surround View Camera System
    - ### 4. Perception Tasks
    - ### 5. Public Datasets And Research Directions

<br>

## **Fisheye Camera의 특징과 Pinhole Camera와의 차이점**

<br>

<br>

## **Fisheye Camera의 Vignetting 영역 인식 방법**

<br>

- 아래는 `Fisheye Camera`의 `vignetting` 영역을 인식하는 방법입니다.
- `Vignetting` 영역은 상이 맺히지 않아 일반적으로 검은색으로 나타나며 `grayscale` 형태로 이미지를 나타냈을 때, 0에 가까운 값을 가지게 됩니다.
- 아래 코드는 이미지의 상단부터 하단 까지 좌/우 양끝에서 `vignetting` 영역이 아닌 픽셀 까지 찾은 다음에 `least square`를 통하여 `circle`을 찾습니다. `circle`은 `center point`와 `radius`를 
- 마지막으로 안전하게 `radius` 값을 `margin` 만큼 줄이면 안전하게 내부 영역을 찾을 수 있습니다.

<br>

```python
from scipy.optimize import leastsq

# Function to calculate the residuals for least squares circle fit
def calculate_residuals(c, x, y):
    xi = c[0]
    yi = c[1]
    ri = c[2]
    return ((x-xi)**2 + (y-yi)**2 - ri**2)

# Initialize lists to store the coordinates of the first non-black pixels from left and right for each row
x_coords = []
y_coords = []

non_vignetting_threshold = 20
inner_circle_margin = 10

img = cv2.imread("image.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Scan each row of the image
for i in range(img_gray.shape[0]):

    # Scan from the left
    for j in range(img_gray.shape[1]):
        if np.any(img_gray[i,j] > non_vignetting_threshold):
            x_coords.append(j)
            y_coords.append(i)
            break

    # Scan from the right
    for j in range(img_gray.shape[1]-1, -1, -1):
        if np.any(img_gray[i,j] > non_vignetting_threshold):
            x_coords.append(j)
            y_coords.append(i)
            break

# Convert the lists to numpy arrays
x = np.array(x_coords)
y = np.array(y_coords)

# Initial guess for circle parameters (center at middle of image, radius half the image width)
c0 = [img_gray.shape[1]/2, img_gray.shape[0]/2, img_gray.shape[1]/4]

# Perform least squares circle fit
c, _ = leastsq(calculate_residuals, c0, args=(x, y))

img_color = img.copy()
# Draw the circle on the original image
cv2.circle(img_color, (int(c[0]), int(c[1])), int(c[2])-10, (0, 255, 0), 2);

# Fill in the inside of the circle
mask_valid = np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
cv2.circle(mask_valid, (int(c[0]), int(c[1])), int(c[2])-inner_circle_margin, 1, -1);
```

<br>
<center><img src="../assets/img/vision/concept/fisheye_camera/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 내부 `circle`을 찾을 수 있으며 `circle` 내부의 영역만 실제 유효한 `RGB` 값이 존재하는 영역임을 알 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/fisheye_camera/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `circle` 내부 영역을 표시하면 위 그림과 같습니다.

<br>

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>