---
layout: post
title: 카메라 모델과 렌즈 왜곡 (lense distortion)
date: 2022-03-29 00:00:00
img: vision/concept/lense_distortion/0.png
categories: [vision-concept] 
tags: [lense distortion, 카메라 모델, 렌즈 왜곡] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 참조 : https://kr.mathworks.com/help/vision/ug/camera-calibration.html
- 참조 : http://jinyongjeong.github.io/2020/06/19/SLAM-Opencv-Camera-model-%EC%A0%95%EB%A6%AC/
- 참조 : http://jinyongjeong.github.io/2020/06/15/Camera_and_distortion_model/
- 참조 : https://docs.nvidia.com/vpi/algo_ldc.html
- 참조 : http://www.gisdeveloper.co.kr/?p=6868
- 참조 : https://ori.codes/artificial-intelligence/camera-calibration/camera-distortions/
- 참조 : http://www.close-range.com/docs/Decentering_Distortion_of_Lenses_Brown_1966_may_444-462.pdf

<br>

- 이번 글에서는 카메라 모델의 특성, 카메라 렌즈 왜곡 모델 그리고 렌즈 왜곡을 제거하는 방법 등에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [화각에 따른 카메라의 종류](#)
- ### [표준 카메라 렌즈 왜곡 모델 ](#)
- ### [표준 카메라 렌즈 왜곡 보정 방법](#)
- ### [표준 카메라의 렌즈 왜곡 보정 실습](#)
    - ### [remap을 이용한 왜곡 영상 → 핀홀 모델 영상](#)
    - ### [왜곡 영상 Points → 핀홀 모델 영상 Points](#)
    - ### [Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상](#)
- ### [어안 카메라 렌즈 왜곡 모델 ](#)
- ### [어안 카메라 렌즈 왜곡 보정 방법](#)
- ### [어안 카메라의 렌즈 왜곡 보정 실습](#)
    - ### [remap을 이용한 왜곡 영상 → 핀홀 모델 영상](#)
    - ### [왜곡 영상 Points → 핀홀 모델 영상 Points](#)
    - ### [Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상](#)
- ### [렌즈 왜곡 모델링의 이론적 이해](#)

<br>

## **화각에 따른 카메라의 종류**

<br>

- 카메라에서 가장 중요한 부분 중 하나가 렌즈입니다. 핀홀 모델 카메라는 이론적으로 빛의 직진성을 이용하여 만든 이상적이면서 간단한 카메라 모델이지만 빛의 유입량이 적어 정상적인 이미지를 만들어낼 수 없습니다.
- 따라서 렌즈를 이용하여 빛의 양이 많이 유입될 수 있도록 (사람의 수정체와 같습니다.) 카메라에서 사용하며 **렌즈의 형태에 따라 카메라가 빛을 유입할 수 있는 영역이 달라지기 때문에** 아래 그림과 같이 **렌즈에 따른 화각이 결정됩니다.**

<br>
<center><img src="../assets/img/vision/concept/lense_distortion/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 화각과 초점 거리 (focal length)는 설명을 위한 예시이며 절대적인 기준은 아닙니다.
- 이번 글에서는 `표준 렌즈`를 사용하는 `표준 카메라`와 `어안 렌즈`를 사용하는 `어안 카메라`에서 렌즈로 인한 물체가 휘어져 보이는 현상이 발생하였을 때, 컴퓨터 비전에서 어떻게 처리하는 지 그 방식과 원리에 대하여 살펴보도록 하겠습니다.

<br>

## **표준 카메라 렌즈 왜곡 모델**

<br>

<br>

## **표준 카메라 렌즈 왜곡 보정 방법**

<br>

<br>

## **표준 카메라의 렌즈 왜곡 보정 실습**

<br>

- 실습을 위해 사용한 데이터는 아래 링크를 사용하였습니다.
    - 링크 : https://data.caltech.edu/records/jx9cx-fdh55
- 본 글에서 사용한 이미지를 그대로 사용하시려면 아래 링크에서 다운 받으시면 됩니다.
    - 링크 : https://drive.google.com/file/d/1MaWrXvGuudMld2prhGk_KQDM4Q5qjpw9/view?usp=share_link

<br>

- opencv에서는 `undistort` 함수를 통하여 왜곡 보정을 하거나 `initUndistortRectifyMap`을 이용하여 왜곡 보정하는 방법이 있습니다.
- 본 글에서는 `initUndistortRectifyMap`을 이용하여 `map_x`, `map_y`를 구하고 이 값을 이용하여 **`remap` 함수를 사용하여 이미지 전체를 왜곡 보정**하거나 **포인트의 매핑을 이용하여 포인트 단위로 왜곡 보정**하는 방법에 대하여 살펴보도록 하겠습니다.
- `remap` 함수를 사용하는 방식을 소개하는 이유는 이 방법이 실제 사용하기에 현실적이며 `undistort` 함수는 느려서 실시간으로 사용할 수 없기 때문입니다. 시간 측정 시, 수백배의 시간 차이가 나는 것을 확인할 수 있습니다.

<br>

### **remap을 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

- `remap` 함수는 입력 이미지의 x, y 좌표를 출력 이미지의 x, y 좌표 어디에 대응시켜야 할 지 대응 관계를 아래 코드에서 살펴볼 `map_x, map_y`에 정의해 두고 그 관계를 매핑 시켜주는 함수 입니다.
- `remap` 함수는 입력과 출력이 비선형 관계이어서 관계식이 복잡할 때, 간단히 픽셀 별 대응을 통하여 복잡한 비선형 관계를 매핑 관계로 단순화 하기 때문에 연산도 간단하고 컨셉도 단순하여 많이 사용합니다.
    - 관련 링크 : [remap 함수를 이용한 remapping](https://gaussian37.github.io/vision-concept-image_transformation/#remap-%ED%95%A8%EC%88%98%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-remapping-1)
- `remap` 함수를 사용하기 위해서는 `map_x`, `map_y`룰 구해야 하며 opencv의 `cv2.initUndistortRectifyMap`를 이용하여 이 값들을 구할 수 있습니다.
- 앞에서 표준 카메라의 렌즈 왜곡 

<br>

```python
# 기존 intrinsic과 distortion을 이용하여 출력할 이미지 사이즈 크기의 왜곡 보정 영상을 생성하기 위한 방법
# 아래 함수를 통해 왜곡 보정된 이미지에서 동작하는 new_intrinsic을 구할 수 있음
new_intrinsic, roi = cv2.getOptimalNewCameraMatrix(
    intrinsic, distortion, imageSize=(width, height), alpha, newImageSize=(new_width, new_height)
)

# (new_width, new_height) 크기의 undistortion 이미지를 만들어 냅니다.
# cv2.getOptimalNewCameraMatrix()의 newImageSize와 같은 크기로 만들어야 외곽의 여백을 최소화 할 수 있습니다.
map_x, map_y = cv2.initUndistortRectifyMap(
    intrinsic, distortion, None, new_intrinsic, (new_width, new_height)
)

dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
```

<br>

<br>

### **왜곡 영상 Points → 핀홀 모델 영상 Points**

<br>

<br>

### **Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

## **어안 카메라 렌즈 왜곡 모델**

<br>

<br>

## **어안 카메라 렌즈 왜곡 보정 방법**

<br>

- 어안 카메라 또한 렌즈 왜곡 보정 방법은 표준 카메라 왜곡 보정 방법과 같으며 계산 과정 중 사용하는 렌즈 왜곡 모델이 다릅니다. 상세 내용은 앞에서 설명한 표준 카메라 왜곡 보정 방법을 참조하시면 됩니다.
- 앞에서 살펴본 표준 카메라 모델의 `cv2.initUndistortRectifyMap`과 앞으로 살펴 볼 어안 카메라 모델의 `cv2.fisheye.initUndistortRectifyMap`의 왜곡 보정 계수를 $$ p1 = 0, p2 = 0 $$ 으로 맞추고 $$ k1, k2, ... $$ 등의 값을 맞추더라도 왜곡 모델에 사용된 식이 일부 다르기 때문에 다른 값이 나옵니다.

<br>

## **어안 카메라의 렌즈 왜곡 보정 실습**

<br>

- 실습을 위해 사용한 데이터는 아래 링크를 사용하였습니다.
    - 링크 : https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
- 본 글에서 사용한 이미지를 그대로 사용하시려면 아래 링크에서 다운 받으시면 됩니다.
    - 링크 : https://drive.google.com/file/d/1ApfPPwRIcTVdkZA3Mfwqqg17PlzLdBd2/view?usp=share_link

<br>

### **remap을 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

### **왜곡 영상 Points → 핀홀 모델 영상 Points**

<br>

<br>

### **Pytorch를 이용한 왜곡 영상 → 핀홀 모델 영상**

<br>

<br>

## **렌즈 왜곡 모델링의 이론적 이해**

<br>

<br>


<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>