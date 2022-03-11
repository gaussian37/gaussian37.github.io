---
layout: post
title: 카메라 캘리브레이션의 이해와 Python 실습
date: 2022-01-28 00:00:00
img: vision/concept/calibration/0.png
categories: [vision-concept] 
tags: [vision, concept, calibaration, 캘리브레이션] # add tag
---

<br>

- 참조 : https://ms-neerajkrishna.medium.com/
- 참조 : https://darkpgmr.tistory.com/32
- 참조 : https://www.mathworks.com/help/vision/ug/camera-calibration.html

<br>

- 이번 글에서는 컴퓨터 비전을 위한 카메라 내용 및 카메라 캘리브레이션 관련 내용과 파이썬을 이용하여 실습을 해보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [이미지 형성과 핀홀 모델 카메라](#이미지-형성과-핀홀-모델-카메라-1)
- ### [Camera Extrinsic Matrix with Example in Python](#camera-extrinsic-matrix-with-example-in-python-1)
- ### [Camera Intrinsic Matrix with Example in Python](#camera-intrinsic-matrix-with-example-in-python-1)
- ### [Find the Minimum Stretching Direction of Positive Definite Matrices](#)
- ### [Camera Calibration with Example in Python](#camera-calibration-with-example-in-python-1)

<br>

## **이미지 형성과 핀홀 모델 카메라**

<br>

- 이미지 형성의 기본 아이디어는 `object`에서 `medium`으로 반사되는 광선(Rays)을 포착하는 것에서 부터 시작합니다.
- 가장 단순한 방법은 `object` 앞에 `medium`을 놓고 반사되어 들어오는 광선을 캡쳐하면 됩니다. 하지만 단순히 이러한 방식으로 하면 필름 전체에 회색만 보일 수 있습니다. 왜냐하면 object의 다른 지점에서 나오는 광선이 필름에서 서로 겹쳐서 엉망이 되기 때문입니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 살펴보면 object에서 부터 반사되어 나온 광선이 Medium의 같은 위치에서 수집될 수 있습니다. 이런 경우 object로 부터 반사되어 나온 위치의 형상을 정확히 알 수 없습니다. 

<br>
<center><img src="../assets/img/vision/concept/calibration/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 위 그림과 같이 `pin hole` 구조를 이용하여 object의 어떤 위치가 medium의 픽셀 간 일대일 대응이 될 수 있도록 만듭니다. 이와 같은 방법을 이용하여 medium의 한 픽셀에 object의 여러 부분이 겹치는 문제를 개선할 수 있습니다.
- 하지만 위 그림에서 한가지 문제점이 있습니다. 광선이 medium에 맺히는 위치가 반전이 되어 있습니다. 예를 들어 나무 윗부분의 물체가 medium의 아랫 부분에 위치해 있는 것을 알 수 있습니다. 이것이 `pin hole` 카메라의 특성이며 이 성질을 이용하여 수학적으로 모델링을 해야 합니다.

<br>

#### **Pinhoe camera**

<br>
<center><img src="../assets/img/vision/concept/calibration/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서 설명한 핀홀 카메라 모델은 대부분의 컴퓨터 비전 분야에서 기본적으로 사용됩니다. 하지만 핀홀 카메라 모델은 매우 이상적인 카메라 모델이며 실제로는 렌즈의 특성이 반영되어야 하므로 영상 왜곡 등도 같이 고려되어야 합니다.

<br>

- 그러면 핀홀 카메라의 원리를 이용하여 수학적으로 모델링 하기 위하여 아래 2가지 가정을 전제로 두겠습니다.
- ① 앞의 그림에서 광선을 캡쳐하는 medium 또는 필름 구조를 `image plane` 이라고 하며 촬영자 - `image plane` - 핀홀- 물체 순서로 위치해 있습니다. 편의상 핀홀 앞쪽이라고 말하겠습니다. 하지만 실제로는 `image plane`은 핀홀 뒤쪽에 위치해 있습니다. 이와 같은 구조로 설계하는 것이 물체로부터 반사된 광선을 image plane에 projection하기 쉽기 때문이며 앞서 언급한 반전된 이미지 문제도 해소할 수 있기 때문입니다. 자세한 구조는 생략하며 여기서 짚고 넘어갈 점은 촬영자 - 핀홀 - `image plane` - `world space` 순서로 위치한 점입니다.
- ② world space에서 부터 반사된 수많은 광선은 핀홀로 수렴된다고 가정합니다. 이 지점을 `center of projection` 또는 `camera center`라고 합니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이러한 가정을 이용하면 `world space`에 있는 물체들이 `image plane`에 투영되고 최종적으로 `center of projection (camera center)`로 수렴되는 구조로 이해할 수 있습니다.
- `image plane`은 XY plane과 평행하고 `center of projection`과 일정 거리 떨어져 있습니다. 이 거리를 `focal length`라고 하며 이후에 다룰 예정입니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 좌표계는 위 그림과 같으며 오른손으로 엄지, 검지, 중지로 나타내었을 때, 각각 y, z, x 축을 가리킵니다. 이를 참조하여 좌표계를 이해하시면 됩니다.

<br>

#### **Geometric camera calibration**

<br>

- 카메라는 어느 위치에나 장착될 수 있고 카메라가 보는 방향도 제각각 입니다. 여기서 해결해야 할 점은 `world space`에서 반사되어 들어오는 광선을 어떻게 `image plane`에 투영할 지 정해야 한다는 것입니다. 바꿔 말하면 `world space`와 `image plane` 간의 관계를 알아내어야 실제 이미지를 만들어 낼 수 있습니다.
- 이 관계는 `world space` → `image planem`으로 변환하는 행렬을 구해야 하며 이 때 필요한 2가지 행렬을 `extrinsic`, `intrinsic` 이라고 합니다.
- `extrinsic` : `world space`의 좌표계를 `world coordinate system`이라고 하고 앞에서 $$ X, Y, Z $$ 축으로 표현한 좌표계를 `camera coordinate system`이라고 합니다. 이 때, `world coordinate system` → `camera coordinate system`으로 좌표계를 변환할 때 사용하는 행렬을 `extrinsic`이라고 합니다. 이 행렬은 카메라가 실제 장착된 위치 등의 환경과 관련이 있습니다.
- `intrinsic` : `camera coordinate system`의 점들을 `image plane`의 좌표로 변환하는 행렬을 `intrinsic`이라고 합니다. 이 행렬은 카메라 내부 환경과 관련이 있습니다.
- `extrinsic`과 `intrinsic`을 확인하는 것을 카메라 캘리브레이션이라고 합니다.

<br>

## **Camera Extrinsic Matrix with Example in Python**

<br>




<br>

## **Camera Intrinsic Matrix with Example in Python**

<br>


<br>

## **Find the Minimum Stretching Direction of Positive Definite Matrices**

<br>


<br>

## **Camera Calibration with Example in Python**

<br>


<br>


