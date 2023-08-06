---
layout: post
title: Translating Images into Maps
date: 2023-08-06 00:00:00
img: vision/fusion/translating_images_into_maps/0.png
categories: [vision-fusion] 
tags: [nvautonet, multi-camera fusion] # add tag
---

<br>

[fusion 관련 글 목차](https://gaussian37.github.io/vision-fusion-table/)

<br>

- 논문 : https://arxiv.org/abs/2303.12976
- NVIDIA DRIVE Perception : https://developer.nvidia.com/drive/perception

<br>

- 이번 글에서는 `NVIDIA`에서 arxiv에 등재한 `NVAutoNet` 이라는 논문을 살펴보도록 하겠습니다. 

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 논문의 핵심 내용은 부제인 `Fast and Accurate 360° 3D Visual Perception For Self Driving`에서 알 수 있듯이 ① `Fast`, ② `Accurate`, ③ `360°`, ④ `Visual Perception` 4가지 요소를 만족하도록 네트워크가 설계되어 있음을 설명합니다.
- 즉, 자율주행을 위한 딥러닝 모델을 설계하였는데, `latency`가 적고, `accuracy`가 정확한 모델을 만들고자 하면서 `360°` 주변 전체를 `vision` 방식으로 인식하고자 한 것입니다.
- 논문의 전체 내용은 `NVIDIA`의 자체 SoC에 실현 가능성 있게 구현을 잘 해본 것에 의의를 두고 있으나 코드 공개 및 실제 public 데이터셋과의 비교는 없어 다소 아쉬움이 있습니다.

<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [1. Introduction](#1-introduction-1)
- ### [2. Related Work](#2-related-work-1)
- ### [3. NVAutoNet](#3-nvautonet-1)
- ### [4. Perception Tasks](#4-perception-tasks-1)
- ### [5. Multi-task Learning and Loss Balancing](#5-multi-task-learning-and-loss-balancing-1)
- ### [6. Experimental Evaluation](#6-experimental-evaluation-1)

<br>

## **Abstract**

<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림 및 테이블은 `NVAutoNet`에서 표현하고자 하는 전체 내용을 일목요연하게 보여줍니다. 
- ① `Input Images` : 차량 주변의 360° 환경을 촬영한 이미지를 입력으로 `Image Encoder`에 넣어줍니다.
- ② `Image Encoders → Image Features` : 각 이미지를 처리하는 `Image Encoder`가 존재하여 각 이미지를 입력 받은 후 처리하여 `Image Feature`로 만듭니다.
- ③ `2D-to-BEV + Calibrations → BEV Features` : `Image Feauture`는  2D image에서 특징들을 추출한 것입니다. 따라서 아직 2D 이미지 공간에 있는 정보들입니다. 이 값들을 `BEV (Bird's Eye View)` 공간에 표현하기 위해 `uplift` 하는 과정을 거칩니다. 이 때, 카메라 `calibration (intrinsic 으로 추정함)` 값이 입력값으로 사용 됩니다. 이 과정을 통해 각 카메라 별 `BEV Feature`를 생성합니다.
- ④ `BEV Fusion + Calibrations → Fused BEV Features` : 각 카메라 별 `BEV Feature`를 하나의 `BEV` 공간에 합치기 위해 `fusion`을 합니다. 이 때, 통일된 하나의 공간으로 `fusion` 하기 위하여 카메라 `calibration (extrinsic 으로 추정함)` 값을 사용하여 하나의 `BEV`공간에 `fusion`된 `Fused BEV Features`를 생성합니다.
- ⑤ `BEV Encoder → 3D outputs` : `Fused BEV Features`에서 각 Task 별 원하는 출력을 생성하기 위하여 `BEV Encoder`를 사용합니다. 본 논문에서는 `3D Object Detection`과 `Freespace Detection` 2가지 문제를 풀기 위한 학습을 `Multi-Task` 방식으로 풀어 나갑니다. 

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 본 논문은 `time-synced camera image`를 입력으로 받아서 3D 정보를 출력하는 `multi-task, multi-camera`를 고려한 네트워크 설계를 소개합니다. (`NVAutoNet`) 여기서 3D 정보는 인식 대상의 크기, 방향, 위치 뿐 아니라 parking space 및 free space 등을 의미합니다.
- 논문에서 제안하는 네트워크의 컨셉은 `End-to-End`이므로 별도의 post-processing을 필요로 하지 않으며 한번에 전체를 학습하는 방식을 이용합니다.
- `NVAutoNet`은 `NVIDIA Orin SoC`에서 `53 fps`의 수행 속도를 가지며 `NVIDIA ORGIN`의 스펙은 다음 링크에서 참조 가능합니다. 아래 사이트에서는 `ORIN`의 한국 총판을 맡고 있는 듯 합니다. 
    - 링크 : https://www.mdstech.co.kr/AGXOrin
- `NVAutoNet`은 실제 차량에서도 잘 동작함을 장점으로 꼽고 있습니다. 논문의 뒷편에 일반 차량과 트럭 모두에 적용할 수 있음을 보여주는데 카메라 장착 위치의 변화를 카메라 캘리브레이션 파라미터를 입력받아서 학습 및 테스트 시 사용하여 적절한 `fine-tuning` 했음을 보여줍니다.

<br>

## **1. Introduction**

<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 멀티 카메라가 있는 차량 환경에서도 단일 카메라에서 독립적으로 주변 환경을 인식하는 방식을 사용해 왔었습니다. 이와 같은 경우에는 각각의 카메라에서 인지한 결과에 에러가 발생하면 각 카메라 결과를 퓨전하는 데 어려움이 발생하나 이와 같이 카메라를 독립적으로 운용 (`camera independence`)하는 방법이 제품의 안정성 등과 연관되어 단점이 있음에도 불구하고 사용되어 왔습니다.
- 제품의 확장성 (`production scalability`) 측면도 중요한 요소입니다. 차량의 종류 별로 카메라 장착 사양이 다르기 때문에 학습된 네트워크가 다양한 차량에 대응할 수 있도록 설계되어야 합니다. 본 논문에서는 카메라 사양으로 `camera mounting angles`, `positions`, `different radial distortion`, `focal length`를 언급하였고 앞의 2개는 `camera extrinsic`에 해당하고 뒤의 2개는 `camera intrinsic`에 해당하는 것을 알 수 있습니다.
- 마지막으로 저전력이면서 `real-time` 성능이 나올 수 있도록 설계를 해야함을 언급합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 `NVAutoNet`은 5가지의 목표인 ① `accurate 3D estimation`, ② `camera independence`, ③ `end-to-end leaning`, ④ `scalability`, ⑤ `efficiency` 를 달성하기 위하여 전체 구조를 설계 합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이와 같은 네트워크 설계를 위하여 크게 4가지를 고려합니다. 
- ① `NAS`를 이용한 `image → BEV feature extract` 네트워크를 설계합니다.
- ② `feature level`에서 멀티 카메라 퓨전을 하여 early fusion과 late fusion을 결합한 `mid-level fusion`을 사용합니다.
- ③ `2d - to 3d uplift` 시 `depth prediction` 기반의 방법을 사용하지 않고 `MLP` 기반의 `uplift` 방법을 사용합니다. 이 때, `camera intrinsic/extrinsic` 정보를 입력값으로 사용합니다.
- ④ 모든 task는 `detection task`와 같이 구조화 되어 별도 post-processing을 필요하지 않도록 정의합니다.

<br>

## **2. Related Work**

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 설명하였듯이 `NVAutoNet`은 `Multi-camera`의 정보를 `early fusion` 방식을 고려하여 단일 네트워크로 전달하며 `BEV representation` 상에서 최종 출력을 생성해 냅니다.
- 따라서 `Multi-camera`를 `fusion` 하는 방식은 어떻게 각 이미지의 정보를 `BEV representation`으로 나타내는 것인 지와 밀접한 연관이 있습니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/8.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `2.2 Perspective to 3D View Transformation.`에서는 본 논문의 `image fusion`의 핵심인 `perspective image` 를 `BEV`로 변환하는 방식에 대한 연관 내용을 소개합니다.
- `vision` 기반의 `BEV` 표현 방식에 대한 내용은 아래 링크에서도 확인할 수 있습니다.
    - [Vision-Centric BEV Perception (Survey)](https://gaussian37.github.io/vision-fusion-vision_centric_bev_perception/)

<br>

- `View Transformation`을 하는 방식에는 크게 3가지 ① `depth-based approach`, ② `ML based approach`, ③ `transformer based approach`가 있습니다.
- 본 논문에서 제안하는 방식은 `ML based approach`를 경량화 한 방식을 사용합니다.

<br>

- 먼저 ① `depth-based approach`은 가장 고전적인 방법으로 `flat-world model`을 가정하거나 이미지의 픽셀 별 `depth` 정보를 알 수 있는 경우 사용 가능합니다. 하지만 `depth` 정보가 부정확하면 `3d uplifting (2D image → BEV representation)` 결과가 부정확해지는 문제가 발생하고 2D 이미지의 pixel 단위 별로 `3d uplifting`을 하게 되면 전체 픽셀 변환량이 많아서 비효율적으로 변환을 해야 합니다. 뿐만 아니라 원거리 영역에서의 `depth`가 부정확하기 때문에 원거리 영역의 `3d uplifting`이 더 나빠지게 됩니다.

<br>

- 다음으로 ② `ML based approach`은 `MLP (Multi-layer Perceptron)`을 사용하여 `3d uplifting`를 하며 BEV 관련 논문에서 많이 등장하는 VPN, FISHNET 등에서 이와 같은 방법을 사용하였습니다. 하지만 기존에 사용하였던 방식은 카메라의 intrinsic, extrinsic 정보가 사용되지 않아 한번 학습된 카메라 셋팅 환경과 달라지면 사용할 수 없다는 단점과 `MLP` 특유의 계산량이 많다는 단점이 존재합니다.
- 본 논문에서는 이러한 계산량 문제를 개선하기 위하여 `MLP` 연산을 각 `column` 방향으로 독립적으로 연산하는 방식을 사용합니다.

<br>

- 마지막으로 ③ `transformer based approach`가 있으며 이 방법은 `3d uplifting`을 위하여 transformer 아키텍쳐를 적극적으로 사용하는 방법을 의미합니다. 앞의 `MLP based approach`보다 좀 더 복잡도가 증가한 방법입니다.
- `transformer based approach`는 궁극적으로 실제 차량 환경에서 `real-time`으로 동작하기 어려운 문제를 위 글에서는 문제로 삼고 있습니다. `sparse set of BEV queries` 문제는 실제 모든 pixel에 대하여 transformer를 적용하였을 때, 계산량이 너무 복잡하여 `attention mechanism`을 적용할 pixel을 전체 pixel의 subset으로 구성해야 하는 것을 의미하는데, 이 경우 `BEV representation`과 같은 `dense prediction`에는 적합하지 않음을 의미합니다. 
- 따라서 `sparse set of BEV queries`를 사용 시 `BEV representation`에 충분하지 않다는 단점과 `transformer` 구조가 가지는 계산 비용에서 오는 한계점으로 인하여 실차 환경에 적합하지 않음을 설명합니다.

<br>

- 따라서 본 논문에서는 `MLP` 연산을 각 `column` 방향으로 독립적으로 연산하는 방식을 집중적으로 설명할 예정입니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/9.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 2.1과 2.2의 설명을 통해 `Multi-camera fusion`을 통하여 `BEV` 상에 정보를 표현하고자 함을 확인하였습니다. 2.3에서는 `NVAutoNet`이 출력하고자 하는 정보가 `obstacles`, `free/parking spaces`임을 말해줍니다.

<br>

- 기존의 `3D Object Detection`은 2D 이미지에서 3D Bounding Box를 추출하는 Task이지만 `depth` 정보의 정확도 문제로 한계점이 있습니다.
- 본 논문에서는 2D 이미지에서 3D 정보를 추출할 때, `image space`가 아닌 `BEV space`를 바로 이용하는 방식을 사용합니다. 실차 환경의 구현을 위해 transfomer를 사용하지 않고 `fully convolutional layer`을 사용하고 3D object 정보를 추출하기 위하여 object의 3D 정보를 알기 위한 파라미터를 추론할 뿐 아니라 그 파라미터의 불확실성 (uncertainty) 까지 추론해 냅니다. `BEV space`에서 정보를 추론하지만 `roll`, `pitch`, `yaw` 모두의 방향에 대한 정보를 추론하는 것을 언급합니다.

<br>

- `Free space detection`을 할 때에도 2D 이미지에서 정보를 찾는 기존의 방식을 사용하지 않고 `BEV`에서 찾는 방식을 사용합니다. 따라서 `BEV` 공간에서 `polygon` 형태로 `free space`를 나타내는 방식을 통해 별도의 post-processing을 거치지 않는 방식을 취합니다.

<br>

## **3. NVAutoNet**

<br>

- 챕터 3에서는 2D image feuatre를 최종적으로 어떻게 fusion 하는 지 집중적으로 다룹니다. 

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/10.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `NVAutoNet`은 360◦ FOV를 가질 수 있도록 $$ N $$ 개의 카메라와 각 카메라의 `intrinsic`, `extrinsic`을 입력값으로 사용합니다.
- `2D Image Encoder`는 2D image에서 feature를 추출하는 역할을 하며 각 `2D image feature`는 `BEV feauture`로 만들어진 후 최종적으로 `BEV Encoder`로 보내집니다. `BEV Encoder`는 최종 출력인 `obstacle (obejct)`의 `position, size, orientation`과 `parking/free spaces`를 출력합니다. 이와 관련된 전체 과정은 아래 내용을 다시 살펴보시면 됩니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/11.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/12.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `2D image`는 feature extractor를 통하여 `multi level feature` $$ F_{k} $$ 를 추출하며 $$ F_{k} $$ 의 shape은 다음과 같이 여러 해상도의 크기를 가집니다.

<br>

- $$ \text{shape of } F_{k} = \frac{H}{2^{k+1}} \times \frac{W}{2^{k+1}} \times C $$

<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/13.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 Table 2의 `Channels` 정보를 보면 $$ k = 0, 1, 2, 3, 4 $$ 로 5개가 적용된 것으로 추정 됩니다.
- 위 테이블에서의 각 `convolution`의 파라미터는 `NAS (neural architectural search)`를 통해 추정하였으며 논문에서는 latency와 accuracy를 고려하여 찾은 것으로 설명하였습니다.
- `latency` 개선을 위하여 `residual connection`을 제거되었으며 (성능 감소를 감수한 것으로 추정됩니다.) 각 level의 `multi-level feature`를 `U-Net` 또는 `FPN` 구조와 같이 잘 조합하여 좋은 feature를 구성한 것으로 추정됩니다.
- (위 테이블에서 각 카메라 별 인풋은 동일하지만) 서로 다른 카메라의 인풋 사이즈의 크기가 다르더라도 `2D Image Feature Extractor`가 각 카메라 별 이미지에 공용으로 사용할 수 있도록 설계하였음을 설명합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/14.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `BEV plane`은 차량의 중심을 지나면서 `Z` 축에 직교하도록 구성됩니다. 본 논문에서 `BEV grid` $$ G^{\text{bev}} $$ 는 $$ W^{\text{bev}} \times H^{\text{bev}} $$ 크기의 `dimension`을 가지며 `grid`의 셀 크기 및 표현 방식은 뒷 부분에서 설명합니다.
- `BEV grid`의 중심은 자차의 중심과 일치하도록 구성합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/15.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/16.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/17.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/18.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/19.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `BEV transformation function`인 $$ B(\cdot) $$ 은 단순한 1개의 `hidden layer`를 가지는 `MLP block`이며 각 이미지의 column 별 다른 `BEV tranformation` 방법을 가지기 때문에 이미지의 column 별 $$ B(\cdot) $$ 독립적인 function이 존재합니다.
- `MLP`를 사용하는 것은 `attention`의 목적과 같으며 `global contextual information`을 잘 사용하여 `image feature`를 `BEV` 상의 올바른 위치에 위치 시키도록 하는 것에 목적이 있습니다. (직접적인 `depth`를 추정하는 부분이 없으므로 object의 높이 부분이 있다면 그 부분을 반영해야 하기 때문입니다.)
- 이와 같은 $$ B(\cdot) $$ 을 학습하기 위한 직접적인 `supervision loss`는 없으며 최종 `Loss`를 통해 이 부분이 학습되도록 구성합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/20.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `NVAutoNet`의 최대 200m 까지 인식하고자 하며 이와 같은 인식 거리를 `grid` 셀의 크기를 0.25 m로 설정하여 `Cartesian grid`로 구성한다면 메모리도 많이 차지할 뿐 아니라 계산 비용도 너무 커지는 문제가 발생합니다.
- 따라서 앞에서 설명한 바와 같이 `Polar Coordinate`를 사용합니다. `Polar Coordinate`는 `angle`과 각 `angle`의 `depth`를 사용합니다.
- `angle`에 해당하는 $$ W^{\text{bev}} $$ 는 360의 크기를 가지도록 설정하며 이는 360◦를 의미합니다. `depth`에 해당하는 $$ H^{\text{bev}} $$ 는 64의 크기를 가지도록 설정하며 가까운 `depth`는 조밀하게, 먼 `depth`는 간격을 도어 설정하도록 하였습니다. `Cartesian grid`에서는 균일한 grid를 설정하는 것과 대조적입니다.
- 따라서 `NVAutoNet`에서의 `BEV grid`인 $$ G^{\text{bev}} $$ 는 $$ 360 \times 64 $$ 의 크기를 가집니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/21.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/22.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `BEV Encoder`의 `input size`는 앞의 `polar coordinate`를 반영하여 $$ 360 \times 64 $$ 가 되며 위 테이블과 같은 형태로 `convolution`을 이용하여 구성됩니다.
- `BEV Encoder`를 이용하여 추출된 feature인 $$ \hat{F}_{\text{bev}} $$ 를 3가지 3D detection 태스크인 `3D Object Detection`, `3D Freespace Detection`, `3D Parking space Detection`에서 사용합니다.

<br>

## **4. Perception Tasks**

<br>

- 이번 챕터에서는 앞에서 정의한 `BEV feature fusion` 정보를 이용하여 `3D Object Detection`, `3D Freespace`, `3D Parking Space` 기능을 구현하기 위한 출력과 학습 방법을 소개합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/23.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `3D Object Detection`에서 구하고자 하는 `3D cuboid`는 총 9개의 파라미터이며 `position`, `dimension`, `orientation`의 각 3 자유도 씩 총 9개의 자유도를 구하는 것에 해당하며 NMS와 같은 별도 후처리를 필요로 하지 않습니다.
- `BEV Encoder`를 통해 추출한 $$ \hat{F}_{\text{bev}}, \text{shape : } M \times N \times C $$ 에서 추가적인 `convolution`을 통하여 `3D cuboid` 정보를 추출합니다. 
- BEV feature의 $$ M \times N $$ 는 `spatial dimenstion`으로 `grid cell`의 크기에 대응되며 각 `grid` 별 객체를 추정하므로 총 $$ \hat{K} = M \times N $$ 개의 객체를 추정할 수 있습니다.
- `3d cuboid`를 추정하기 위하여 별도의 `head`를 추가하여 아래와 같이 `classification`, `position`, `dimension`, `orientation`을 각각 추정합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/24.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `Position`의 `azimuth angle`과 `elevation`은 아래 표시를 참조하시면 됩니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/24_1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/25.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 학습을 할 때에는 총 $$ K $$ 개의 GT와 $$ \hat{K} $$ 개의 prediction이 있을 때, 먼저 GT와 Pred를 매칭한 다음에 Loss를 계산합니다. 매칭을 잘못하면 Loss 계산이 잘못되므로 매칭 문제도 성능에 큰 영향을 미칩니다.
- 논문에서는 `classification`, `position`, `dimension`, `orientation` 중 `position`에 더 큰 가중치를 두어서 매칭하고 BEV grid 기준으로 GT와 Pred가 멀리 떨어져있으면 매칭하지 않는 간단한 greedy matching 방식을 사용하였음을 설명합니다. 이와 같은 방법을 통해 헝가리안 알고리즘과 같은 방식을 사용하지 않고 효과적으로 GT와 Pred를 매칭 하였습니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/26.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `classification`, `position`, `dimension`, `orientation`에 대한 각 Loss는 위 식을 참조하면 됩니다. 
- `Loss`의 분모에 $$ \sigma $$ 는 각 구하고자 하는 값의 `uncertainty`를 의미하며 이 `uncertainty`는 별도 출력을 구할 수 있도록 모델을 설계하여 값을 도출해 냅니다. 
- 컨셉은 간단합니다. `uncertainty`가 높으면 말 그대로 불확실하므로 이 `Loss`가 정확하게 계산되었는 지 아닌 지 알 수 없음을 의미합니다. 따라서 `uncertainty`가 높아지면 `Loss`가 작아지도록 하여 잘못 학습이 되는 것을 방지합니다. 하지만 무한정 `uncertainty`가 높아지게 되면 학습이 전혀 되지 않을 우려가 있으므로 $$ \log{(2\sigma)} $$ 를 `regularization term`으로 추가하여 `uncertainty`가 무한정으로 커지는 것을 방지합니다.
- 반면 `uncertainty`가 낮아지면 그만큼 신뢰하고 `Loss`를 사용할 수 있다는 뜻이므로 `Loss`를 더 크게 만들어 학습에 반영합니다.
- 이와 같은 컨셉으로 모든 `Loss` term에 $$ \sigma $$ 가 적용되어 있습니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/27.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `3D Freespace`의 인식 대상은 `vehicle`, `VRU (vulnerable road users)`, `Others(random hazard obstacels + static obstacles)`이며 인식 대상은 `RDM (Radial Distance Map)`에 나타냅니다.
- `VRU`의 용어 정의는 **pedestrians, motorcycle riders, cyclists, children 7-years and under, the elderly and users of mobility devices**이며 상세 내용은 다음과 링크를 참조하시면 됩니다.
    - 링크 : [https://www.roadsafety.gov.au/nrss/fact-sheets/vulnerable-road-users](https://www.roadsafety.gov.au/nrss/fact-sheets/vulnerable-road-users)

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/28.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `RDM` 에서는 `equaiangle`과 `radial distance`를 이용하여 영역을 나누고 자차와 각 `angle` 별 자차와 가장 가까운 `radial distance`의 경계를 찾는 방식으로 `3D Freespace`를 인식합니다.
- 따라서 `BEV plane`의 중심인 자차에서 부터 360도 전체 각도 별로 가장 가까운 `radial distance`를 찾고 그 점들을 이어서 `polygon` 형태로 만드는 것이 최종 목적이 됩니다. 이와 같은 방식을 사용하면 추가적인 post-processing 없이 3D boundary를 만들 수 있습니다.
- `polygon`을 형성할 때 필요한 점들이 `label`이 되며 라벨은 $$ (r, c) $$ 형태로 표현할 수 있습니다. $$ r $$ 은 `radial distance vector`이며 $$ c $$ 는 `boundary semantic vector` 입니다. 즉, `RDM` 에서의 3D Boundary가 표시되는 점의 위치와 클래스가 됩니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/29.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `3D Freespace Detection`을 위한 추가적인 layer는 `shared neck`과 `two seperated head`이며 각 `head`는 `radial distance`와 `classification map`을 추정합니다. 앞의 `3D Detection` 케이스와 동일하게 `shared neck`은 $$ \hat{F}_{\text{bev}} $$ 를 입력으로 받아 처리합니다.

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/30.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

## **5. Multi-task Learning and Loss Balancing**

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/31.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

## **6. Experimental Evaluation**

<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/32.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/33.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/34.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/35.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/36.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/37.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/38.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/39.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/40.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/41.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/42.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/43.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/44.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/45.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/fusion/nvautonet/46.png" alt="Drawing" style="width: 600px;"/></center>
<br>


<br>

[fusion 관련 글 목차](https://gaussian37.github.io/vision-fusion-table/)

<br>
