---
layout: post
title: KITTI 데이터셋
date: 2018-12-31 00:00:00
img: vision/dataset/kitti/0.png
categories: [vision-dataset] 
tags: [vision, kitti, dataset] # add tag
---

<br>

- 참조 : https://github.com/lkk688/3DDepth

<br>

- 이번 글에서는 컴퓨터 비전 Task로 많이 사용되는 KITTI 데이터셋에 대한 내용을 다루도록 하겠습니다.

<br>

## **Kitti data format**

<br>

- KITTI 데이터 센서셋 구성 : http://www.cvlibs.net/datasets/kitti/setup.php

<br>

- KITTI 데이터셋를 취득하기 위한 차량의 셋업은 아래와 같습니다.

<br>
<center><img src="../assets/img/vision/dataset/kitti/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림 중 유심히 볼 부분은 라이다와 카메라의 좌표축입니다. 라이다는 오른손 좌표계를 기준으로 엄지손가락이 Z축, 검지 손가락이 X축, 중지 손가락이 Y축에 해당합니다.
- 반면 카메라 좌표계에서는 이미지 좌표계와 동일한 방향으로 X, Y 축을 가지고 깊이 방향으로 Z 축을 가집니다.

<br>
<center><img src="../assets/img/vision/dataset/kitti/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 위 그림과 같은 좌표축을 가지게 됨을 알 수 있습니다.

<br>
<center><img src="../assets/img/vision/dataset/kitti/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 라이다와 카메라만 따로 떼어서 보면 위 그림과 같습니다.

<br>

- KITTI 데이터셋의 폴더 구조는 다음과 같습니다.
    - {training,testing}/image_2/id.png
    - {training,testing}/image_3/id.png
    - {training,testing}/label_2/id.txt
    - {training,testing}/velodyne/id.bin
    - {training,testing}/calib/id.txt

<br>

- 센서의 좌표축을 정리하면 다음과 같습니다.
    - `Camera`: x = right, y = down, z = forward
    - `Velodyne`: x = forward, y = left, z = up
    - `GPS/IMU`: x = forward, y = left, z = up

<br>

- 카메라는 접지면과 거의 수평으로 장착됩니다. 카메라 이미지는 `1382 x 512` 픽셀 크기로 잘립니다. 보정 (rectification) 후 이미지가 약간 작아집니다. 카메라는 라디아의 레이저 스캐너가 앞을 향할 때 `초당 10 프레임`으로 트리거됩니다.