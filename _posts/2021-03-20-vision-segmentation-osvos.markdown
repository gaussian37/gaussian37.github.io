---
layout: post
title: OSVOS(One-Shot Video Object Segmentation)
date: 2021-03-20 00:00:00
img: vision/segmentation/osvos/0.png
categories: [vision-segmentation] 
tags: [vision, deep learning, osvos, one shot video object segmentation, vos] # add tag
---

<br>

- 논문 : https://arxiv.org/abs/1611.05198
- 공식 페이지 : https://cvlsegmentation.github.io/osvos/
- 깃헙 : https://github.com/kmaninis/OSVOS-PyTorch
- 참조 : https://eungbean.github.io/2019/07/03/OSVOS/

<br>

- 이번 글에서는 `Video Object Segmentation` 관련 논문에서 가장 인용수가 많고 성능 육성에 큰 영향을 준 OSVOS, One-Shot Video Object Segmentation (S. Caelles,K.-K. Maninis, CVPR 2017)에 대하여 알아보도록 하곘습니다.
- 먼저 Video Object Segmentation 문제의 정의는 동영상에서 **특정 물체를 연속적으로 세그멘테이션** 하는 작업을 뜻합니다. 이와 유사하게 Sementic Segmentation은 동영상 또는 이미지에서 특정 물체가 아닌 배경을 포함한 모든 물체를 대상으로 세그멘테이션하는 것입니다. 따라서 Video Object Segmentation과 Sementic Segmentation에는 세그멘테이션 하는 대상에 차이가 있습니다.

<br>

## **목차**

<br>

- ### Introduction(#introduction-1)
- ### One Shot Deep Learning(#one-shot-deep-learning-1)
- ### Contour snapping(#contour-snapping-1)

<br>

## **Introduction**

<br>

- OSVOS 논문에서는 One-Shot 방법을 적용하여 Video Ojbect Segmentation 하는 방법에 대하여 다룹니다. 딥러닝에서 One shot이란 한 번 타겟을 보고 작업을 수행하는 것을 뜻합니다.

<br>
<center><img src="../assets/img/vision/segmentation/osvos/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이 논문에서 다루는 One Shot의 의미는 **첫 프레임(또는 그 이상의 프레임)에서 찾고자 하는 물체에 마스크를 제공**하면 나머지 프레임에서 같은 물체를 찾아내는 방법을 뜻합니다. 이러한 방식의 VOS를 `semi-supervised video object segmentation` 이라고 합니다. 위 예제를 참조하시기 바랍니다.

<br>

- 논문에서는 주요 Contribution으로 다음 3가지를 설명합니다.

<br>
<center><img src="../assets/img/vision/segmentation/osvos/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 첫 프레임의 Object를 Segment해주기만 하면 나머지 프레임에서도 물체를 찾아낼 수 있습니다.

<br>
<center><img src="../assets/img/vision/segmentation/osvos/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 딥러닝을 사용하면 각 프레임을 독립적으로 연산하더라도 `Temporal Consistency`를 얻을 수 있어서 Occlusion등에 강건한 장점을 가집니다.
- 여기서 `Temporal Consistency`는 동영상에서 연속적인 시간에서 일관성을 가질 수 있도록 하는 것을 뜻합니다.

<br>
<center><img src="../assets/img/vision/segmentation/osvos/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 마지막으로 속도와 성능간의 Trade-off가 자유롭도록 모델을 설계할 수 있다는 점입니다.

<br>

## **One Shot Deep Learning**

<br>
<center><img src="../assets/img/vision/segmentation/osvos/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- OSVOS는 크게 3가지 단계를 이용하여 학습을 합니다. 위 그림과 같이 `Base Network`, `Parent Network`, `Test Network`가 이에 해당합니다.
- `Base Network` : 이미지넷에서 학습된 backbone 네트워크를 이용하여 영상의 feature를 얻습니다. 
- `Parent Network` : Base Network와 DAVIS 데이터 셋을 이용하여 영상에서 모든 픽셀에 대하여 물체와 배경을 분리하는 binary classification 네트워크를 학습합니다.
- `Test Network` : Parent Network를 통하여 배경으로 부터 분리된 Object들 중 특정 Object만을 세그멘테이션할 수 있도록 하는 네트워크 입니다.
