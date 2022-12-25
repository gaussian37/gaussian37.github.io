---
layout: post
title: How Do Neural Networks See Depth in Single Images 리뷰
date: 2022-11-26 00:00:00
img: vision/depth/how_nn_see_depth/0.png
categories: [vision-depth]
tags: [depth estimation, neural network, depth, cvpr] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 논문 : https://openaccess.thecvf.com/content_ICCV_2019/papers/van_Dijk_How_Do_Neural_Networks_See_Depth_in_Single_Images_ICCV_2019_paper.pdf

<br>

- 이번 글에서는 Depth Estimation을 위한 뉴럴 네트워크가 단일 이미지에서 물체를 어떻게 인식하는 지 분석한 논문에 대하여 살펴보도록 하겠습니다. 논문 제목은 `How Do Neural Networks See Depth in Single Images` 입니다.

<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [Introduction](#introduction-1)
- ### [Related Work](#related-work-1)
- ### [Position Vs. Apparent Size](#position-vs-apparent-size-1)
- ### [Camera Pose: Constant or Estimated](#camera-pose-constant-or-estimated-1)
- ### [Obstacle Recognition](#obstacle-recognition-1)
- ### [Concolusion and Future Work](#concolusion-and-future-work-1)

<br>

## **Abstract**

<br>

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이번 글에서 다루는 논문은 뉴럴 네트워크를 이용한 Depth Estimation의 성능이 향상이 되었지만 어떻게 깊이를 추정하는 지에 대한 자세한 분석이 없어 그 분석에 대한 연구를 다룹니다.
- 논문에서 사용하는 모델은 [monodepth2](https://gaussian37.github.io/vision-depth-monodepth2/)이며 이 모델을 기반으로 뉴럴 네트워크가 깊이를 추정할 때 민감하게 반응하는 `visual cues`를 살펴봅니다.
- 그 결과 뉴럴 네트워크가 객체의 이미지 상의 `vertical position`에 민감한 것을 확인하였으며 객체와 지면과의 접촉면에서의 `edge`가 중요함을 확인하였습니다.

<br>

## **Introduction**

<br>

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- CNN 기반의 Depth Estimation 연구가 시작되면서 Depth Estimation의 성능이 많이 발전하였으며 처음에는 Supervised Learning 방식의 연구가 진행되었으나 본 논문의 저자인 Godard의 monodepth1, monodepth2의 발전으로 Self-supervised Learning 방식의 Depth Estimation이 연구로 방향이 전환되었습니다.

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 본 논문의 연구 바향은 뉴럴 네트워크가 이러한 좋은 성능을 내기 위해서 어떤 방향으로 학습하는 지 알기 위함으로 접근되었습니다.
- 이러한 접근 방식이 중요한 이유는 ① Depth Estimation 뉴럴 네트워크의 Visual Cue와 학습 메커니즘을 모르면 예상치 못한 시나리오에서의 동작 방식을 보장할 수 없고 ② 학습 메커니즘의 파악을 통하여 실제 학습 시 도움을 줄 수 있으며 ③ 카메라 장착 위치의 변화에 따른 부작용을 예상할 수 있기 때문입니다.

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 본 논문의 본문은 크게 3가지 파트로 구분되어 있습니다.
- `Section Ⅲ`에서는 monodepth 모델이 객체를 인지할 때, 객체의 크기가 아니라 이미지에서의 `vertical position`에 의존적임을 보여줍니다.
- `Section Ⅳ` 에서는 monodepth 모델이 카메라 포즈를 고정된 값으로 여기는 지 살펴봅니다.
- `Section Ⅴ` 에서는 monodepth 모델이 어떻게 객체와 배경을 구분하는 지와 학습 데이터셋에 없는 개체를 인지할 때 어떤 요소가 작용해서 객체를 인지하는 지 살펴봅니다.

<br>


## **Related Work**

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 뉴럴 네트워크의 동작 방식을 확인하기 위하여 feature visualization을 이용하는 방법이 대중적인 방법인 반면 이번 논문에서는 입/출력 간의 상관 관계를 확인하기 위하여 특정 `visual cue`를 추가하였을 때, 결과가 어떻게 바뀌는 지 확인하고 이 결과의 상관관계를 살펴보는 방식으로 분석합니다.
- 위 8개 항목에 대하여 분석하고자 하였으나 KITTI 데이터셋 환경에서의 제한으로 `Position in the image`, `Apparent size of objects`에 대하여 집중 분석하였습니다.

<br>

## **Position Vs. Apparent Size**

<br>

- 먼저 객체의 위치 (Position)과 크기 (Size)가 깊이를 추정하는 데 어떤 영향이 있는 지 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 위치와 크기를 모두 변경하거나 위치만 변경 또는 크기만 변경한 경우로 나누어서 분석하였으며 크기의 변경은 Relative distance가 1.0, 1.5, 3.0인 경우로 나누어서 적용하였습니다. 첫 행의 Position and scale이 위치에 따라 크기도 변경되는 일반적인 경우에 해당합니다.

<br>
<center><img src="../assets/img/vision/depth/how_nn_see_depth/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `true relative distance`와 `expected relative distance`와의 관계를 확인하였을 때, `position and scale`의 expected relative distnace가 true relative distance와 가장 비슷하였습니다. 그 다음으로 `position only`가 비슷하였으며 에러가 더 커졌지만 유사한 수준으로 보입니다.
- 반면 `scale only`의 경우 distance 변화가 거의 없어보이며 에러가 점점 커지는 것을 볼 수 있습니다. 즉, relative distance가 1.0인 경우를 그대로 사용하는 것으로 보입니다.
- 따라서 `position`이 깊이를 추정하는 데 큰 역할을 하는 것으로 확인할 수 있습니다.

<br>


## **Camera Pose: Constant or Estimated**

<br>

<br>


## **Obstacle Recognition**

<br>

<br>


## **Concolusion and Future Work**

<br>

<br>





<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>
