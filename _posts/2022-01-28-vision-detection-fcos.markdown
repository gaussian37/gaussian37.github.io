---
layout: post
title: FCOS (Fully Convolutional One-Stage Object Detection)
date: 2022-01-28 00:00:00
img: vision/detection/fcos/0.png
categories: [vision-detection] 
tags: [vision, detection, fcos] # add tag
---

<br>

- 논문 : https://arxiv.org/pdf/1904.01355.pdf
- 코드 : https://github.com/tianzhi0549/FCOS
- 코드 : https://github.com/rosinality/fcos-pytorch

<br>

- 배경 지식 : [https://gaussian37.github.io/vision-segmentation-fcn/](https://gaussian37.github.io/vision-segmentation-fcn/)

<br>

- 이번 글에서는 `Anchor Free` 기반의 Object Detection 모델인 `FCOS`, Fully Convolutional One-Stage Object Detection 논문에 대하여 알아보도록 하겠습니다.
- 전체적으로 논문의 내용을 리뷰한 뒤, Pytorch 코드를 확인하는 순서로 알아보겠습니다.

<br>

## **목차**

<br>

- ### [Abstract](#abstract-1)
- ### [Introduction](#introduction-1)
- ### [Related Work](#related-work-1)
- ### [Approach](#approach-1)
- ### [Ablation Study](#ablation-study-1)
- ### [Concolusion](#concolusion-1)
- ### [Pytorch Code](#pytorch-code-1)

<br>

## **Abstract**

<br>

<br>
<center><img src="../assets/img/vision/detection/fcos/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- FCOS는 one-stage detector의 한 종류이며 다른 one-stage detector와의 차이점은 semantic segmentation과 유사하게 픽셀 단위의 예측을 통해 Object Detection 문제를 해결하고자 합니다.
- 특히 anchor box가 없다는 점에서 anchor free의 대표적인 방법으로 사용되고 있습니다. anchor box를 제거함으로써 FCOS는 복잡함 anchor box 관련 연산을 없앨 수 있었습니다. 또한 어떤 anchor box를 사용해야 하는 지 이 또한 고민거리이지만 anchor box를 없앰으로써 고려하지 않아도 됩니다.
- 정리하면 `FCOS`는 semantic segmentation과 유사한 방법의 픽셀 단위의 prediction 방법을 이용하는 **anchor free one-stage detector**이며 이 방법은 간단하면서도 성능이 좋고 다른 instance-level task에도 사용될 수 있습니다.

<br>

## **Introduction**

<br>

- Object Detection의 발전은 다양한 Anchor Box 기반의 모델과 함께 발전해 왔습니다. 즉, anchor box를 잘 사용하는 것이 detector의 성능을 높이는 중요한 역할이라고 믿어 왔습니다.
- 하지만 anchor box를 사용하는 모델에는 몇가지 단점이 있습니다. FCOS 논문에서는 크게 4가지 항목으로 문제를 지적하였습니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- ① Object Detection 모델의 성능이 anchor box에 민감하다는 것이 단점이 됩니다. 즉, anchor box의 크기, 비율, 갯수 등을 잘 선정해 주어야 좋은 성능을 기대할 수 있습니다. 하이퍼파라미터의 영역으로 anchor box가 남게 되어 사용하는 데 어려움이 있습니다.
- ② anchor box가 정해지면 학습할 때에는 보통 고정이 되는데, 어떤 물체의 크기의 변화가 크다면 anchor box가 효과적으로 사용되지 못할 수 있습니다. 즉, 물체의 변화가 커서 생각한 것과 많이 다른 크기의 형상을 가진다면 기존에 선정한 anchor box의 비율과 맞지 않게 되어 검출을 할 수 없습니다. 특히, 이러한 경향은 작은 물체에 대하여 종종 나타납니다.
- ③ 높은 recall 수치를 얻기 위해서는 다양한 갯수의 anchor box가 필요해 집니다. 어떤 경우에는 굉장히 dense한 형태로 anchor box의 경우의 수를 다양화하여 180,000 개 정도를 사용하기도 합니다. 이와 같은 경우의 문제점은 대부분의 경우가 negative sample로 분류되기 때문에 학습 시, positive와 negative간 갯수의 불균형이 심하게 발생할 수 있다는 점입니다.
- ④ anchor box를 이용한 GT와의 IoU (Intersection over union) 계산 비용이 별도로 필요해 집니다.

<br>

- 위 4가지의 anchor based 모델의 단점이 확인된 가운데 FCN (Fully Convolutional Network)과 같은 구조를 이용하여 semantic segmentation, depth estimation, keypoint estimation 등의 task에 좋은 성능을 보여 주어 object detection 또한 자연스럽게 이와 같은 방법 (per-pixel prediction)을 이용하고자 하는 시도가 FCOS를 통해 시도 되었습니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- FCN 구조를 이용하여 object detection을 할 때, semantic segmentation과 같이 단순히 픽셀 별 classification 만을 하는 것이 아니라 픽셀 별로 4개의 원소값을 가지는 4D vector를 추가적으로 가지게 됩니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림의 왼쪽의 야구 이미지와 오른쪽의 테니스 이미지를 분리하여 살펴보겠습니다.
- 먼저 왼쪽의 야구 이미지를 보면 박스 내의 어떤 점을 기준으로 상/하/좌/우의 경계 부분까지의 거리를 추정하게 됩니다.
- 이와 같은 방법을 이용하여 박스를 추정하였을 때, 한가지 문제가 생기는데 바로 오른쪽 테니스 같은 이미지가 예입니다. 만약 어떤 하나의 위치를 기준으로 여러개의 상/하/좌/우의 경계를 추정하려면 어떻게 해야 할까요? 각 픽셀당 1개의 상/하/좌/우 위치를 추정하기 때문에 이와 같은 예시는 모호해질 수 있습니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 오버랩이 되는 영역에서의 모호함을 개선하기 위하여 `FPN (Feature Pyramid Network)` 구조를 사용합니다. Feature Pyramid 구조를 통하여 다양한 크기의 Feature를 사용할 수 있습니다. 이 내용은 이 글의 Approach 부분에서 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- FCOS에서는 `center-ness` 라는 개념을 도입하여 실제 물체의 중심점으로 부터 멀리 떨어져 있게 바운딩 박스를 예측한 경우를 제한하도록 하였습니다. center-ness의 구체적인 개념은 Approach에서 알아볼 예정입니다. 이러한 center-ness의 도입으로 ancor 기반의 모델보다 더 좋은 성능을 가질 수 있을 수 있었습니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- Introduction에서 설명한 내용에 대하여 정리해 보도록 하겠습니다.
- ① semantic segmentation 문제를 푸는 FCN 구조를 이용하여 Detection에 접목할 수 있고 이러한 아이디어를 이용하여 다른 Task에도 접목할 수 있습니다.
- ② FCOS는 Region Proposal과 Anchor 모두가 free한 one-stage anchor free 모델입니다. 이러한 모델의 구조는 추가적인 하이퍼파라미터 튜닝 없이 구조가 간단하다는 장점이 있습니다.
- ③ Anchor의 제거는 Anchor box의 IoU 연산 제거와 Anchor box와 GT 간의 비교 연산을 없앨 수 있습니다.
- ④ FCOS는 one-stage detector 중에서 성능이 좋으며 FCOS의 결과는 two-stage detector의 Region Proposal로 사용할 수 있습니다.
- ⑤ 구조를 조금 수정하면 다른 task에 접목할 수 있고 특히 instance 단위의 prediction을 할 때 좋은 방법이 될 수 있습니다.

<br>

## **Related Work**

<br>

- `Anchor-based Detector` : Region Proposal의 반복적인 연산을 제거하기 위하여 Anchor Box 개념을 도입하여 Faster-RCNN, SSD, YOLOv2 등이 개발이 되어 왔습니다. 앞에서 언급한 바와 같이 Anchor를 잘 선정하기 위한 튜닝에 어려움이 있으며 수많은 Anchor들이 Negative Sample로 빠지게 되는 단점이 있습니다.
- `Anchor-free Detector` : 기존에 유명한 Anchor-free Detector는 YOLOv1이 있었습니다. 하지만 YOLOv1은 Recall 성능이 떨어진다는 단점이 있어서 YOLOv2에서 부터는 Anchor Box를 사용하게 되었습니다. `FCOS`는 YOLOv1과 같이 GT bounding box 이내의 모든 점들에 대하여 bounding box를 예측하도록 하고 low-quality를 가지는 점들에 대해서는 `center-ness` 개념을 도입하여 출력되지 않도록 억누르는 역할을 합니다. 이러한 방법으로 recall의 성능을 높일 수 있도록 하였습니다.

<br>
<center><img src="../assets/img/vision/detection/fcos/8.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 기존의 Anchor-freee Detector들은 Post Processing이 다소 복잡하거나 겹치는 bounding box를 처리하는 문제 또는 recall이 상대적으로 낮은 문제가 있었습니다. 하지만 `multi-level FPN` 구조와 `center-ness`를 통하여 이 문제를 개선하였고 더 간단한 구조로 구현할 수 있었습니다.

<br>

## **Approach**

<br>



<br>

## **Ablation Study**

<br>

<br>

## **Concolusion**

<br>

<br>

## **Pytorch Code**

<br>

<br>



