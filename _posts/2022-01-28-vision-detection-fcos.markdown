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

- 이번 글에서는 `Anchor Free` 기반의 Object Detection 모델인 `FCOS`, Fully Convolutional One-Stage Object Detection 논문에 대하여 알아보도록 하겠습니다.
- 전체적으로 논문의 내용을 리뷰한 뒤, Pytorch 코드를 확인하는 순서로 알아보겠습니다.

<br>

## **목차**

<br>

- ### Abstract
- ### Introduction
- ### Related Work
- ### Approach
- ### Ablation Study
- ### Concolusion
- ### Pytorch Code

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

## **Related Work**

<br>

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



