---
layout: post
title: Fully Convolutional Networks for Semantic Segmentation
date: 2019-08-21 00:00:00
img: vision/segmentation/fcn/fcn.jpg
categories: [vision-segmentation] 
tags: [vision, segmentation, fcn] # add tag
---

- 이번 글에서 다루어 볼 논문은 `FCN`으로 유명한 **Fully Convolutional Networks for Semantic Segmentation** 입니다.

<br>

## **Abstract**

<br>

- CNN으로 end-to-end, pixels-to-pixels로 학습한 Semantic Segmentation 모델이 좋은 성능을 보였습니다.
- 여기서 이 논문의 핵심은 **FCN(Fully Convolutional Network)** 가 임의의 사이즈의 이미지를 입력으로 받아서 그것에 상응하는 사이즈의 아웃풋을 만들어 내는 것에 있습니다.
  - 첨언하면 기존의 CNN 기반의 작업에서는 고정 입력, 고정 출력이 전제였는데 그 점을 개선한 것입니다.
- 논문에서 다루는 내용은 **FCN**에 대한 정의와 그것에 대한 상세화를 하고 **FCN**이 어떻게 공간상에 prediction 하는 작업을 하는 지 설명합니다. 그리고 기존에 사용되었던 딥러닝 모델과 연결해 보려고 합니다.
- 여기서는 현대 classification network를 FCN에 접목시켜 보았습니다. 그리고 transfer learning을 사용하였는데 기존에 학습된 representation을 segmentation 작업에 fine tuning 작업을 거쳤습니다.
- 그리고 **skip architecture** 구조에 대하여 설명을 하였는데, 이 구조는 깊은 layer와 얕은 layer를 결합하여 정확도와 디테일한 segmentation 작업에 도움을 줍니다.
- 결과적으로 이 논문은 그 당시에 segmentation 작업에 좋은 성능을 내었었습니다.

<br>

## **Introduction**

<br>

