---
layout: post
title: DeepLab v3 (Rethinking Atrous Convolution for Semantic Image Segmentation)
date: 2019-11-04 00:00:00
img: vision/segmentation/deeplabv3/0.png
categories: [vision-segmentation] 
tags: [segmentation, deeplab v3+, deeplab, deeplab v3] # add tag
---

<br>

- 참조 : https://arxiv.org/abs/1706.05587
- 참조 : https://medium.com/free-code-camp/diving-into-deep-convolutional-semantic-segmentation-networks-and-deeplab-v3-4f094fa387df
- 참조 : https://towardsdatascience.com/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch-f770863d6a42

<br>

- 이번 글에서는 Semantic Segmentation 내용과 DeepLab v3 내용에 대하여 간략하게 알아보도록 하겠습니다.
- DeepLab v3의 핵심은 `ASPP (Atrous Spatial Pyramid Pooling)`이며 이 개념의 도입으로 DeepLab v2 대비 성능 향상이 되었고 이전에 사용한 추가적인 Post Processing을 제거함으로써 End-to-End 학습을 구축하였습니다.

<br>

## **목차**

<br>

- ### Semantic Segmentation
- ### Model Architecture
- ### ResNets
- ### Atrous Convolutions
- ### Atrous Spatial Pyramid Pooling
- ### Implementation Details
- ### Results

<br>