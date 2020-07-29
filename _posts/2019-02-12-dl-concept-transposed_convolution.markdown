---
layout: post
title: transposed convolution을 이용한 Upsampling
date: 2018-08-01 00:00:00
img: dl/concept/transposed-dilated-convolution/thumbnail.PNG
categories: [dl-concept] 
tags: [deep learning, convolution, transposed] # add tag
---

<br>

- 참조 : https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0

<br>

+ Transposed convolution은 `upsampling`을 목적으로 input image의 중간 중간을 0으로 채워서 output을 키우는 역할을 합니다.
+ <img src="../assets/img/dl/concept/transposed-dilated-convolution/transposed.gif" alt="Drawing" style="width: 600px;"/>
    + 파란색이 인풋, 초록색이 아웃풋 입니다.
+ 위 예제에서는 3x3 행렬이 5x5로 커졌습니다.   

<br>

+ Dilated는 segmentation에서 조금 더 디테일한 정보를 얻기 위해 receptive field를 늘리는 역할을 합니다.
    + 따라서 weights 중간을 0으로 채워주는 느낌입니다.

+ <img src="../assets/img/dl/concept/transposed-dilated-convolution/dilated.gif" alt="Drawing" style="width: 600px;"/>