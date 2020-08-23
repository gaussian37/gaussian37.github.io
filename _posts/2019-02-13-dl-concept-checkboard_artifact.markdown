---
layout: post
title: Transposed Convolution과 Checkboard artifact
date: 2019-02-13 00:00:00
img: dl/concept/checkboard_artifact/0.png
categories: [dl-concept] 
tags: [deep learning, convolution, transposed, checkboard artifact] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

<br>

- 참조 : https://distill.pub/2016/deconv-checkerboard/

<br>

## **목차**

<br>

- ### Checkboard pattern 이란
- ### Transposed Convolution과 Overlap
- ### Overlap과 Learning
- ### 더 좋은 Upsampling 방법
- ### Image Generation 결과
- ### Gradient의 Artifact
- ### 결론

<br>

## **Checkboard pattern 이란**

<br>

- 딥러닝에서 feature를 Upsampling 할 때, 사용하는 방법 중 하나인 `Transposed Convolution`을 사용할 때 발생하는 문제인 `Checkboard artifact`에 대하여 다루어 보겠습니다.


<br>
<center><img src="../assets/img/dl/concept/checkboard_artifact/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 뉴럴 네트워크에 의해 생성된 이미지들을 자세히 들여다 보면 위 그림처럼 인공적인 체크보드 패턴을 가지는 결과를 종종 가집니다.
- 신기하게도, 체크보드 패턴은 강한 색상을 가진 이미지에서 가장 두드러지는 경향이 있습니다.

## **Transposed Convolution과 Overlap**

<br>

- 뉴럴 네트워크가 이미지를 생성하였을 때, 이 이미지는 종종 해상도는 낮으면서 높은 수준의 description을 가지는 경우가 있습니다.
- 이 경우 네트워크는 낮은 해상도에서 대략적인 이미지를 설명(describe)하고 그리고 높은 해상도로 이미지를 키워 나가면서 상세 정보들을 채워나아갑니다.
- 이를 위해서는 저해상도 이미지에서 고해상도 이미지로 변환하는 방법이 필요합니다. 이러한 방법 중 Transposed Convolution 이 있습니다. Transposed Convolution을 사용하면 작은 이미지의 모든 점을 사용하여 큰 이미지를 만들어 낼 수 있습니다.
- Convolution 연산은 커널이 슬라이딩 윈도우 방식으로 이동하면서 진행됩니다. 특히, **kernel의 크기와 stride의 크기에 따라서 convolution 연산의 overlap 영역이 발생할 수 있습니다.** 이 때 Transposed Convolution을 어떻게 사용하느냐에 따라서 overlap이 없을 수도 있고 또는 많이 생길 수도 있습니다.

<br>
<center><img src="../assets/img/dl/concept/checkboard_artifact/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 보면 `stride = 2`로 고정하였을 때, `kernel_size`를 변경하였을 때, overlap 영역의 변화를 살펴볼 수 있습니다.
- 


<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>