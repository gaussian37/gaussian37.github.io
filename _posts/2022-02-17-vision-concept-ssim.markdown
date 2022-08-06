---
layout: post
title: SSIM (Structural Similarity Index)
date: 2022-02-17 00:00:00
img: vision/concept/ssim/0.png
categories: [vision-concept] 
tags: [SSIM, Structural Similarity Index] # add tag
---

<br>

- 참조 : https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e
- 참조 : https://bskyvision.com/878
- 참조 : https://walkaroundthedevelop.tistory.com/m/56
- 참조 : https://nate9389.tistory.com/2067
- 참조 : https://medium.com/@sanari85/image-reconstruction-%EC%97%90%EC%84%9C-ssim-index%EC%9D%98-%EC%9E%AC%EC%A1%B0%EB%AA%85-b3ca26434fb1

<br>

- 이번 글에서는 두 이미지를 비교하는 지표인 `SSIM`에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [SSIM의 정의 및 Pytorch Scratch 구현](#ssim의-정의-및-pytorch-scratch-구현-1)
- ### [SSIM의 Pytorch 외부 라이브러리 사용](#ssim의-pytorch-외부-라이브러리-사용-1)
- ### [SSIM의 skimage 사용](#ssim의-skimage-사용-1)

<br>

## **SSIM의 정의 및 Pytorch Scratch 구현**

<br>

- `SSIM`은 `Structural Similarity Index`의 약어로 사용되며 주어진 2개의 이미지의 `similarity(유사도)`를 계산하는 측도로 사용됩니다.
- `SSIM`은 두 이미지의 단순 유사도를 측정하는데 사용하기도 하지만 풀고자 하는 문제가 두 이미지가 유사해지도록 만들어야 되는 문제일 때 `SSIM`을 Loss Function 형태로 사용하기도 합니다. 왜냐하면 `SSIM`이 gradient-based로 구현되어 있기 때문입니다.
- 딥러닝에서 두 이미지를 유사하게 만드는 문제나 depth estimation을 할 때, 두 이미지 또는 두 패치의 유사도를 측정하여 Loss Function을 사용하는 방법이 많이 사용됩니다. 
- 따라서 이번 챕터에서는 `SSIM`의 원리에 대하여 먼저 알아보고 Pytorch로 어떻게 구현하는 지 살펴보도록 하겠습니다.

<br>

- Objective 

<br>


## **SSIM의 Pytorch 외부 라이브러리 사용**

<br>

<br>


## **SSIM의 skimage 사용**

<br>

<br>

