---
layout: post
title: Stride와 Pooling의 비교
date: 2017-05-03 00:00:00
img: dl/concept/stride_vs_pooling/0.png
categories: [dl-concept] 
tags: [deep learning, stride, pooling, max pooling] # add tag
---

<br>

[deep learning 관련 글 목록](https://gaussian37.github.io/dl-concept-table/)

<br>

- 참조 : https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling

<br>

- Convolutional Neural Network에서 **feature의 resolution을 줄일 때**, `stride=2` 또는 `max/average pooling`을 이용하여 **resolution을 1/2로 줄이는 방법**을 많이 사용합니다.
- `convolution layer`를 이용하여 `stride = 2`로 줄이면 학습 가능한 파라미터가 추가되므로 학습 가능한 방식으로 resolution을 줄이게 되나 그만큼 파라미터의 증가 및 연산량이 증가하게 됩니다.
- 반면 `pooling`을 이용하여 resolution을 줄이게 되면 학습과 무관해지며 학습할 파라미터 없이 정해진 방식 (max, average)으로 resolution을 줄이게 되어 연산 및 학습량은 줄어들지만 `convolution with stride` 방식보다 성능이 좋지 못하다고 알려져 있습니다.

<br>

- 이러한 `stride`와 `pooling`의 성질에 따라 선택하여 사용할 수 있으며 다양한 커뮤니티에서 `stride`와 `pooling` 사용 시 장단점에 대한 의견을 소개 하고 있습니다.
- 이번 글에서는 그 의견들을 정리하고자 하며, `stride`, `pooling`의 장점과 단점 순서로 나열해 보겠습니다.

<br>

## **convolution with stride 방식의 장단점**

<br>

- ① 학습 가능한 파라미터가 추가되므로 네트워크가 resolution을 잘 줄이는 방법을 학습할 수 있어서 pooling보다 성능이 좋습니다.
- ② feature를 뽑기 위한 Convolution Layer와 Downsampling을 위한 stride를 동시에 적용할 수 있습니다. 이 경우 같은 3 x 3 크기의 필터를 사용하더라도 stride가 적용되기 때문에 **더 넓은 receptive field**를 볼 수 있습니다.
- ③ [STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET](https://arxiv.org/pdf/1412.6806.pdf)에서는 모든 Pooling을 Convolution with stride로 변경 시 성능 상승의 효과가 있는 것을 확인하였습니다.
    - `We find that max-pooling can simply be replaced by a convolutional layer with increased stride without loss in accuracy on several image recognition benchmarks`

<br>

## **pooling 방식의 장단점**

<br>

- ① convolution 연산 대비 연산량이 적으며 저장해야 할 파라미터의 숫자도 줄어드므로 학습 시간도 상대적으로 줄일 수 있고 인퍼런스 시 시간도 줄일 수 있습니다.
- ② [FishNet](https://papers.nips.cc/paper/2018/file/75fc093c0ee742f6dddaa13fff98f104-Paper.pdf) 에서 제안한 내용 중에 Skip Connection에서 Convolution layer가 계속 추가되면 backpropagation 시 gradient가 잘 전달이 안될 수 있다고 하여 단순히 Pooling만을 사용한 기법이 적용됩니다. 즉, **layer를 줄여서 gradient 전파에 초점을 두려고 할 때** pooling을 사용하는게 도움이 될 수 있습니다.

<br>

[deep learning 관련 글 목록](https://gaussian37.github.io/dl-concept-table/)

<br>

