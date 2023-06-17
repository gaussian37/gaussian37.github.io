---
layout: post
title: Deformable Convolution 정리
date: 2022-01-25 00:00:00
img: dl/concept/deformable_convolution/0.png
categories: [dl-concept]
tags: [deformable convolution, convolution] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 논문1 : https://arxiv.org/abs/1703.06211
- 논문2 : https://arxiv.org/abs/1811.11168
- 논문 발표 영상 : https://youtu.be/HRLMSrxw2To
- 참조 : https://medium.com/@phelixlau/notes-on-deformable-convolutional-networks-baaabbc11cf3
- 참조 : https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2
- 참조 : https://youtu.be/RRwaz0fBQ0Y


- 이번 글에서는 `deformable convolution`에 대하여 다루어 보도록 하겠습니다. `deformable convolution`은 `receptive field`가 직사각형 형태로 졍형화 되어 있는 기본적인 `convolution` 연산에 비해 좀 더 자유도를 가지는 형태의 `receptive field`를 가지는 `convolution` 연산입니다.
- 현재 기준으로는 `deformable convolution`은 `torch.nn`에 구현되어 있지 않고 

<br>

<br>

 
<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
