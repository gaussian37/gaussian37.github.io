---
layout: post
title: Convolution 연산 정리
date: 2019-11-05 00:00:00
img: dl/concept/conv/conv.gif
categories: [dl-concept] 
tags: [convolution operation, 컨볼루션 연산] # add tag
---

<br>

- 출처: https://github.com/vdumoulin/conv_arithmetic

<br>

- 아래 애니메이션의 `파란색`이 `인풋`이고 `청록색`이 `아웃풋`입니다.

<br>

## **Basic Convolution Operation**

<br>

- Convolution 연산을 이용하면 input의 feature를 압축하게 되므로 convolution 연산 이후의 feature map의 크기는 더 줄어들게 됩니다.
- 아래 애니메이션들도 보면 파란색의 인풋이 convolution 연산을 거치면서 청록색 아웃풋 처럼 사이즈가 작아지게 된 것을 볼 수 있습니다.

<br>

<br>
<center><img src="../assets/img/dl/concept/conv/no_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/arbitrary_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/same_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/full_padding_no_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/no_padding_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/padding_strides.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/conv/padding_strides_odd.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

