---
layout: post
title: Depthwise separable convolution 연산
date: 2019-10-14 00:00:00
img: dl/concept/dwsconv/dwsconv.png
categories: [dl-concept] 
tags: [mobileNet, inception, xception, depthwise convolution, pointwise convolution, depthwise separable convolution] # add tag
---

<br>

- 출처: https://youtu.be/T7o3xvJLuHk

<br>

- 이번 글에서는 `convolution` 연산 방법 중 하나인 `depthwise separable convolution`에 내용에 다루어 보도록 하겠습니다.
- 제 블로그의 `mobilenet` 관련 글을 참조하셔도 좋습니다.

<br>
<center><img src="../assets/img/dl/concept/dwsconv/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 슬라이드는 일반적인 convolution 연산을 나타냅니다.
- 살펴보면 $$ D_{F} $$는 height와 width의 크기이고 $$ M $$은 채널의 크기입니다. 인풋이 컬러 이미지라면 $$ M $$은 R,G,B로 3입니다.
    - 따라서 인풋의 shape은 $$ (D_{F}, D_{F}, M) $$이 됩니다.
- 그리고 $$ D_{K} $$는 필터(커널)에 해당합니다. 필터의 채널은 인풋의 채널과 같기 때문에 똑같이 $$ M $$이 됩니다.
- 인풋이 필터를 거쳐 convolution 연산을 하게 되면 결과물로 매트릭스가 한 개 나오게 됩니다. (채널이 1인 텐서인 형태입니다.)

<br>
<center><img src="../assets/img/dl/concept/dwsconv/2.gif" alt="Drawing" style="width: 300px;"/></center>
<br>

- 그 다음 필터의 갯수가 $$ N $$에 해당하는데 필터의 종류(N개) 만큼 convolution 연산을 하기 때문에 총 N개의 매트릭스가 결과물로 나오게 됩니다.
- 이것들을 모두 쌓으면 출력단에 $$ (D_{G}, D_{G}, N) $$ shape의 volume이 출력물로 나오게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/dwsconv/3.png" alt="Drawing" style="width: 300px;"/></center>
<br>  

- 그러면 이 연산의 연산량이 얼마가 되는지 간략하게 알아보겠습니다. 연산량을 간단하게 알아보기 위해 `곱`연산을 기준으로 알아보겠습니다.
- 먼저 한 개의 필터가 인풋의 한 위치에서 처리할 때 사용되는 곱연산은 $$ D_{K}^{2} \times M $$이 됩니다.
- 이 때, 이동할 수 있는 위치의 경우의 수가 $$ D_{G} \times D_{G} $$의 경우가 있기 때문에($$ D_{G} $$는 아웃풋의 height, wdith 크기) 필터 하나가 인풋 하나를 처리하는 데 필요한 연산 수는 $$ D_{G}^{2} \times D_{K}^{2} \times M $$이 됩니다.
- 마지막으로 $$ N $$개의 필터가 있다면 아웃풋은 $$ (D_{G], D_{G}, N ) $$ 크기를 가지고 연산량도 $$ N \times D_{G}^{2} \times D_{K}^{2} \times M $$가 됩니다. 
- `depthwise separable convolution`은 $$ N \times D_{G}^{2} \times D_{K}^{2} \times M $$ 연산량을 줄이는 것이 목표입니다.

<br>

- `depthwise separable convolution`은 두가지 과정을 거칩니다.
    - `depthwise convolution`: filtering stage
    - `pointwise convolution`: combination stage

<br>
<center><img src="../assets/img/dl/concept/dwsconv/4.png" alt="Drawing" style="width: 300px;"/></center>
<br>  




