---
layout: post
title: CNN 파라미터 숫자 계산
date: 2018-12-23 00:00:00
img: dl/keras/keras.png
categories: [dl-keras] 
tags: [deep learning, keras, 케라스, CNN, 파라미터, parameter] # add tag
---

Keras에서 `model.summary()`를 하면 parameter의 갯수를 쉽게 구할 수 있습니다.
**Convolutional Neural Network**에서는 어떻게 parameter를 계산하는 것일까요?

먼저 간단하게 CNN 구조를 아래와 같이 만들어 보고 `model.summary()`를 통하여 파라미터 수를 확인해 보겠습니다.

<img src="../assets/img/dl/keras/weight/weight1.png" alt="Drawing" style="width: 500px;"/>

<br>

+ input : (150, 150, 3) 형태로 height, width가 각각 150 이고 RGB 값이 3개라고 하겠습니다.

<img src="../assets/img/dl/keras/weight/weight2.png" alt="Drawing" style="width: 500px;"/>

<br>

+ 첫 번째 conv2d 에서는 (150, 150, 3) 이미지에 (3, 3) 필터를 32개 사용하였습니다.
+ 이 때, 어떻게 파라미터의 갯수가 896개가 나올 수 있을까요?
    + 정확한 내용은 본 블로그의 dl-concept 에서 cnn 블로그 내용을 확인해 보시면 도움이 되겠습니다.
    + 먼저 (3, 3) 필터 한개에는 3 x 3 = `9`개의 파라미터가 있습니다.
    + 그리고 입력되는 3-channel 각각에 서로 다른 파라미터들이 입력 되므로 R, G, B 에 해당하는 `3`이 곱해집니다.
    + 그리고 `Conv2D(32, ...)` 에서의 32는 32개의 필터를 적용하여 다음 층에서는 채널이 총 `32개`가 되도록 만든다는 뜻입니다.
    + 여기에 bias로 더해질 상수가 각각의 채널 마다 존재하므로 `32개`가 추가로 더해지게 됩니다.
    + 정리하면, 3 x 3(필터 크기) x 3 (#입력 채널(RGB)) x 32(#출력 채널) + 32(출력 채널 bias) = 896이 됩니다.
    
<img src="../assets/img/dl/keras/weight/weight2.png" alt="Drawing" style="width: 500px;"/>

<br>

+ 두 번째 conv2d 에서는 입력이 (72, 72, 32) 이고 (3, 3) 필터가 64개 입니다.
    + 앞에서 내용과 똑같이 적용해 보겠습니다.
    + 3 x 3 (필터 크기) x 32 (#입력 채널) x 64(#출력 채널) + 64 = 18496 입니다.

이 내용을 정확히 이해하려면 반드시 **convolution, padding, pooling**등의 내용을 참조하시기 바랍니다.

도움이 되셨다면 광고 클릭 한번이 저에게 큰 도움이 됩니다. 꾸벅.

<a href="https://coupa.ng/bgl1OZ" target="_blank"><img src="https://static.coupangcdn.com/image/affiliate/category/20180610/electronic-640-x-100.jpg" alt=""></a>

<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.4%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>  




