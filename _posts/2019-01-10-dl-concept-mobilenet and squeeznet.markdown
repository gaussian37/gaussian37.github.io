---
layout: post
title: MobileNet(모바일넷)과 SqueezeNet(스퀴즈넷)
date: 2019-01-10 03:49:00
img: dl/concept/mobileSqueezeNet/squeezenet.png
categories: [dl-concept] 
tags: [deep learning, cnn, mobilenet, squeezenet, 모바일넷, 스퀴즈넷] # add tag
---

## SqueezNet과 MobileNet 사용 목적

+ 스마트폰이나 저가형 하드웨어에서 이미지 처리를 하고 싶을 때 기존의 무거운 CNN 모델을 업로드 하기 어렵습니다.
+ 제한된 resource를 사용하여 빠른 prediction을 하고 싶을 때, `모델의 사이즈`가 중요한 요소 입니다.
+ `MobileNet`과 `SqueezeNet`은 다소 가벼운 모델이지만 AlexNet을 넘나드는 정확도를 보여줍니다.
	+ `MobileNet`과 `SqueezeNet`은 정확도를 조금 손해보더라도 모델은 최대한 작고 효율적으로 설계하였습니다.

<br>

## SqueezeNet과 MobileNet을 통하여 배울점

<br>

+ SqueezeNet의 컨셉과 `fire modules`
+ Fully connected layer의 문제점과 `max pooling layer`로 해결하는 방법
+ MobileNet의 컨셉과 `spatial separable convolution`

<br>

## SqueezeNet에 대하여 알아봅시다.

<br>

+ `SqueezeNet`은 `AlexNet`과 비슷한 정확도를 가지지만 weight는 $$ \frac{1}{50} $$배 수준으로 상당히 가볍습니다.
+ `SqueezeNet`은 다음과 같은 컨셉을 가지고 있습니다.
	+ 기존의 사용하였던 `3 x 3` 필터를 일부를 `1 x 1` 필터로 대체하였습니다. 
		+ `1 x 1` 필터를 사용함으로써 weight의 수를 $$ \frac{1}{9} $$로 줄일 수 있습니다.
	+ input 채널의 수를 `3 x 3` 필터를 이용하여 줄였습니다.
		+ convolution layer 에서의 파라미터의 수는 filter의 사이즈, 채널의 수 그리고 필터의 수에 따라 달라집니다.
	+     

+ 출처 : https://aiinpractice.com/squeezenet-mobilenet/
