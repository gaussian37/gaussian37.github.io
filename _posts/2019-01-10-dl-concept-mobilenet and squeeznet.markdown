---
layout: post
title: SqueezeNet(스퀴즈넷)과 MobileNet(모바일넷)
date: 2019-01-10 03:49:00
img: dl/concept/mobileSqueezeNet/squeezenet.png
categories: [dl-concept] 
tags: [deep learning, cnn, mobilenet, squeezenet, 모바일넷, 스퀴즈넷] # add tag
---

<br>

- squeezenet 관련 필자의 다른 블로그 내용 : https://gaussian37.github.io/dl-concept-squeezenet/
- mobilenet v1 관련 필자의 다른 블로그 내용 : https://gaussian37.github.io/dl-concept-mobilenet/

## SqueezNet과 MobileNet 사용 목적

+ 스마트폰이나 저가형 하드웨어에서 이미지 처리를 하고 싶을 때 기존의 무거운 CNN 모델을 업로드 하기 어렵습니다.
+ 제한된 resource를 사용하여 빠른 prediction을 하고 싶을 때, `모델의 사이즈`가 중요한 요소 입니다.
+ `MobileNet`과 `SqueezeNet`은 다소 가벼운 모델이지만 AlexNet을 넘나드는 정확도를 보여줍니다.
	+ `MobileNet`과 `SqueezeNet`은 정확도를 조금 손해보더라도 모델은 최대한 작고 효율적으로 설계하였습니다.

<br><br>

## SqueezeNet과 MobileNet을 통하여 배울점

+ SqueezeNet의 컨셉과 `fire modules`
+ Fully connected layer의 문제점과 `max pooling layer`로 해결하는 방법
+ MobileNet의 컨셉과 `spatial separable convolution`

<br><br>

## SqueezeNet에 대하여 알아봅시다.

+ `SqueezeNet`은 `AlexNet`과 비슷한 정확도를 가지지만 weight는 $$ \frac{1}{50} $$배 수준으로 상당히 가볍습니다.
+ `SqueezeNet`은 다음과 같은 컨셉을 가지고 있습니다.
+ 기존의 사용하였던 `3 x 3` 필터를 일부를 `1 x 1` 필터로 대체하였습니다. 
	+ `1 x 1` 필터를 사용함으로써 weight의 수를 $$ \frac{1}{9} $$로 줄일 수 있습니다.
+ input 채널의 수를 `3 x 3` 필터를 이용하여 줄였습니다.
	+ convolution layer 에서의 파라미터의 수는 filter의 사이즈, 채널의 수 그리고 필터의 수에 따라 달라집니다.
+ downsample을 늦게 적용하여 convolution layer가 `큰 activation map`을 가질 수 있도록 합니다.     
	+ 데이터 downsample을 늦게 할수록 (예를 들어 stride를 1보다 크게 하여서 downsample 하는 작업을 늦게 적용함) 더 많은 정보들이 layer에 포함될 수 있습니다.
	+ 이 방법을 통하여 `모델은 가능한 작게` 만들지만, `정확도는 최대한 크게` 만들 수 있습니다.
		
<br><br>

## Fire module 이란?

<img src="../assets/img/dl/concept/mobileSqueezeNet/squeezenet.png" alt="Drawing" style="width: 500px;"/>

<br>

+ `fire module`은 2개의 layer로 이루어져 있습니다. Squeeze layer와 Expansion layer 입니다.
+ Squeeze layer가 먼저 나온 뒤 Expansion layer가 따라서 나옵니다.
+ Squeeze layer
	+ Squeeze layer는 1x1 convolution으로 이루어져 있습니다.
	+ 1x1 convolution은 input 데이터의 모든 channel을 하나로 합친 뒤 input channel의 수를 줄여서 다음 layer에 전달하는 역할을 합니다.
		+ 간단하게 말하면 channel 수를 조절하는 역할을 합니다.		
+ expansion layer
	+ 1 x 1 convolution이 3 x 3 convolution과 섞인 형태 입니다. 
	+ 1 x 1 convolution은 spatial structure를 감지하기 어렵지만 이전 layer의 channel들을 다양한 방법으로 결합시킵니다.
	+ 3 x 3 convolution은 이미지 내의 structure를 잘 잡아냅니다.
	+ 1 x 1 과 3 x 3 필터 사이즈의 convolution을 섞어서 사용함으로써 표현력이 증가하고 파라미터의 숫자도 동시에 줄일 수 있습니다.
	+ 이 때 주의할 점은 1 x 1 convolution과 3 x 3 convolution 결과의 크기를 같게 하려면 적당한 `padding`을 잘 해주어야 합니다. 사이즈가 같아져야만 layer들을 쌓을 수 있습니다.
	
<br><br>

## SqueezeNet Architecture

+ SqueezeNet은 8개의 `fire module`을 사용하고 input/output 각각에 1개의 convolution layer를 사용합니다. 주목할만한 점은 SqueezeNet에서 Fully Connected Layer는 전혀 사용되지 않았다는 것입니다.
+ Fully connected layer는 상당히 많은 양의 `parameter`를 가지고 있습니다. convolution layer에 비교하면 상당히 많습니다.
	+ 너무 많은 양의 parameter에 맞추다 보면 `overfitting`이 발생할 확률도 커집니다. 
+ 따라서 SqueezeNet에서는 `Global Average Pooling`을 사용합니다.
	+ Global average pooling은 이전의 convolution layer로 부터 각각의 channel을 전달 받은 다음 모든 값에 대하여 average를 취해줍니다.
	+ Pooling layer은 weight 값이 없기 때문에 model 사이즈를 더 크게 만들진 않습니다.
	+ 또한 parameter를 더 많들어 내지 않기 때문에 Fully connected layer에 비해서 overfitting 문제에 좀 더 자유롭습니다.



<br><br>






+ 출처 : https://aiinpractice.com/squeezenet-mobilenet/
