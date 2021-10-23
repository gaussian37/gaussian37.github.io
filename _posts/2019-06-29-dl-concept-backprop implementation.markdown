---
layout: post
title: 딥러닝 Backpropagation 정리
date: 2019-06-29 00:00:00
img: dl/concept/deep-neural-network.jpg
categories: [dl-concept] 
tags: [python, deep learning, backpropagation, 역전파] # add tag
---

<br>

[deep learning 관련 글 목록](https://gaussian37.github.io/dl-concept-table/)

<br>

- 딥러닝의 backpropagation 관련 내용과 자주 사용되는 식들의 미분값에 대하여 정리해 보겠습니다.

<br>

## **목차**

<br>

- ### sigmoid
- ### binary cross entropy
- ### binary cross entropy with sigmoid
- ### softmax
- ### cross entropy with softmax
- ### multi layer perceptron

<br>

## **sigmoid**

<br>

- `sigmoid` 함수의 정의는 다음과 같습니다.

<br>

- $$ \sigma(x) = \frac{1}{1 + e^{-x}} \tag{1} $$

<br>

- `sigmoid` 함수의 미분 결과는 다음과 같습니다.

<br>

- $$ \frac{d\sigma(x)}{dx} = \frac{d}{dx} (1 + e^{-x})^{-1} \tag{2} $$

- $$ = (-1)\frac{1}{(1 + e^{-x})^{2}} \frac{d}{dx}(1 + e^{-x}) \tag{3} $$

- $$ = (-1)\frac{1}{(1 + e^{-x})^{2}} (0 + e^{-x}) \frac{d}{dx}(-x) \tag{4} $$

- $$ = (-1)\frac{1}{(1 + e^{-x})^{2}}e^{-x}(-1) \tag{5} $$

- $$ = \frac{e^{-x}}{(1 + e^{-x})^{2}} \tag{6} $$

- $$ = \frac{1 + e^{-x} - 1}{(1 + e^{-x})^{2}} \tag{7} $$

- $$ = \frac{1 + e^{-x}}{(1 + e^{-x})^{2}} - \frac{1}{(1 + e^{-x})^{2}} \tag{8} $$

- $$ = \frac{1}{(1 + e^{-x})} - \frac{1}{(1 + e^{-x})^{2}} \tag{9} $$

- $$ = \frac{1}{(1 + e^{-x})}(1 - \frac{1}{(1 + e^{-x})} \tag{10} $$

- $$ = \sigma(x)(1 - \sigma(x)) \tag{11} $$

<br>

- $$ \therefore  \frac{d\sigma(x)}{dx} = \sigma(x)(1 - \sigma(x)) \tag{12} $$

<br>

## **binary cross entropy**

<br>

- `binary cross entropy`의 식은 다음과 같습니다.

<br>

- $$ J = -y\ln{(p)} - (1-y)\ln{(1-p)} \tag {13} $$

<br>

- 위 식에서 $$ y $$ 는 정답에 해당 값이며 0 또는 1을 가집니다. $$ p $$ (predict)는 모델의 출력값을 나타냅니다. 

<br>

- 식(13)의 `binary cross entropy` 함수의 미분 결과는 다음과 같습니다.

<br>

- $$ \frac{dJ}{dp} = -\frac{y}{p} -\frac{1-y}{1-p}\frac{1-p}{dp} = -\frac{y}{p} + \frac{1-y}{1-p} \tag{14} $$

<br>

## **binary cross entropy with sigmoid**

<br>

- 일반적으로 `binary classification` 같은 문제에서는 `binary cross entropy`와 `sigmoid`를 섞어서 사용합니다.
- 식 (13)의 $$ p $$ 가 모델의 최종 출력 $$ z $$ 에 `sigmoid` 함수를 적용한 $$ \sigma(z) $$ 라고 가정해 보겠습니다. 그러면 다음과 같이 식을 적을 수 있습니다.

<br>

- $$ J = -y\ln{(p)} - (1-y)\ln{(1-p)} = -y\ln{(\sigma(z))} - (1-y)\ln{(1-\sigma(z))} \tag{15} $$

<br>

- 식 (15)를 $$ z $$에 대하여 미분하기 위해 `chain rule`을 이용하여 나타내면 다음과 같습니다.

<br>

- $$ \frac{dJ}{dz} = \frac{dJ}{dp}\frac{dp}{dz} \tag{16} $$

<br>

- 식 (16)에서 $$ \frac{dJ}{dp} $$ 의 결과는 식 (14)를 통해 구하였고 $$ \frac{dp}{dz} $$ 는 식 (12)를 통하여 구하였습니다. 따라서 다음과 같이 정리할 수 있습니다.

<br>

- $$ \frac{dJ}{dz} = \frac{dJ}{dp}\frac{dp}{dz} = (-\frac{y}{p} + \frac{1-y}{1-p})\sigma(z)(1 - \sigma(z)) \tag{17} $$

- $$ = (-\frac{y}{\sigma(z)} + \frac{1-y}{1-\sigma(z)})\sigma(z)(1 - \sigma(z)) \tag{18} $$

- $$ = \frac{-y + y\sigma(z) + \sigma(z) -y\sigma(z)}{\sigma(z)(1 - \sigma(z))}\sigma(z)(1 - \sigma(z)) \tag{19} $$

- $$ = \frac{-y + \sigma(z)}{\sigma(z)(1 - \sigma(z))}\sigma(z)(1 - \sigma(z)) \tag{20} $$

- $$ = \sigma(z) - y \tag{21} $$

<br>

- $$ \therefore \frac{dJ}{dz} = \frac{-y\ln{(\sigma(z))} - (1-y)\ln{(1-\sigma(z))}}{dz} = \sigma(z) - y \tag{22} $$

<br>

## **softmax**

<br>

- `sofrmax`의 식은 다음과 같습니다.


<br>

## **cross entropy with softmax**

<br>


<br>

## **multi layer perceptron**

<br>


<br>





<br>

[deep learning 관련 글 목록](https://gaussian37.github.io/dl-concept-table/)

<br>