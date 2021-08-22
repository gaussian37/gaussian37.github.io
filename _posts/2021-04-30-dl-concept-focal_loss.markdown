---
layout: post
title: Focal Loss (Focal Loss for Dense Object Detection) 알아보기
date: 2021-04-30 00:00:00
img: dl/concept/focal_loss/0.png
categories: [dl-concept]
tags: [deep learning, focal loss] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 논문 : https://arxiv.org/pdf/1708.02002.pdf
- 참조 : https://youtu.be/d5cHhLyWoeg
- 참조 : https://www.analyticsvidhya.com/blog/2020/08/a-beginners-guide-to-focal-loss-in-object-detection/
- 참조 : https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/44tlnmmt3h0" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

## **목차**

<br>

- ### Focal Loss의 필요성
- ### Cross Entropy Loss의 문제점
- ### Focal Loss 알아보기

<br>

- 이번 글에서는 **Focal Loss for Dense Object Detection** 라는 논문의 내용을 알아보겠습니다. 이 논문에서는 핵심 내용은 `Focal Loss`와 이 Loss를 사용한 `RetinaNet`이라는 Object Detection 네트워크를 소개합니다. 다만, RetinaNet에 대한 내용은 생략하고 Loss 내용에만 집중하도록 하겠습니다.

<br>

## **Focal Loss의 필요성**

<br>

- Object Detection에는 크게 2가지 종류의 알고리즘이 있습니다. R-CNN 계열의 `two-stage detector`와 YOLO, SSD 계열의 `one-stage detector` 입니다.
- one-stage, two-stage detector의 상세 내용은 다음 링크를 참조해 주시기 바랍니다. ([https://gaussian37.github.io/vision-detection-table/](https://gaussian37.github.io/vision-detection-table/))
- 간략하게 설명하면 Object Detection은 여러 object들을 Bounding Box를 통해 `Localization` 즉, 위치를 찾고 `Classification` 즉, 어떤 물체인지 분류를 하는 작업을 합니다. 
- `two-stage detector`는 먼저 localization을 한 다음에 classification이 순차적으로 이루어지고 `one-stage detector`는 localization과 classification을 동시에 처리합니다. 정확도 성능으로는 two-stage detector가 좋지만 연산 속도가 오래 걸리는 단점이 있습니다.

<br>

- `Focal Loss`는 one-stage detector의 정확도 성능을 개선하기 위하여 고안되었습니다. one-stage detector가 two-stage detector에 비하여 가지고 있는 문제점은 `학습 중 클래스 불균형 문제`가 심하다는 것입니다. 
- 예를 들어 학습 중 배경에 대하여 박스를 친 것과 실제 객체에 대하여 박스를 친 것의 비율을 살펴보면 압도적으로 배경에 대하여 박스를 친 것이 많다는 것입니다. 학습 중에서 배경에 대한 박스를 출력하면 오류라고 학습이 되지만 그 빈도수가 너무 많다는 것이 학습에 방해가 된다는 뜻입니다. (SSD에서는 학습 시 한 이미지 당 만개 이상의 background에 대한 박스가 있다고 합니다.)
- 이와 같은 문제점이 발생하는 이유는 `dense sampling of anchor boxes (possible object locations)`로 알려져 있습니다. 예를 들어 RetinaNet에서는 각각의 pyramid layer에서 anchor box가 수천개가 추출됩니다.

<br>

- 정리하면 이와 같은 클래스 불균형 문제는 다음 2가지 문제의 원인이 됩니다.
- ① 대부분의 Location은 학습에 기여하지 않는 `easy negative`이므로 (detector에 의해 background로 쉽게 분류될 수 있음을 의미함) **학습에 비효율적**입니다.
- ② `easy negative` 각각은 높은 확률로 객체가 아님을 잘 구분할 수 있습니다. 즉, 각각의 `loss` 값은 작습니다. 하지만 비율이 굉장히 크므로 전체 `loss` 및 `gradient`를 계산할 때, **easy negative의 영향이 압도적으로 커지는 문제가 발생**합니다.

<br>

- 이러한 문제를 개선하기 위하여 `Focal Loss` 개념이 도입됩니다.
- Focal Loss는 간단히 말하면 `Cross Entropy`의 클래스 불균형 문제를 다루기 위한 개선된 버전이라고 말할 수 있으며 **어렵거나 쉽게 오분류되는 케이스에 대하여 더 큰 가중치를 주는 방법**을 사용합니다. (객체 일부분만 있거나, 실제 분류해야 되는 객체들이 이에 해당합니다.) 반대로 쉬운 케이스의 경우 낮은 가중치를 반영합니다. (background object가 이에 해당합니다.)

<br>

## **Cross Entropy Loss의 문제점**

<br>

- `Cross Entropy Loss`의 경우 잘 분류한 경우 보다 **잘못 예측한 경우에 대하여 페널티를 부여하는 것에 초점**을 둡니다.
- 예를 들어 이진 분류에 대한 Cross Entropy Loss는 다음과 같은 식을 따릅니다.

<br>

- $$ \text{Cross Entropy Loss} = -Y_{\text{act}} \log{(Y_{\pred})} - (1 - Y_{\text{act}})\log{1 - Y_{\text{pred}}} \tag{1} $$ 

- $$ \text{where} Y_{\text{act}} = \text{Actual Value of Y} $$

- $$ \text{where} Y_{\text{pred}} = \text{Predicted Value of Y} $$

<br>

- 표기를 간단하게 하기 위하여 $$  Y_{\text{act}} $$ 는 $$ Y $$ 로 표기하고 $$  Y_{\text{pred}} $$ 는 $$ p $$ 로 표기하겠습니다.
- 잘못 예측한 경우에 대하여 페널티를 부여하는 예시를 살펴보겠습니다. $$ Y = 1 $$ 인 케이스에 대하여 Cross Entropy Loss를 살펴보면 다음과 같습니다.

<br>

- $$ \text{CE}(p, y) = -\log{(p)} + 0 = -\log{(p)} \tag{2} $$

<br>

- 만약 $$ p $$ 의 값이 1이면 $$ \text{CE}(p, y) = 0 $$ 이 됩니다. 즉, 잘 예측하였지만 보상은 없으며 단지, 페널티가 없어집니다.
- 반면 $$ p $$ 의 값을 0에 가깝게 예측하게 되면 $$ \text{CE}(p, y) \approx \infty $$ 가 됩니다. 즉, 페널티가 굉장히 커지게 됩니다.

<br>

- 



<br>

## **Focal Loss 알아보기**

<br>


<br>





<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
