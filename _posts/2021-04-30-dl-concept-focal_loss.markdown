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
- ### Balanced Cross Entropy Loss의 한계
- ### Focal Loss 알아보기
- ### Object Detection의 Focal Loss
- ### Semantic Segmentation의 Focal Loss

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

- $$ \text{Cross Entropy Loss} = -Y_{\text{act}} \log{(Y_{pred})} - (1 - Y_{\text{act}})\log{(1 - Y_{\text{pred}})} \tag{1} $$ 

- $$ \text{where,  } Y_{\text{act}} = \text{Actual Value of Y} $$

- $$ \text{where,  } Y_{\text{pred}} = \text{Predicted Value of Y} $$

<br>

- 표기를 간단하게 하기 위하여 $$  Y_{\text{act}} $$ 는 $$ Y $$ 로 표기하고 $$  Y_{\text{pred}} $$ 는 $$ p $$ 로 표기하겠습니다.
- 잘못 예측한 경우에 대하여 페널티를 부여하는 예시를 살펴보겠습니다. $$ Y = 1 $$ 인 케이스에 대하여 Cross Entropy Loss를 살펴보면 다음과 같습니다.

<br>

- $$ \text{CE}(p, y) = -\log{(p)} + 0 = -\log{(p)} \tag{2} $$

<br>

- 만약 $$ p $$ 의 값이 1이면 $$ \text{CE}(p, y) = 0 $$ 이 됩니다. 즉, 잘 예측하였지만 보상은 없으며 단지, 페널티가 없어집니다.
- 반면 $$ p $$ 의 값을 0에 가깝게 예측하게 되면 $$ \text{CE}(p, y) \approx \infty $$ 가 됩니다. 즉, 페널티가 굉장히 커지게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/focal_loss/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- $$ \text{Cross Entropy}(p_{t}) = -\log{(p_{t})} \tag{3} $$

- $$ \text{Focal Loss} = -(1 - p_{t})^{\gamma}\log{(p_{t})} \tag{4} $$

<br>

- 위 그래프와 식은 `Focal Loss`를 나타냅니다. `Focal Loss`와 `Cross Entropy`의 식을 비교해 보면 기본적인 Cross Entropy Loss에 $$ (1 - p_{t})^{\gamma} $$ term이 더 추가된 것을 확인할 수 있습니다. 기본적인 Cross Entropy는 $$ \gamma $$ 가 0일 때 입니다.
- 여기서 추가된 $$ (1 - p_{t})^{\gamma} $$ 의 역할은 `easy example`에 사용되는 **loss의 가중치를 줄이기 위함**입니다.
- 예를 들어 다음과 같은 2가지 경우를 살펴보겠습니다. 첫번째는 `Foreground` 케이스이며 이 때, $$ Y = 1 $$ 이라고 하며 $$ p = 0.95 $$ 라고 가정하겠습니다. 두번째는 `Background` 케이스이며 이 때, $$ Y = 0 $$ 이라고 하며 $$ p = 0.05 $$ 라고 가정하겠습니다.

<br>

- $$ \text{CE(Foreground)} = -\log{(0.95)} = 0.05 \tag{5} $$

- $$ \text{CE(Background)} = -\log{(1 - 0.05)} = 0.05 \tag{6} $$

<br>

- 식 (5)의 Foreground 케이스를 살펴보면 Foreground인 객체에 대하여 높은 확률인 0.95로 잘 분류하였고 그 결과 Loss가 0.05로 작은 것을 알 수 있습니다.
- 이와 유사하게 식 (6)의 Background 케이스를 살펴보면 Background임에 따라 낮은 확률인 0.05로 잘 분류하였고 그 결과 Loss가 0.05로 작은 것을 알 수 있습니다.
- 문제가 없어보이지만 여기서 발생하는 문제점은 **Foregound 케이스와 Background 케이스 모두 같은 Loss 값을 가진다는 것**에 있습니다. 왜냐하면 Background 케이스의 수가 훨씬 더 많기 때문에 같은 비율로 Loss 값이 업데이트되면 Background에 대하여 학습이 훨씬 많이 될 것이고 이 작업이 계속 누적되면 Foreground에 대한 학습량이 현저히 줄어들기 때문입니다.

<br>

## **Balanced Cross Entropy Loss의 한계**

<br>

- Cross Entropy 케이스에서 발생하는 문제인 Foreground와 Background 케이스의 비율이 다른 점을 개선하기 위하여 Cross Entropy Loss 자체에 비율을 보상하기 위한 weight를 추가로 곱해주는 방법을 사용할 수 있습니다.
- 예를 들어 Foreground 객체의 클래스 수와 Background 객체의 클래스 수 각각의 역수의 갯수를 각 Loss에 곱한다면 클래스 수가 많은 Background의 경우 Loss가 작게 반영될 것이고 클래스 수가 적은 Foreground의 경우 Loss가 크게 반영될 것입니다.
- 이와 같이 **각 클래스의 Loss 비율을 조절하는 weight** $$ w_{t} $$ 를 곱해주어 imbalance class 문제에 대한 개선을 하고자 하는 방법이 `Balanced Cross Entropy Loss` 라고 합니다.
- 일반적으로 $$ 0 \le w_{t} \le 1 $$ 범위의 값을 사용하며 식으로 표현하면 다음과 같습니다.

<br>

- $$ \text{CE}(p_{t}) = -w_{t}\log{(p_{t})} \tag{7} $$

<br>

- `Cross Entropy Loss`의 근본적인 문제가 Foreground 대비 Background의 객체가 굉장히 많이 나오는 class imbalance 문제에 해당하였습니다. 따라서 `Balanced Cross Entropy Loss`의 `weight` $$ w $$ 를 이용하면 $$ w $$ 에 대한 값의 조절을 통해 해결할 수 있을 것으로 보입니다. 즉, Forground의 weight는 크게 Background의 weight는 작게 적용하는 방향으로 개선하고자 하는 것입니다.
- 하지만 이와 같은 방법에는 문제점이 있습니다. 바로, **Easy/Hard example 구분을 할 수 없다는 점**입니다. **단순히 갯수가 많다고 Easy라고 판단하거나 Hard라고 판단하는 것에는 오차가 발생할 수 있습니다.**
- 다음과 같은 예제를 살펴보겠습니다. 0.95의 확률로 Foreground 객체라고 분류한 Foreground 케이스에 weight 0.75를 주는 경우와 0.05의 확률로 Foreground 객체라고 분류 (즉, 0.95의 확률로 Background 객체라고 분류)한 Background 케이스에 weight 0.25를 주는 경우를 살펴보겠습니다.

<br>

- $$ \text{CE(FG)} = -0.75 \log{(0.95)} = 0.038 \tag{8} $$

- $$ \text{CE(BG)} = -(1 - 0.75) \log{(1 - 0.05)} = 0.0128 \tag{9} $$

<br>

- 앞에서 설명한 바와 같이 통상적으로 Background 객체의 수가 작으므로 더 낮은 Loss를 반영하기 위해 더 작은 weight를 반영하도록 하였습니다.
- 그리고 식을 살펴보면 `Easy/Hard Example`에 대한 반영은 전혀 없는 것 또한 알 수 있습니다.
- 따라서, `Easy/Hard Example`을 반영하기 위하여 이 글의 주제인 `Focal Loss`에 대하여 다루어 보도록 하겠습니다.

<br>

## **Focal Loss 알아보기**

<br>

- `Focal Loss`는 **Easy Example의 weight를 줄이고 Hard Negative Example에 대한 학습에 초점을 맞추는 Cross Entropy Loss 함수의 확장판**이라고 말할 수 있습니다.



<br>







<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
