---
layout: post
title: 딥러닝에서의 다양한 Normalization 방법
date: 2020-11-23 00:00:00
img: dl/concept/various_normalization/0.png
categories: [dl-concept]
tags: [normalization, 정규화] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 참조 : fastcampus
- 참조 : https://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/

<br>


## **Normalization과 Regularization**

<br>

- `Normalization` : **데이터**에 scale을 조정하는 작업
- `Regularization` : **predict function**에 복잡도를 조정하는 작업

<br>

- 예를 들어 데이터가 매우 다른 scale(특히 낮은 범위에서 높은 범위)에있는 경우 데이터를 `Normalization` 할 수 있습니다. 표준 편차 및 표준 편차와 같은 동일한 (또는 호환 가능한) 기본 통계를 갖도록 데이터를 변경합니다. 평균. 이는 학습된 모델이 정확도를 손상시키지 않고 처리 할 수있는 scale로 피팅 매개 변수를 유지하는 데 유용합니다.

<br>

- 모델 학습의 한 가지 목표는 중요한 feature를 식별하고 noise(모델의 최종 목적과 실제로 관련이 없는 random variation)를 무시하는 것입니다. 주어진 데이터에 대한 오류를 최소화하기 위해 모델을 자유롭게 조정하는 경우 과적합이 될 수 있습니다. 모델은 이러한 임의의 변형을 포함하여 데이터 세트를 정확하게 예측해야합니다.
- `Regularization`은 복잡한 함수보다 더 간단한 피팅 함수에 보상(reward)을 합니다. 예를 들어, RMS 에러가 x 인 단순 로그 함수가 오류가 x / 2 인 15 차 다항식보다 낫다고 말할 수 있습니다. 이러한 모델 단순화의 정도와 에러에 대한 트레이드 오프 조정은 모델 개발자에게 달려 있습니다.





<br>
<center><img src="../assets/img/dl/concept/various_normalization/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>




<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>