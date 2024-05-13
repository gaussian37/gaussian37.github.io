---
layout: post
title: 최적화 이론 기초 정리 (Gradient Descent, Newton Method, Gauss-Newton, Levenberg-Marquardt 등)
date: 2024-05-05 00:00:00
img: math/calculus/jacobian/jacobian.png
categories: [math-calculus] 
tags: [Jacobian, 자코비안] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 사전 지식 필요 : [최소제곱법](https://gaussian37.github.io/math-la-least_squares/)
- 사전 지식 필요 : [Gradient, Jacobian, Hessian 개념](https://gaussian37.github.io/math-calculus-jacobian/)
- 참조 : https://youtu.be/C6DCtQjKkdY

<br>

- 이번 글에서는 최적화 기법을 이용할 때, 가장 기본적으로 사용되는 기법들에 대하여 다루어 보려고 합니다. 본 글에서 다룬 내용은 최적화 기법의 가장 기초적인 내용이며 풀고자 하는 문제에 맞게 각 기법들을 적용하여 사용하시면 됩니다.
- 본 글을 읽기 전에 위에 명시되어 있는 [최소제곱법](https://gaussian37.github.io/math-la-least_squares/)과 [Gradient, Jacobian, Hessian 개념](https://gaussian37.github.io/math-calculus-jacobian/) 개념 숙지가 필요한 점 참조 부탁 드립니다.

<br>

## **목차**

<br>

- ### [Gradient Descent](#gradient-descent-1)
- ### [SGD](#sgd-1)
- ### [Momentum](#momentum-1)
- ### [RMSProp](#rmsprop-1)
- ### [Adam](#adam-1)
- ### [Newton Method](#newton-method-1)
- ### [Newton Method for Non-Linear Least Squares](#newton-method-1)
- ### [Gauss-Newton Method for Non-Linear Least Squares](#gauss-newton-method-for-non-linear-least-squares-1)
- ### [Levenberg-Marquardt Method for Non-Linear Least Squares](#levenberg-marquardt-method-for-non-linear-least-squares-1)
- ### [Quasi Newton Method for Non-Linear Least Squares](#quasi-newton-method-for-non-linear-least-squares-1)
- ### [Lagrange Multiplier](#lagrange-multiplier-1)

<br>

## **Gradient Descent**

<br>

<br>

## **SGD**

<br>

<br>

## **Momentum**

<br>

<br>

## **RMSProp**

<br>

<br>

## **Adam**

<br>

<br>

## **Newton Method**

<br>

- 참조 : https://gaussian37.github.io/math-mfml-intro_to_optimisation/#newton-raphson-method-1

<br>

## **Newton Method for Non-Linear Least Squares**

<br>

- 앞에서 배운 `newton method`에서는 $$ f(x) = 0 $$ 의 해를 찾기 위하여 다음과 같이 식을 정의 후 iteration을 통하여 $$ f(x) = 0 $$ 를 만족하는 해를 점진적으로 찾아 갔습니다. 

<br>

- $$ x_{i+1} = x_{i} - \frac{f(x_{i})}{f'(x_{i})} $$

<br>

- 문제의 관점을 조금 바꾸어서 단순히 $$ f(x) = 0 $$ 의 $$ f(x) $$ 가 임의의 다항식이 아니라 `Objective Function`이라고 가정해 보겠습니다. 물론 `Objective Function` 또한 앞에서 정의한 다항식과 동일하게 표현될 수 있지만 의미상으로 정답 $$ y_{i} $$ 와 예측값 $$ \hat{y}_{i} $$ 간의 차이를 정의한 함수라고 생각해 보겠습니다.
- 이번에 다룰 예제 함수는 다음과 같은 함수입니다.

<br>

- $$ y = a \cdot e^{-kt} $$

<br>

- 위 식에서 $$ a $$ 는 상수이고 구하고자 하는 파라미터는 $$ k $$ 입니다. $$ t $$ 가 입력값이고 $$ y $$ 가 출력값입니다.
- 위 함수식을 이용하여 `Objective Function`을 `SSE(Sum of Square Error)` 형태로 아래와 같이 정의하겠습니다. 사용되는 데이터의 갯수는 $$ n $$ 개로 이번 글에서는 8개의 데이터 샘플을 사용할 예정입니다.

<br>

- $$ \text{SSE} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2} = \sum_{i=1}^{n}(y_{i} - a e^{-kt_{i}}) $$

<br>

- `SSE`는 `convex function`으로 아래로 볼록한 형태를 가지게 됩니다. 따라서 `critical point` (임계점)을 찾기 위해 `SSE`의 1차 미분이 0이되는 지점을 찾는 문제로 변환하여 풀 수 있습니다.
- `newton metehod`를 이용하여 이 문제를 접근한다면 다음과 같이 식을 정의할 수 있습니다.

<br>

- $$ k_{\text{new}} = k_{\text{old}} - \frac{f'(k_{\text{old}})}{f''(k_{\text{old}})} $$

- $$ \text{SSE} = \sum_{i} (y_{i} - a e^{-kt_{i}}) $$

- $$ \frac{\partial \text{SSE}}{\partial k} = \sum_{i=1}^{n}2(y_{i} - ae^{-kt_{i}}) \cdot t_{i}ae^{-kt_{i}} $$

- $$ \frac{\partial^{2} \text{SSE}}{\partial k^{2}} = \sum_{i=1}^{n} -2y_{i}t_{i}t_{i}a e^{-kt_{i}} - ((2ae^{-kt_{i}})(-t_{i}t_{i}ae^{-kt_{i}}) + (-2t_{i}ae^{-kt_{i}})(t_{i}ae^{-kt_{i}})) $$

<br>

- 위 식들을 이용하여 `newton method`를 적용하면 다음과 같습니다.

<br>

- $$ k_{\text{new}} = k_{\text{old}} - \frac{\left(\frac{\partial \text{SSE}}{\partial k}\right)}{\left(\frac{\partial^{2}\text{SSE}}{\partial k^{2}}\right)} $$

<br>

- 단순히 1차 미분과 2차 미분을 이용하여 $$ k_{\text{old}} \to k_{\text{new}} $$ 로 업데이트 한다는 것에서 앞에서 살펴본 다차 방정식의 해를 구하는 것과 큰 차이는 없습니다.

<br>

- 이번에는 위 함수에서 상수 $$ a $$ 를 파라미터로 변경하여 다음과 같이 2개의 변수 $$ k = (k_{0}, k_{1}) $$ 를 가지는 `multivariable function`으로 변경해 보도록 하겠습니다. 
- 최종적으로 구하고자 하는 형태가 `vector-valued multivariable function`에서의 최적화이기 때문입니다.

<br>

- $$ y = k_{0} \cdot e^{-k_{1}t} $$

<br>

- $$ k_{\text{new}} = k_{\text{old}} - \frac{f'(k_{\text{old}})}{f''(k_{\text{old}})} $$

- $$ \begin{bmatrix}k_{{0}_{\text{new}}} \\ k_{{1}_{\text{new}}} \end{bmatrix} = \begin{bmatrix} k_{{0}_{\text{old}}} \\ k_{{1}_{\text{old}}} \end{bmatrix} - H^{-1}G $$

- $$ H (\text{Hessian}) = \begin{bmatrix} \frac{\partial^{2}f}{\partial k_{0}^{2}} & \frac{\partial^{2}f}{\partial k_{0} \partial k_{1}} \\ \frac{\partial^{2}f}{\partial k_{1} \partial k_{0}} & \frac{\partial^{2}f}{\partial k_{1}^{2}} \end{bmatrix} $$

- $$ G (\text{Gradient}) = \begin{bmatrix} \frac{\partial f}{\partial k_{0}} \\ \frac{\partial f}{\partial k_{1}} \end{bmatrix} $$

<br>

- 위 식에서 $$ H^{-1} $$ 과 $$ G $$ 는 `newton method`에서 각각 아래 값과 대응됩니다.

<br>

- $$ f'(k) = G $$

- $$ \frac{1}{f''(k)} = H^{-1} $$

<br>

- 작성중 ... 

<br>

## **Gauss-Newton Method for Non-Linear Least Squares**

<br>

- 

<br>

## **Levenberg-Marquardt Method for Non-Linear Least Squares**

<br>

<br>

## **Quasi Newton Method for Non-Linear Least Squares**

<br>

<br>


<br>

## **Lagrange Multiplier**

<br>

<br>


<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>