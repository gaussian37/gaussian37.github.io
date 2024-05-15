---
layout: post
title: 최적화 이론 기초 정리 (Gradient Descent, Newton Method, Gauss-Newton, Levenberg-Marquardt 등)
date: 2024-05-05 00:00:00
img: math/calculus/basic_optimization/0.png
categories: [math-calculus] 
tags: [Gradient Descent, SGD, Momentum, RMSProp, Adam, Newton, Gauss Newton, Levenberg-Marquardt, Quasi Newton, Lagrange Multiplier, Jacobian, 자코비안] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 사전 지식 필요 : [최소제곱법](https://gaussian37.github.io/math-la-least_squares/)
- 사전 지식 필요 : [Gradient, Jacobian, Hessian 개념](https://gaussian37.github.io/math-calculus-jacobian/)
- 참조 : https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf
- 참조 : https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504
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
- ### [Quadratic approximation](#quadratic-approximation-1)
- ### [Newton Method for Optimization](#newton-method-for-optimization-1)
- ### [Gauss-Newton Method for Non-Linear Least Squares](#gauss-newton-method-for-non-linear-least-squares-1)
- ### [Levenberg-Marquardt Method for Non-Linear Least Squares](#levenberg-marquardt-method-for-non-linear-least-squares-1)
- ### [Weighted Residuals](#weighted-residuals-1)
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

## **Quadratic approximation**

<br>

<br>

## **Newton Method for Optimization**

<br>

- 앞에서 배운 `newton method`에서는 $$ f(x) = 0 $$ 의 해를 찾기 위하여 다음과 같이 식을 정의 후 iteration을 통하여 $$ f(x) = 0 $$ 를 만족하는 해를 점진적으로 찾아 갔습니다. 

<br>

- $$ x_{i+1} = x_{i} - \frac{f(x_{i})}{f'(x_{i})} $$

<br>

- 문제의 관점을 조금 바꾸어서 단순히 $$ f(x) = 0 $$ 의 $$ f(x) $$ 가 임의의 다항식이 아니라 `Objective Function`이라고 가정해 보겠습니다. 물론 `Objective Function` 또한 앞에서 정의한 다항식과 동일하게 표현될 수 있지만 의미상으로 정답 $$ y_{i} $$ 와 예측값 $$ \hat{y}_{i} $$ 간의 차이를 정의한 함수라고 생각해 보겠습니다.
- 이번에 다룰 예제 함수는 다음과 같은 함수입니다.

<br>

- $$ y = a \cdot e^{-wt} $$

<br>

- 위 식에서 $$ a $$ 는 상수이고 구하고자 하는 파라미터는 $$ w $$ 입니다. $$ t $$ 가 입력값이고 $$ y $$ 가 출력값입니다.
- 위 함수식을 이용하여 `Objective Function`을 `SSE(Sum of Square Error)` 형태로 아래와 같이 정의하겠습니다. 사용되는 데이터의 갯수는 $$ n $$ 개로 이번 글에서는 8개의 데이터 샘플을 사용할 예정입니다.

<br>

- $$ \text{SSE} = \sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^{2} = \sum_{i=1}^{n}(y_{i} - a e^{-wt_{i}}) $$

<br>

- `SSE`는 `convex function`으로 아래로 볼록한 형태를 가지게 됩니다. 따라서 `critical point` (임계점)을 찾기 위해 `SSE`의 1차 미분이 0이되는 지점을 찾는 문제로 변환하여 풀 수 있습니다.
- `newton metehod`를 이용하여 이 문제를 접근한다면 다음과 같이 식을 정의할 수 있습니다.

<br>

- $$ w_{\text{new}} = w_{\text{old}} - \frac{f'(w_{\text{old}})}{f''(w_{\text{old}})} $$

- $$ \text{SSE} = \sum_{i} (y_{i} - a e^{-wt_{i}}) $$

- $$ \frac{\partial \text{SSE}}{\partial w} = \sum_{i=1}^{n}2(y_{i} - ae^{-wt_{i}}) \cdot t_{i}ae^{-wt_{i}} $$

- $$ \frac{\partial^{2} \text{SSE}}{\partial w^{2}} = \sum_{i=1}^{n} -2y_{i}t_{i}t_{i}a e^{-wt_{i}} - ((2ae^{-wt_{i}})(-t_{i}t_{i}ae^{-wt_{i}}) + (-2t_{i}ae^{-wt_{i}})(t_{i}ae^{-wt_{i}})) $$

<br>

- 위 식들을 이용하여 `newton method`를 적용하면 다음과 같습니다.

<br>

- $$ w_{\text{new}} = w_{\text{old}} - \frac{\left(\frac{\partial \text{SSE}}{\partial w}\right)}{\left(\frac{\partial^{2}\text{SSE}}{\partial w^{2}}\right)} $$

<br>

- 단순히 1차 미분과 2차 미분을 이용하여 $$ w_{\text{old}} \to w_{\text{new}} $$ 로 업데이트 한다는 것에서 앞에서 살펴본 다차 방정식의 해를 구하는 것과 큰 차이는 없습니다.

<br>

- 이번에는 위 함수에서 상수 $$ a $$ 를 파라미터로 변경하여 다음과 같이 2개의 변수 $$ w = (w_{1}, w_{2}) $$ 를 가지는 `multivariable function`으로 변경해 보도록 하겠습니다. 
- 최종적으로 구하고자 하는 형태가 `vector-valued multivariable function`에서의 최적화이기 때문입니다.

<br>

- $$ y = w_{1} \cdot e^{-w_{2}t} $$

- $$ w_{\text{new}} = w_{\text{old}} - \frac{f'(w_{\text{old}})}{f''(w_{\text{old}})} $$

- $$ \begin{bmatrix}w_{0(\text{new})} \\ w_{1(\text{new})} \end{bmatrix} = \begin{bmatrix} w_{0(\text{old})} \\ w_{1(\text{old})} \end{bmatrix} - H^{-1}G $$

- $$ G (\text{Gradient}) = \begin{bmatrix} \frac{\partial f}{\partial w_{1}} \\ \frac{\partial f}{\partial w_{2}} \end{bmatrix} = \begin{bmatrix} e^{-w_{2}t} \\ -w_{1}te^{-w_{2}t} \end{bmatrix} $$

- $$ H (\text{Hessian}) = \begin{bmatrix} \frac{\partial^{2}f}{\partial w_{1}^{2}} & \frac{\partial^{2}f}{\partial w_{1} \partial w_{2}} \\ \frac{\partial^{2}f}{\partial w_{2} \partial w_{1}} & \frac{\partial^{2}f}{\partial w_{2}^{2}} \end{bmatrix} = \begin{bmatrix} 0 & -t e^{-w_{2}t} \\ -t e^{-w_{2}t} & w_{1}t^{2}e^{-w_{2}t} \end{bmatrix} $$

<br>

- 위 식에서 $$ H^{-1} $$ 과 $$ G $$ 는 `newton method`에서 각각 아래 값과 대응됩니다.

<br>

- $$ f'(w) = G $$

- $$ \frac{1}{f''(w)} = H^{-1} $$

<br>

- 따라서 `newton method`를 이용하여 `Objective Function`을 최적화 할 때, 위 식을 통하여 점진적으로 해를 구할 수 있습니다.
- 위 식을 통하여 아래 데이터에 대하여 $$ w_{1}, w_{2} $$ 을 최적화 해보려고 합니다.

<br>

```python
t = [0, 20, 40, 60, 80, 100, 120, 140]
y = [147.8, 78.3, 44.7, 29.5, 15.2, 7.8, 3.2, 3.9]
```

<br>

- 위 데이터는 총 8개의 쌍으로 다음과 같이 기호로 적을 수 있습니다.

<br>

- $$ \begin{bmatrix} x_{0} & y_{0} \\ x_{1} & y_{1} \\ x_{2} & y_{2} \\ x_{3} & y_{3} \\ x_{4} & y_{4} \\ x_{5} & y_{5} \\ x_{6} & y_{6} \\ x_{7} & y_{7} \end{bmatrix}  = \begin{bmatrix} 0 & 147.8 \\ 20 & 78.3 \\ 40 & 44.7 \\ 60 & 29.5 \\ 80 & 15.2 \\ 100 & 7.8 \\ 120 & 3.2 \\ 140 &3.9 \end{bmatrix} $$

<br>

- 그런데 `newton method`를 이용하여 최적화 하려고 하면 각 데이터에 대하여 `Hessian`을 구해야 하는 복잡함이 발생합니다. 따라서 각 행마다 다음과 같이 `Hessian`을 구하여 표현해 주어야 합니다.

<br>

- $$ H_{0} = \begin{bmatrix} 0 & -(0) e^{-w_{2}(0)} \\ -(0) e^{-w_{2}(0)} & w_{1}(0)^{2}e^{-w_{2}(0)} \end{bmatrix} $$

- $$ H_{1} = \begin{bmatrix} 0 & -(20) e^{-w_{2}(20)} \\ -(20) e^{-w_{2}(20)} & w_{1}(20)^{2}e^{-w_{2}(20)} \end{bmatrix} $$

- $$ \vdots $$

- $$ H_{7} = \begin{bmatrix} 0 & -(140) e^{-w_{2}(140)} \\ -(140) e^{-w_{2}(140)} & w_{1}(140)^{2}e^{-w_{2}(140)} \end{bmatrix} $$

<br>

- 예를 들어 `least squares`와 같은 경우에는 $$ Ax = b $$ 에서 $$ x = (A^{T}A)^{-1}A^{T} b $$ 와 같이 해를 구할 때, 모든 데이터 성분을 2차원 행렬 ( $$ (n, m) $$ ) $$ A $$ 와 벡터 $$ b $$ 에 표현합니다.
- 하지만 `newton method`를 이용하여 최적해를 구할 때에는 `Hessian`으로 인하여 하나의 식으로 표현한다면 2차원 행렬인 `Hessian`을 행의 요소로 가지는 3차원 텐서로 표현해주어야 하는 복잡함이 발생합니다. (`Hessian`을 이용한 `newton method`의 출력은 `matrix-valued multivariable function`이 됩니다.) 그리고 2차 편미분에 대한 실제 계산도 복잡합니다. 
- 따라서 `newton method`를 `least squares`와 같이 표현하여 계산 복잡도를 줄이는 방법이 필요합니다.

<br>

## **Gauss-Newton Method for Non-Linear Least Squares**

<br>

- `gauss-newton method`는 `optimization` 문제를 풀 때, 2차 미분을 사용하는 `Hessian` 대신에 1차 미분을 사용하는 `Gradient`를 이용하여 근사해를 구하는 방식입니다. 이 때, `Gradient`를 $$ n $$ 개의 `function`에 대하여 (`vector-valued multivariable function`) 적용하므로 `Jacobian`을 이용하게 됩니다.
- `Jacobian`을 이용하여 `gauss-newton method`에서 표현하고자 하는 것은 `least squares`([최소제곱법](https://gaussian37.github.io/math-la-least_squares/))를 반복적(`iterative`)으로 적용하는 것입니다. 
- 이 방식을 통해 `linear function`의 해를 구하는 `least squares`를 `non-linear function`의 해를 구하는 `non-linear least squares`로 표현하는 것이 `gauss-newton method`의 핵심이라고 말할 수 있습니다.
- `linear function`의 해는 `least squares`를 한번 적용하여 해 또는 근사해를 구할 수 있습니다. 반면 `non-linear function`의 해는 `linear function`에 사용되는 `least suqares`를 단 한번 적용해서 근사해를 바로 구하기 어렵기 때문에 반복적으로 적용하며 점근적으로 근사해를 찾아갑니다.

<br>

- 이번에는 `newton method for optimization` 부분에서 다룬 내용을 이어서 어떻게 `gauss-newton method`를 통해 `non-linear least squares`를 해결하는 지 살펴보도록 하겠습니다.

<br>

- $$ y = w_{1} \cdot e^{-w_{2}t} $$

- $$ \text{SSE} = \sum_{i}^{n} (y_{i} - w_{1} \cdot e^{-w_{2}t})^{2} $$

- $$ \text{Let } r_{i} = (y_{i} - w_{1} \cdot e^{-w_{2}t}). \ \ r \text{: residual} $$

- $$ \text{SSE} = \sum_{i=1}^{n} r_{i}^{2} = r^{t} r $$

<br>

- `chain rule`을 이용하여 편미분을 적용해 보도록 하겠습니다.

<br>

- $$ \frac{\partial \text{SSE}}{\partial w_{j}} = 2 \sum_{i}^{n} r_{i} \cdot \frac{\partial r_{i}}{w_{j}} \to \sum_{i}^{n} r_{i} \cdot \frac{\partial r_{i}}{w_{j}} $$

- $$ \because \text{Constant(2) are deleted. It does not affect parameter estimation.} $$

<br>

- 아래와 같이 함수의 곱의 미분법에 따라  $$ \frac{\partial^{2} \text{SSE}}{\partial w_{j} \partial w_{k}} $$ 를 정의할 수 있습니다.


- $$ r(x) = f(x)g(x) $$

- $$ r'(x) = f'(x)g(x) + f(x)g'(x) $$

<br>

- $$ \frac{\partial^{2} \text{SSE}}{\partial w_{j} \partial w_{k}} = \sum_{i}^{n} \left( \frac{\partial r_{i}}{\partial w_{j}}\frac{\partial r_{i}}{\partial w_{k}} + r_{i}\frac{\partial^{2} r_{i}}{\partial w_{j}\partial w_{k}} \right) $$

<br>

- 여기까지는 앞에서 다룬 `newton method`와 동일합니다. 앞에서 제기한 `newton method`의 문제점인 표현 및 계산의 복잡성을 단순화하고 `least squares`의 구조를 가질 수 있도록 식을 변경하기 위해서는 $$ \frac{\partial^{2} \text{SSE}}{\partial w_{j} \partial w_{k}} $$ 를 아래와 같이 단순화 해야 합니다.

<br>

- $$ \begin{align} \frac{\partial^{2} \text{SSE}}{\partial w_{j} \partial w_{k}} &= \sum_{i}^{n} \left( \frac{\partial r_{i}}{\partial w_{j}}\frac{\partial r_{i}}{\partial w_{k}} + \color{red}{r_{i}\frac{\partial^{2} r_{i}}{\partial w_{j}\partial w_{k}}} \right) \\ &\approx \sum_{i}^{n} \left( \frac{\partial r_{i}}{\partial w_{j}}\frac{\partial r_{i}}{\partial w_{k}} \right) \end{align} $$

<br>

- 현재 예제에서는 `Jacobian`이 다음과 같이 정의 됩니다.

<br>

- $$ J_{r} = \begin{bmatrix} \frac{\partial r_{1}}{\partial w_{1}} & \frac{\partial r_{1}}{\partial w_{2}} \\ \frac{\partial r_{2}}{\partial w_{1}} & \frac{\partial r_{2}}{\partial w_{2}} \\ \vdots & \vdots \\ \frac{\partial r_{n}}{\partial w_{1}} & \frac{\partial r_{n}}{\partial w_{2}} \end{bmatrix} $$

<br>

따라서 `SSE`의 1차 미분과 2차 미분의 결과는 다음과 같이 `Jacobian`으로 표현할 수 있습니다.

<br>

- $$ \frac{\partial \text{SSE}}{\partial w_{j}} \approx \sum_{i}^{n} r_{i} \cdot \frac{\partial r_{i}}{w_{j}} = J_{r}^{T} r $$

- $$ \frac{\partial^{2} \text{SSE}}{\partial w_{j} \partial w_{k}} \approx \sum_{i}^{n} \left( \frac{\partial r_{i}}{\partial w_{j}}\frac{\partial r_{i}}{\partial w_{k}} \right) = J_{r}^{T} J_{r} $$

<br>

- 위 결과값을 이용하면 `newton method`에서 사용하였던 $$ G $$ 와 $$ H $$ 는 다음과 같이 정의 됩니다.

<br>

- $$ G = \frac{\partial \text{SSE}}{\partial w_{j}} \approx J_{r}^{T} r $$

- $$ H = \frac{\partial^{2} \text{SSE}}{\partial w_{j} \partial w_{k}} \approx J_{r}^{T} J_{r} $$

<br>

- 따라서 `newton method`에서 사용하였던 파라미터 업데이트 수식을 다음과 같이 변경할 수 있습니다.

<br>

- $$ \begin{align} \begin{bmatrix}w_{0(\text{new})} \\ w_{1(\text{new})} \end{bmatrix} &= \begin{bmatrix} w_{0(\text{old})} \\ w_{1(\text{old})} \end{bmatrix} - H^{-1}G \\ &= \begin{bmatrix} w_{0(\text{old})} \\ w_{1(\text{old})} \end{bmatrix} - (J_{r}^{T} J_{r})^{-1}J_{r}^{T} r  \end{align} $$

<br>

- ... 작성 중 ...

<br>

## **Levenberg-Marquardt Method for Non-Linear Least Squares**

<br>

<br>

## **Weighted Residuals**

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