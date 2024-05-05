---
layout: post
title: 최적화 이론 기초 정리 (Gradient Descent, Newton Method, Gauss-Newton, LM 등)
date: 2024-05-05 00:00:00
img: math/calculus/jacobian/jacobian.png
categories: [math-calculus] 
tags: [Jacobian, 자코비안] # add tag
---

<br>

- 참조 : http://t-robotics.blogspot.com/2013/12/jacobian.html#.XGlnkegzaUk
- 참조 : https://suhak.tistory.com/944

<br>

- 자코비안은 다양한 문제에서 `approximation` 접근법을 사용할 때 자주 사용 되는 방법입니다.
- 예를 들어 비선형 칼만필터를 사용할 때, 비선형 식을 선형으로 근사시켜서 모델링 할 때 사용하는 **Extended Kalman Filter**가 대표적인 예가 될 수 있습니다.
- 자코비안은 정말 많이 쓰기 때문에 익혀두면 상당히 좋습니다. 이 글에서 자코비안에 대하여 다루어 보도록 하겠습니다.

<br>

<br>
<center><img src="../assets/img/math/calculus/jacobian/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서 말했듯이 `자코비안의 목적`은 복잡하게 얽혀있는 식을 미분을 통하여 `linear approximation` 시킴으로써 간단한 `근사 선형식`을 만들어 주는 것입니다.
- 위 그래프에서 미분 기울기를 통하여 $$ \Delta x $$ 후의 y값을 `선형 근사`하여 예측하는 것과 비슷한 원리 입니다.
- 그런데 위 그래프에서 가장 알고 싶은 것은 $$ f'(x_{1}) $$ 에서의 함수 입니다.  
- 물론 $$ y = f(x) $$와 같은 `1변수 함수`에서는 미분값도 스칼라 값이 나오기도 합니다. 
- 하지만 $$ x = (x_{1}, x_{2}, ...), y = (y_{1}, y_{2}, ...) $$와 같이 일반화한 경우 미분값이 스칼라 값이 아니라 `행렬`형태로 나오게 됩니다.

<br>
<center><img src="../assets/img/math/calculus/jacobian/2.png" alt="Drawing" style="width: 300px;"/></center>
<br>
    
- 여기서 `J`가 앞의 그래프 예시에 있는 함수 $$ f'(x) $$ 입니다.

<br>
 
- 위키피디아의 Jacobian 행렬 정의를 찾아보면 **"The Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function"** 으로 나옵니다. 
- 즉, 자코비안 행렬은 모든 벡터들의 `1차 편미분값`으로된 행렬로 각 행렬의 값은 **다변수 함수일 때의 미분값**입니다.

<br>
    
- 첫 그래프를 보고 `자코비안을 구한 이유`에 대해서 다시 생각해 보면, $$ q = f^{-1}(x) $$를 구하기 어렵기 때문에 선형화 하여 근사값을 구한 것입니다. 따라서 이 문제를 자코비안의 역행렬을 이용하여 푼다면 근사해를 구할 수 있습니다.

<br>
    
- $$ dx = Jdq $$

- $$ dq = J^{-1}dx $$

- $$ \therefore q_{2} = q_{1} + J^{-1}dx  $$
    
<br>

- 이번에는 다른 방법을 통하여 자코비안의 사용 예시를 설명드리겠습니다.

<br>

- $$ \int_{a}^{b}f(g(x))g^{\prime}(x)dx=\int_{g(a)}^{g(b)}f(u)du \quad\quad( u=g(x)\; \iff \;du=g^{\prime}(x)dx) $$

<br>

- $$ \begin{split}\int_{0}^{1} \sqrt{1-x^2} d x  &=\int_{0}^{\pi/2} \sqrt{1-\sin^2 \theta} \cos \theta d \theta \quad (x=\sin \theta\; \iff \;dx=\cos \theta d \theta) \\ &=\int_{0}^{\pi/2} \cos^2 \theta d \theta =\int_{0}^{\pi/2} \frac{1}{2} (1+\cos 2 \theta) d \theta \\ &= \frac{1}{2} \left[ \theta + \frac{1}{2} \sin 2 \theta \right]_0^{\pi/2}=\frac{\pi}{4} \end{split} $$

- $$ \cos{2\theta} = 2\cos{\alpha}^{2} - 1 $$
