---
layout: post
title: 자코비안 
date: 2019-02-06 00:00:00
img: math/calculus/jacobian/jacobian.png
categories: [math-calculus] 
tags: [Jacobian, 자코비안] # add tag
---

+ 출처 : 
    + http://t-robotics.blogspot.com/2013/12/jacobian.html#.XGlnkegzaUk

+ 복잡하게 얽혀있는 식을 미분을 통하여 `linear approximation` 시킴으로써 간단한 근사 선형식을 만들어 주는 것입니다.
+ 미분 기울기를 통하여 $$ \Delta x $$ 후의 y값을 선형 근사하여 예측하는 것과 비슷한 원리 입니다.

<br>
<center><img src="../assets/img/math/calculus/jacobian/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 여기서 알고 싶은 것은 $$ f'(x_{1}) $$ 에서의 함수 입니다.  
- 　$$ y = f(x) $$와 같은 1변수 함수에서는 미분값도 스칼라 값이 나오기도 합니다.
- 만약 $$ x = (x_{1}, x_{2}, ...), y = (y_{1}, y_{2}, ...) $$와 같은 경우 미분값이 스칼라 값이 아니라 `행렬`형태로 나오게 됩니다.

<br>
<center><img src="../assets/img/math/calculus/jacobian/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>
    
- 여기서 `J`가 앞의 그래프 예시에 있는 함수 $$ f'(x) $$ 입니다.
- 식을 정리하면 $$ y_{1} + Jdq $$가 됨을 확인할 수 있습니다.
 
- 위키피디아의 Jacobian 행렬 정의는 다음과 같습니다.
- "The Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function" 
- 자코비안 행렬은 모든 벡터들의 `first-order 편미분값`으로된 행렬 입니다. **다변수 함수일 때의 미분값**입니다.

<br>
    
- 첫 그래프를 보고 `자코비안을 구한 이유`에 대해서 다시 생각해 보면, $$ q = f^{-1}(x) $$ 를 구하기 어려웠기 때문이었습니다.
- 이 문제를 자코비안의 역행렬을 이용하여 푼다면 근사해를 구할 수 있습니다.

<br>
    
$$ dx = Jdq $$

$$ dq = J^{-1}dx $$

$$ \therefore q_{2} = q_{1} + J^{-1}dx  $$
    
<br>
