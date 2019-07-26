---
layout: post
title: Eienstein summation convention  
date: 2018-09-27 15:00:00
img: math/la/overall.jpg
categories: [math-la] 
tags: [Linear algebra, eienstein summation convetion] # add tag
---

- 이번 글에서는 Eienstein summation convention에 대하여 간략하게 다루어 보겠습니다.
- summation을 조금 더 간편하게 표기하기 위한 방법이라고 생각하면 됩니다.
- 먼저 아래와 같은 행렬곱의 수식이 있다고 가정해 보겠습니다.

$$ 

\begin{pmatrix}
a_{11} & a_{11} & \cdots \ a_{1n} \\
a_{21} & a_{22} & \cdots \ a_{21} \\
\cdots & \cdots & \cdots \ \cdots \\
a_{n1} & a_{n2} & \cdots \ a_{nn} \\
\end{pmatrix}

\begin{pmatrix}
b_{11} & b_{11} & \cdots \ b_{1n} \\
b_{21} & b_{22} & \cdots \ b_{21} \\
\cdots & \cdots & \cdots \ \cdots \\
b_{n1} & b_{n2} & \cdots \ b_{nn} \\
\end{pmatrix}

$$

- 이 때 행렬 곱의 한 원소를 구하려면 $$ \sum_{j}a_{ij}b_{jk} $$ 식을 통하여 구해야 합니다.
- 이 식을 간단하게 표현하자고 도입한 것이 Eienstein summation convention 입니다.
- 즉, $$ ab_{ik} = \sum_{j}a_{ij}b_{jk} $$ 로 표현하여 중간에 계산 과정 상 필요한 $$ j $$ 를 생략하는 방법입니다.
- 예를 들면 $$ ab_{23} = a_{21}b_{13} + a_{22}b_{23} + \cdots + a_{2n}b_{n3} $$ 이 됩니다.

<br>

- 또한 벡터의 곱을 나타낼 때에도 간단하게 표현할 수 있습니다.
- 예를 들어 $$ u = [u_{1}, u_{2}, \cdots, u_{n}] $$ 이고 $$ v = [v_{1}, v_{2}, \cdots, v_{n}] $$ 이면 두 벡터의 내적은 다양하게 표현될 수 있습니다.
    - 먼저 계산 과정 그대로 $$ [u_{1}, u_{2}, \cdots, u_{n}]*[u_{1}, u_{2}, \cdots, u_{n}]^{T} $$이 될 수 있습니다.
    - 그냥 간단하게 $$ u \cdot v $$로 표현할 수도 있습니다.
    - 마지막으로 Eienstein summation convetion을 따르면 **$$ u_{i}v_{i} $$**로 표현 가능합니다.

 
