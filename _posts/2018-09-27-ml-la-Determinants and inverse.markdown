---
layout: post
title: Determinants and inverse  
date: 2018-09-27 14:10:00
img: math/la/overall.jpg
categories: [math-mfml] 
tags: [Linear algebra, coursera, mathematics for machine learning] # add tag
---

![1](../assets/img/math/la/Determinants and inverse/Determinants and inverse1.png)

![2](../assets/img/math/la/Determinants and inverse/Determinants and inverse2.png)

<br>

- `determinant`의 기하학적 의미는 위 그림과 같이 2D와 관련된 2 x 2 행렬에서는 넓이를 나타내고 3 x 3 행렬에서는 부피를 나타냅니다.
- 따라서 2 x 2 행렬에서 `determinant`가 0 이라면 행렬이 나타내는 면적이 0이라는 뜻입니다. 기하학적으로 보면 행렬이 span 하는 공간이 2차원을 만들어 내지 못한다는 의미가 됩니다. 따라서 선분이 되거나 점이 된다는 의미입니다. 
- 간단하게 표현하면 `rank`가 2보다 작다는 것을 의미합니다. column 벡터 또는 row 벡터가 independent 하지 못하다고도 말할 수 있습니다.

<br>

- 반면 3 x 3 행렬에서 `determinant`가 0이라면 행렬이 나타내는 부피가 0이라는 뜻입니다. 행렬이 span 하는 공간이 3차원을 만들어 내지 못하는 것이며 면을 의미하거나 선, 점을 의미할 수도 있습니다.
- 간단하게 표현하면 `rank`가 3보다 작다는 것을 의미합니다.