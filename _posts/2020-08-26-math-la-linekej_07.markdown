---
layout: post
title: 행렬식과 행렬식의 성질
date: 2020-08-26 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [선형대수학, 행렬식, determinant, det] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 먼저 행렬식의 대표적인 성질에 대하여 간략하게 정리하도록 하겠습니다. 아래 성질은 행렬식 계산에 많이 사용되니 숙지 및 참조하시면 도움이 됩니다. 아래 10가지 성질은 행렬식의 정의와 추가적인 선형대수학의 개념을 알아야 이해가 되는 부분이므로 스킵하셔도 됩니다.
- 아래 성질은 행렬 $$ A $$ 의 크기가 $$ n \times n $$ 이라고 가정합니다.

<br>

- ① $$ \text{det}(A) = 0 \iff  A \text{ is singular} \tag{1} $$ 
- ② $$ A \text{ is rank-deficient} \iff \text{det}(A) = 0 \tag{2} $$
- ③ $$ \text{For diagonal matrix}, \text{det}(A) = a_{11}a_{22}\cdots a_{nn} \tag{3} $$
- ④ $$ \text{For triangular matrix}, \text{det}(A) = a_{11}a_{22}\cdots a_{nn} \tag{4} $$
- ⑤ $$ \text{det}(I) = 1 \tag{5} $$
- ⑥ $$ \text{det}(cA) = c^{n}\text{det}(A) \tag{6} $$
- ⑦ $$ \text{det}(A^{T}) = \text{det}(A) \tag{7} $$
- ⑧ $$ \text{det}(AB) = \text{det}(A)\text{det}(B) \tag{8} $$
- ⑨ $$ \text{det}(A^{-1}) = \frac{1}{\text{det}(A)} \tag{9} $$
- ⑩ $$ \text{det}(A) = \lambda_{1}\lambda_{2} \cdots \lambda_{n} $$

<br>

- ①은 행렬식이 0이라면 역행렬이 없다는 것을 의미합니다.
- ②는 행렬이 full rank가 아니라면, 즉 자유 변수가 있는 상태라면 행렬식이 0이라는 의미입니다.
- ③은 대각 행렬의 경우 행렬식은 대각 성분의 곱을 이용하여 구할 수 있음을 의미합니다. 즉, 대각행렬의 대각 성분 중 1개라도 0이 있으면 행렬식은 0이 됩니다.
- ④는 대각 행렬 뿐만 아니라 삼각행렬 (상삼각행렬, 하삼각행렬)의 경우에도 대각 성분의 곱으로 행렬식을 구할 수 있음을 의미합니다.
- ⑤는 항등행렬의 경우 행렬식이 1임을 의미합니다. 이는 ③을 통해서도 유도할 수 있습니다.
- ⑨는 ⑧을 이용하여 유도할 수 있습니다. 즉, $$ 1 = \text{det}(A) = \text{det}(AA^{-1}) = \text{det}(A)\text{det}(A^{-1}) $$ 가 됩니다.
- ⑩은 행렬 $$ A $$ 의 행렬식은 모든 고유값의 곱을 통해 구할 수 있음을 의미합니다. 즉 고유값 중 0인 값이 있으면 행렬식은 0이 됩니다.

<br>

- 위 내용은 자주 쓰이는 행렬식의 성질을 나타낸 것이고 지금부터 행렬식의 정의에 대하여 차근차근 살펴보도록 하겠습니다.

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>
