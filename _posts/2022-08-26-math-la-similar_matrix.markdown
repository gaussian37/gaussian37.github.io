---
layout: post
title: Similar Matrix (닮은 행렬)
date: 2022-08-26 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [similar matrix, 닮은 행렬] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 닮은 행렬에 대하여 간단하게 살펴보도록 하겠습니다. 닮은 행렬 끼리는 `rank`, `trace`, `determinant`, `eigenvalue` 등이 보존되기 때문에 이와 같은 성질을 이용하고자 할 때, 자주 사용됩니다.

<br>

## **닮은 행렬의 정의**

<br>

- 먼저 닮은 행렬이란 다음과 같은 관계를 가질 때, $$ A $$ 와 $$ B $$ 의 관계를 의미합니다.
- 아래 식에서 $$ A, B, P $$ 모두 $$ K \times K $$ 의 크기를 가지는 `정사각행렬`을 의미합니다.

<br>

- $$ B = P^{-1} A P  \tag{1} $$

<br>

- 즉, 행렬 $$ A $$ 를 기준으로 좌우에 임의의 $$ K \times K $$ 크기의 정사각행렬 $$ P^{-1}, P $$ 를 곱한 것을 $$ B $$ 라고 하고 두 행렬 $$ A, B $$ 는 닮은 행렬 이라고 합니다.
- 그리고 이와 같은 닮은 행렬을 만드는 연산을 `similarity transformation` 이라고 합니다. 
- 그러면 두 행렬 $$ A, B $$ 가 보존하는 성질을 차례 대로 살펴보도록 하겠습니다. 살펴보는 순서는 `① reflexivity`, `② symmetry`, `③ transitivity`, `④ same rank`, `⑤ same trace`, `⑥ same determinant`, `⑦ same eigenvalues` 입니다.
- 주의할 점은 `닮은 행렬`이면 `① ~ ⑦`을 만족하는 것이지 그 역은 성립하지 않는 다는 점입니다.

<br>

## **① reflexivity**

<br>

- $$ A = I^{-1} A I \tag{2} $$

<br>

- 항등 행렬을 이용하여 $$ A $$ 의 닮은 행렬을 표현하면 그대로 $$ A $$ 로 표현할 수 있습니다. 따라서 `reflexivity`를 만족합니다.

<br>

## **② symmetry**

<br>

- $$ B = P^{-1} A P $$

- $$ A = (P^{-1})^{-1} B P^{-1} \tag{3} $$

<br>

- 따라서 $$ A $$ 가 $$ B $$ 의 닮은 행렬이람련 $$ B $$ 또한 $$ A $$ 의 닮은 행렬임을 알 수 있습니다.

<br>

## **③ transitivity**

<br>

- $$ B = P_{1}^{-1} A P_{1} \tag{4} $$

- $$ C = P_{2}^{-1} B P_{2} \tag{5} $$

- $$ C = (P_{1}P_{2})^{-1} A P_{1}P_{2} \tag{6} $$

<br>

- 식 (6)을 통해 $$ A $$ 와 $$ B $$ 가 닮음이고 $$ B $$ 와 $$ C $$ 가 닮음이면 $$ A $$ 와 $$ C $$ 도 닮음임을 알 수 있습니다.

<br>

## **④ same rank**

<br>

- 행렬 $$ P $$ 는 `invertible` 하므로 `full rank` 행렬입니다. 따라서 다음과 같이 전개할 수 있습니다.

<br>

- $$ \text{rank}(AP) = \text{rank}(A) \tag{7} $$

- $$ \text{rank}(B) = \text{rank}(P^{-1}AP) = \text{rank}(AP) \tag{8} $$

<br>

## **⑤ same trace**

<br>

- `trace`는 행렬에서 주대각성분의 합입니다. 아래와 같은 전개를 통하여 `similarity transformation`에서 `trace`는 유지됩니다.

<br>

- $$ B = P^{-1} A P $$

- $$ \begin{align} \text{tr}(B) &= \text{tr}(P^{-1}AP) \\ &= \text{tr}(P^{-1}(AP)) \\ &= \text{tr}((AP)P^{-1}) \\ &= \text{tr}(A(PP^{-1})) \\ &= \text{tr}(A) \end{align} \tag{9} $$

<br>

## **⑥ same determinant**

<br>

- 닮은 행렬 $$ A, B $$ 의 `determinant`는 아래 식과 같은 전개를 통해 그대로 유지 됩니다.

<br>

- $$ B = P^{-1} A P $$

- $$ \begin{align} \text{det}(B) &= \text{det}(P^{-1}AP) \\ &= \text{det}(P^{-1})\text{det}(A)\text{det}(P) \\ &= \text{det}(A) \quad (\because \text{det}(P^{-1}) = 1 / \text{det}(P)) \end{align} \tag{10} $$

<br>

## **⑦ same eigenvalues**

<br>

- 마지막으로 닮은 행렬 $$ A, B $$ 의 `eigenvalue`가 같음을 보이도록 하겠습니다. 2가지 방법으로 증명할 예정입니다. 첫번째는 `eigenvalue의 정의를 이용`하는 방식이고 두번째는 `characteristic equation`의 `determinant`를 이용하는 방식을 통해 $$ A $$ 와 $$ B $$ 의 `eigenvalue`가 같음을 보여줍니다.

<br>

#### ****eigenvalue의 정의를 이용**

<br>

- $$ Ax = \lambda x $$

- $$ P^{-1}BP x = \lambda x $$

- $$ B(Px) = \lambda (Px) $$

- $$ By = \lambda y \quad (y = Px) \tag{11} $$

<br>

- 식 (11)을 통하여 $$ B $$ 의 `eigenvalue`는 그대로 $$ \lambda $$ 인 것을 확인할 수 있었고 `eigenvector`는 $$ x \to Px = y $$ 로 변경된 것을 확인할 수 있습니다.

<br>

#### **characteristic equation의 determinant 이용**

<br>

- $$ Ax = \lambda x $$

- $$ (\lambda I - A)x = 0 $$

- $$ \text{det}(\lambda I - A) = 0  \tag{12} $$

<br>

- $$ \begin{align} \text{det}(\lambda I - B) &= \text{det}(\lambda P^{-1}P - P^{-1}AP) \\ &= \text{det}(P^{-1}(\lambda I - A)P) \\ &= \text{det}(P^{-1}) \text{det}(\lambda I - A) \text{det}(P) \\ &= \text{det}(\lambda I - A) \end{align} \tag{13} $$

<br>

- 식 (13)를 통하여 $$ \text{det}(\lambda I - B) = \text{det}(\lambda I - A) $$ 를 만족하므로 닮은 행렬에서 `eigenvalue` $$ \lambda $$ 는 같은것 을 알 수 있습니다.

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>
