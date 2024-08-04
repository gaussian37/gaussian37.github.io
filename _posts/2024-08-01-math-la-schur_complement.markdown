---
layout: post
title: Schur Complement (슈어 보상행렬)
date: 2024-08-01 00:00:00
img: math/la/linear_algebra.jpg
categories: [math-la] 
tags: [Schur Complement, 슈어 보상행렬] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 참조 : http://ranking.uos.ac.kr/blog_page/post_src/schur-comp.pdf
- 참조 : https://gaussian37.github.io/math-la-positive_definite_matrix/
- 참조 : https://gaussian37.github.io/math-la-block_matrix_multiplication/

<br>

- 이번 글에서는 행렬 연산의 효율성을 향상시키는 방법인 `Schur Complement (슈어 보상행렬)`에 대하여 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [Schur Complements](#schur-complements-1)
- ### [A Characterization of Symmetric Positive Definite Matrices Using Schur Complements](#a-characterization-of-symmetric-positive-definite-matrices-using-schur-complements-1)
- ### [Pseudo-Inverses](#pseudo-inverses-1)
- ### [A Characterization of Symmetric Positive Semidefinite Matrices Using Schur Complements](#a-characterization-of-symmetric-positive-semidefinite-matrices-using-schur-complements-1)

<br>

## **Schur Complements**

<br>

- 다음과 같이 $$ n \times n $$ 크기의 행렬을 $$ 2 \times 2 $$ 블록 행렬 $$ M $$ 으로 표현해 보도록 하겠습니다.

<br>

- $$ M = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \tag{1} $$

- $$ A \text{ : } p \times p \text{ matrix} $$

- $$ D \text{ : } q \times q \text{ matrix} $$

- $$ n = p + q $$

- $$ B \text{ : } p \times q \text{ matrix} $$

- $$ C \text{ : } q \times p \text{ matrix} $$

<br>

- 블록 행렬 $$ M $$ 을 이용하여 선형연립방정식을 구성하면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} c \\ d \end{bmatrix} \tag{2} $$

- $$ \Rightarrow \begin{cases} Ax + By = c \\ Cx + Dy = d \end{cases} \tag{3} $$

<br>

- 만약 $$ D^{-1} $$ 가 존재한다면 식 (3)에 가우스 소거법을 적용하면 다음과 같이 식을 변경할 수 있습니다.

<br>

- $$ y = D^{-1}(d - Cx) \tag{4} $$

- $$ Ax + By = Ax + B(D^{-1}(d - Cx)) = c \tag{5} $$

- $$ (A - BD^{-1}C)x = c - BD^{-1}d \tag{6} $$

<br>

- 만약 행렬 $$ (A - BD^{-1}C) $$ 가 역행렬이 있다면 $$ x, y $$ 의 해는 다음과 같습니다.

<br>

- $$ \begin{cases} x = (A - BD^{-1}C)^{-1}(c - BD^{-1}d) \\ y = D^{-1}((d - C(A - BD^{-1}C)^{-1})(c - BD^{-1}d)) \end{cases} \tag{7} $$

<br>

- 이 때, $$ (A - BD^{-1}C) $$ 을 행렬 $$ M $$ 의 블록 행렬 $$ D $$ 의 `Schur Complement (슈어 보상행렬)`이라고 합니다. 보상행렬의 뜻은 **더 큰 행렬 내의 부분 행렬**이라는 의미입니다.
- 식 (4)에서는 $$ y $$ 를 먼저 정리한 다음 식 (5)에서 대입하는 방식을 이용하였습니다. 만약 $$ x $$ 에 대하여 먼저 정리한 다음 대입하는 방식을 적용하면 같은 방식으로 `Schur Complement`를 구할 수 있으며 행렬 $$ M $$ 의 블록 행렬 $$ A $$ 의 `Schur Compement`인 $$ (D - CA^{-1}B) $$ 를 얻을 수 있습니다.
- 이번 글에서는 식 (4)에서 $$ y $$ 를 먼저 정리한 방식으로 계속 식을 전개해 보도록 하겠습니다.

<br>

- 그 다음으로 식 (7)의 우변을 $$ c, d $$ 로 묶어서 표현해 보도록 하겠습니다.

<br>

- $$ \begin{cases} x = (A - BD^{-1}C)^{-1}\color{red}{c} - (A - BD^{-1}C)^{-1}BD^{-1}\color{blue}{d} \\ y =  -D^{-1}C(A - BD^{-1}C)^{-1}\color{red}{c} + (D^{-1} + D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1})\color{blue}{d} \tag{8} $$

<br>

- 앞에서 식 (2)에서 행렬 $$ M $$ 의 역행렬을 이용하여 표현하면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} \begin{bmatrix} c \\ d \end{bmatrix} \tag{9} $$

<br>

- 식 (9) 에서 $$ M^{-1} $$ 의 성분만 추출하기 위하여 식 (8)을 다음과 같이 정리할 수 있습니다.

<br>

- $$ \begin{align} \begin{bmatrix} x \\ y \end{bmatrix} &= \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} \begin{bmatrix} c \\ d \end{bmatrix} \\ &= \begin{bmatrix} (A - BD^{-1}C)^{-1}  & -(A - BD^{-1}C)^{-1}BD^{-1} \\ -D^{-1}C(A - BD^{-1}C)^{-1} & (D^{-1} + D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1}) \end{bmatrix} \begin{bmatrix} c \\ d \end{bmatrix} \end{align}\tag{10} $$

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1}C)^{-1}  & -(A - BD^{-1}C)^{-1}BD^{-1} \\ -D^{-1}C(A - BD^{-1}C)^{-1} & (D^{-1} + D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1}) \end{bmatrix} \tag{11} $$

<br>

- 식 (11)의 행렬을 식 (12), 식 (13) 순서로 분해할 수 있습니다.

<br>

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1}C)^{-1} & 0 \\ -D^{-1}C(A - BD^{-1}C)^{-1} & D^{-1} \end{bmatrix} \begin{bmatrix} I & -BD^{-1} \\ 0 & I \end{bmatrix} \tag{12} $$

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} I & 0 \\ -D^{-1}C & I \end{bmatrix} \begin{bmatrix} (A - BD^{-1}C)^{-1} & 0 \\ 0 & D^{-1} \end{bmatrix} \begin{bmatrix} I & -BD^{-1} \\ 0 & I \end{bmatrix} \tag{13} $$

<br>

- 식 (13)을 이용하여 행렬 $$ M $$ 을 구해보면 다음과 같습니다.

<br>

- $$ \begin{align} \begin{bmatrix} A & B \\ C & D \end{bmatrix} &= \left( \begin{bmatrix} I & 0 \\ -D^{-1}C & I \end{bmatrix} \begin{bmatrix} (A - BD^{-1}C)^{-1} & 0 \\ 0 & D^{-1} \end{bmatrix} \begin{bmatrix} I & -BD^{-1} \\ 0 & I \end{bmatrix} \right)^{-1} \\ &= \begin{bmatrix} I & -BD^{-1} \\ 0 & I \end{bmatrix} ^{-1} \begin{bmatrix} (A - BD^{-1}C)^{-1} & 0 \\ 0 & D^{-1} \end{bmatrix}^{-1} \begin{bmatrix} I & 0 \\ -D^{-1}C & I \end{bmatrix}^{-1}  \end{align} \tag{14} $$

- $$ \begin{bmatrix} I & -BD^{-1} \\ 0 & I \end{bmatrix} ^{-1} = (I \cdot I - (-BD^{-1}) \cdot 0 )^{-1} \begin{bmatrix} I & BD^{-1} \\ 0 & I \end{bmatrix} = \begin{bmatrix} I & BD^{-1} \\ 0 & I \end{bmatrix} \quad \because \text{ inverse matrix of 2 by 2 matrix.} \tag{15}$$

- $$ \begin{bmatrix} (A - BD^{-1}C)^{-1} & 0 \\ 0 & D^{-1} \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1}C) & 0 \\ 0 & D \end{bmatrix} \quad \because \text{inverse matrix of diagonal matrix.} \tag{16} $$

- $$ \begin{bmatrix} I & 0 \\ -D^{-1}C & I \end{bmatrix}^{-1}  = (I \cdot I - 0 \cdot (-D^{-1}C))^{-1} \begin{bmatrix} I & 0 \\ D^{-1}C & I \end{bmatrix} = \begin{bmatrix} I & 0 \\ D^{-1}C & I \end{bmatrix} \quad \because \text{ inverse matrix of 2 by 2 matrix.} \tag{17}$$

<br>

- 식 (15), 식 (17)에서는 $$ 2 \times 2 $$ 행렬의 역행렬을 구하는 방식에 따라 각 행렬의 역행렬을 구한 내용입니다. 식 (16)에서는 대각 행렬의 역행렬을 구한 것으로 대각 성분 각각에 역수 (역행렬)을 적용한 결과로 이해할 수 있습니다.
- 이와 같은 방식을 통해 행렬 $$ M $$ 은 식 (14)와 같이 블록 행렬 표현 방식을 통해 표현할 수 있습니다. 식 (14)의 식과 같이 표현하기 위해서는 행렬 $$ D^{-1} $$ 의 존재 여부 확인만을 필요로 합니다.

<br>

- 반면 식 (4) 대신에 $$ x $$ 에 대하여 먼저 정리한 다음 대입하는 방식을 적용하면 같은 방식으로 `Schur Complement`를 구한다면 행렬 $$ M $$ 과 $$ M^{-1} $$ 는 다음과 같이 구할 수 있습니다.

<br>

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix} = \begin{bmatrix} I & 0 \\ CA^{-1} & I \end{bmatrix} \begin{bmatrix} A & 0 \\ 0 & D - CA^{-1}B \end{bmatrix} \begin{bmatrix} I & A^{-1}B & \\ 0 & I \end{bmatrix} \tag{18} $$

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1} & -A^{-1}B(D - CA^{-1}B)^{-1} \\ -(D - CA^{-1}B)^{-1}CA^{-1} & (D - CA^{-1}B)^{-1} \end{bmatrix} \tag{19} $$

<br>

- 이 경우에는 식 (19)에서 확인할 수 있는 것과 같이 행렬 $$ A^{-1} $$ 의 존재 여부 확인만을 필요로 합니다.

<br>

- 만약 $$ A^{-1}, D^{-1} $$ 가 모두 존재하여 각 `Schur Complement`인 $$ A - BD^{-1}C $$ 와 $$ D - CA^{-1}B $$ 의 역행렬이 존재한다면 식 (11)과 식 (19) 의 값을 통해 다음 관계식을 확인할 수 있습니다.

<br>

- $$ (A - BD^{-1}C)^{-1} = A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1} \tag{20} $$

- $$ (D - CA^{-1}B)^{-1} = D^{-1} + D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1} \tag{21} $$

<br>

- 따라서 `Schur Complement`인 $$ A - BD^{-1}C $$ 와 $$ D - CA^{-1}B $$ 만으로 행렬을 정리하면 다음과 같이 정리할 수 있습니다.

<br>

- $$ \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} (A - BD^{-1}C)^{-1} & -A^{-1}B(D - CA^{-1}B)^{-1} \\ -(D - CA^{-1}B)^{-1}CA^{-1} & (D - CA^{-1}B)^{-1} \end{bmatrix} \tag{22} $$ 

<br>

#### **행렬 $$ M^{-1} $$ 정리**

<br>

- 지금까지 살펴본 내용을 한번 정리하고 넘어가겠습니다.

<br>

- $$ M = \begin{bmatrix} A & B \\ C & D \end{bmatrix} $$

- $$ A \text{ : } p \times p \text{ matrix} $$

- $$ D \text{ : } q \times q \text{ matrix} $$

- $$ n = p + q $$

- $$ B \text{ : } p \times q \text{ matrix} $$

- $$ C \text{ : } q \times p \text{ matrix} $$

<br>

- `Case 1` : $$ D \text{ is invertible.} $$

<br>

- $$ \begin{align} \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} &= \begin{bmatrix} (A - BD^{-1}C)^{-1}  & -(A - BD^{-1}C)^{-1}BD^{-1} \\ -D^{-1}C(A - BD^{-1}C)^{-1} & (D^{-1} + D^{-1}C(A - BD^{-1}C)^{-1}BD^{-1}) \end{bmatrix} \\ &= \begin{bmatrix} I & 0 \\ -D^{-1}C & I \end{bmatrix} \begin{bmatrix} (A - BD^{-1}C)^{-1} & 0 \\ 0 & D^{-1} \end{bmatrix} \begin{bmatrix} I & -BD^{-1} \\ 0 & I \end{bmatrix} \end{align} \tag{23} $$

<br>

- `Case 2` : $$ A \text{ is invertible.} $$

<br>

- $$ \begin{align} \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} &= \begin{bmatrix} A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1} & -A^{-1}B(D - CA^{-1}B)^{-1} \\ -(D - CA^{-1}B)^{-1}CA^{-1} & (D - CA^{-1}B)^{-1} \end{bmatrix} \\ &= \begin{bmatrix} I & -A^{-1}B \\ 0 & I \end{bmatrix} \begin{bmatrix} A^{-1} & 0 \\ 0 & (D - CA^{-1}B)^{-1} \end{bmatrix}  \begin{bmatrix} I & 0 \\ -CA^{-1} & I \end{bmatrix}  \end{align} \tag{24} $$

<br>

- `Case 3` : $$ A, D \text{ are invertible.} $$

<br>

- $$ \begin{align} \begin{bmatrix} A & B \\ C & D \end{bmatrix}^{-1} &= \begin{bmatrix} (A - BD^{-1}C)^{-1} & -A^{-1}B(D - CA^{-1}B)^{-1} \\ -(D - CA^{-1}B)^{-1}CA^{-1} & (D - CA^{-1}B)^{-1} \end{bmatrix} \\ &= \begin{bmatrix} S_{1}^{-1}& -A^{-1}BS_{2}^{-1} \\ -S_{2}^{-1}CA^{-1} & S_{2}^{-1} \end{bmatrix} \end{align} \tag{25} $$ 

- $$ S_{1} = A - BD^{-1}C \text{ : (Schur Complement).} \tag{26} $$

- $$ S_{2} = D - CA^{-1}B \text{ : (Schur Complement).} \tag{27} $$

<br>

## **A Characterization of Symmetric Positive Definite Matrices Using Schur Complements**

<br>

- 

<br>

## **Pseudo-Inverses**

<br>

<br>

## **A Characterization of Symmetric Positive Semidefinite Matrices Using Schur Complements**

<br>

<br>


<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>
