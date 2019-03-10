---
layout: post
title: 선형 대수학 관련 기본 내용 
date: 2016-12-01 00:00:00
img: math/la/linear-algebra-image.jpg
categories: [math-la] 
tags: [Linear algebra, 선형대수학] # add tag
---

<iframe src="//partners.coupang.com/cdn/redirect?url=customjs%2Faffiliate%2Fsearch-bar%2F0.0.3%2Flogo-01.html%3FtrackingCode%3DAF1042200" width="100%" height="85" frameborder="0" scrolling="no"></iframe>

출처 : https://darkpgmr.tistory.com/

이번 글에서는 자주 사용하는 선형대수학 개념들에 대하여 정리해 보도록 하겠습니다.
이 글 전체 내용은 [다크프로그래머님 블로그](https://darkpgmr.tistory.com/)에서 필요한 내용한 가져와서 정리하였습니다.

<br><br>

## ※ 자주 사용하는 선형대수학 공식

<br><br>

## ※ 역행렬과 행렬식(determinant)

<br><br>

## ※ 고유값(eigenvalue)과 고유벡터(eigenvector)

+ 선형대수학에서 가장 중요한 개념 중의 하나인 `고유값`과 `고유벡터`에 대해서 다루어 보겠습니다.
+ 고유값과 고유 벡터는 다음 개념 등에 사용되고 있습니다. 
    + SVD(특이값 분해)
    + Pseudo-Inverse
    + 선형연립방정식의 풀이
    + PCA(주성분 분석)
    + ...
    
### 1. 고유값과 고유벡터란?

+ 고유값(eigenvalue)와 고유벡터(eigenvector)에 대한 수학정 정의는 간단합니다.
    + **행렬 A를 선형변환**으로 봤을 때, 선형변환 A에 의한 변환 결과가 자기 자신의 상수배가 되는 0이 아닌 벡터를 `고유벡터` 
    + 이 때, 상수배 값을 `고유값`이라고 합니다.
    + `고유벡터` : n x n 정방행렬(고유값, 고유벡터는 정방행렬에 대해서만 정의됨) A에 대하여 $$ Av = \lambda v $$를 만족하는 0이아닌 벡터
    + `고유값` : 상수 $$ \lambda $$
    
+  $$ Av = \lambda v $$ ... `1번식`

$$ \begin{pmatrix} a_{11} & \cdots & a_{1n} \\     \vdots & \ddots & \vdots \\ a_{n1} & \cdots & a_{nn} \\ \end{pmatrix} \begin{pmatrix} v_{1} \\ \vdots \\ v_{n}\end{pmatrix} = \lambda \begin{pmatrix} v_{1} \\ \vdots \\ v_{n}\end{pmatrix} $$ ... `2번식` 
    
+ 좀 더 정확한 용어로 $$ \lambda $$는 **행렬 A의 고유값**이고 $$ v $$는 행렬 A의 $$\lambda$$ 에 대한 **고유벡터**이다 라고 말할 수 있습니다.
+ 고유값과 고유벡터는 행렬에 따라 정의되는 값으로서 어떤 행렬은 고유값과 고유벡터가 존재하지 않을 수도 있습니다.
    + 어떤 행렬은 하나만 존재하거나, 최대 N x N 행렬에서 N개까지 존재할 수 있습니다.

<br>

### 2. 고유값과 고유벡터의 기하학적 의미

+ 고유벡터 : 선형변환 A에 의해 방향은 보존되고 스케일만 변환되는 방향벡터
+ 고유값 : 고유벡터가 변화되는 스케일
+ 예를 들어 지구가 자전운동하는 것을 3차원 회전변환으로 생각해 보았을 때, 
    + 고유벡터 : 회전축 벡터
    + 고유값 : 1
    
<br>

### 3. 고유값 분해를 이용한 대각화 : eigen-decomposition

+ 고유값, 고유벡터는 정방행렬의 대각화와 밀접한 관련이 있습니다. 
+ 대각행렬과의 행렬곱에 대해 살펴보면, 대각행렬을 뒤에 곱하면 행렬의 열벡터들이 **대각원소의 크기만큼 상수배**가 됩니다
+  $$ \begin{pmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \\ v_{31} & v_{32} & v_{33} \\ \end{pmatrix} \begin{pmatrix} \lambda_{1} & 0 & 0 \\ 0 & \lambda_{1} & 0 \\ 0 & 0 & \lambda_{1} \\ \end{pmatrix} = \begin{pmatrix} \lambda_{1}v_{11} & \lambda_{2}v_{12} & \lambda_{3}v_{13} \\ \lambda_{1}v_{21} & \lambda_{2}v_{22} & \lambda_{3}v_{23} \\ \lambda_{1}v_{31} & \lambda_{2}v_{32} & \lambda_{3}v_{33} \\ \end{pmatrix} $$ ... `3번식`
+ 행렬 A의 고유값, 고유벡터들을 $$ \lambda_{i}, v_{i}, i=1,2,3, ..., n $$ 이라고 하겠습니다.
    +  $$ \begin{matrix} Av_{1} = \lambda_{1}v_{1} \\ Av_{2} = \lambda_{2}v_{2} \\ ... \\ Av_{n} = \lambda_{n}v_{n} \\ \end{matrix} $$ ... `4번식`
+ `4번식`을 한꺼번에 표현하여 정리해보겠습니다.
    +  $$ A[v_{1} \ v_{2} \ ... v_{n}] = [\lambda_{1}v_{1} \ \lambda_{2}v_{2} \ ... \lambda_{n}v_{n}] = [v_{1} \ v_{2} \ ... v_{n}] \begin{bmatrix} \lambda_{1} & 0 & \cdots & 0 \\ 0 & \lambda_{2} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \lambda{n} \\ \end{bmatrix} $$ ... `5번식`
+ 행렬 A의 고유벡터들을 열벡터로 하는 행렬을 $$ P $$, 고유값들을 대각원소로 하는 대각 행렬을 $$ \Lambda $$라고 하면 다음식이 성립합니다.
    +  $$ AP = P\Lambda $$ ... `6번식`
    +  $$ A = P\Lambda P^{-1} $$ ... `7번식`
+ 이와 같이 행렬 A는 자신의 `고유벡터들을 열벡터로 하는 행렬`과($$ P $$) `고유값을 대각원소로 하는 행렬`($$ \Lambda $$)의 곱으로 대각화 분해가 가능합니다.
    + 이러한 대각화 분해를 `eigen-decomposition` 이라고 합니다.
    +  $$ \begin{bmatrix} 1 & 1 & 0 \\ 0 & 2 & 1 \\ 0 & 0 & 3 \\ \end{bmatrix} \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 2 \\ \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \\ \end{bmatrix} \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 2 \\ \end{bmatrix}^{-1} $$ ... `8번식`

<br>

+ 모든 정방행렬이 이런 방식의 eigen decomposition이 가능한 것은 아니지만 대각화 가능한 경우 아래 항목을 쉽게 계산할 수 있습니다.
    + det(A)
        + 　$$ det(A) = det(P\Lambda P^{-1}) $$
        + 　$$ = det(P) \ det(\Lambda) \ det(P)^{-1} $$
        + 　$$ = det(\Lambda) $$
        + 　$$ = \lambda_{1}\lambda_{2}\cdots\lambda_{n} $$ ... `9번식`
        + **행렬 A의 determinant는 고유값들의 곱**과 같습니다.
    + A의 거듭제곱
        + 　$$ A^{k} = (P\Lambda P^{-1})^{k}$$
        + 　$$ = (P\Lambda P^{-1})(P\Lambda P^{-1})\cdots (P\Lambda P^{-1}) $$
        + 　$$ = P\Lambda^{k}P^{-1} $$
        + 　$$ = P \ diag(\lambda_{1}^{k}, \cdots , \lambda_{n}^{k}) P^{-1} $$ ... `10번식`
        + **행렬 A의 거듭제곱은 $$ P $$와 $$ \Lambda $$의 곱**으로 나타내어 집니다.
    + 역행렬
        + 　$$ A^{-1} = (P\Lambda P^{-1})^{-1} $$
        + 　$$ = P\Lambda^{-1}P^{-1} $$
        + 　$$ = P \ diag(\frac{1}{\lambda_{1}}, \cdots , \frac{1}{\lambda_{n}}) P^{-1} $$ ... `11번식`
        + **행렬 A의 역행렬은 $$ P $$와 $$ \Lambda $$의 곱**으로 나타내어 집니다.         
    + 대각합(trace)
        + 　$$ tr(A) = tr(P\Lambda P^{-1})$$
        + 　$$ = tr(\Lambda) (\because tr(AB) = tr(BA) ) $$
        + 　$$ = \lambda_{1} + \cdots + \lambda_{n}$$ ... `12번식`
        + **행렬 A의 대각합은 고유값의 합**과 같습니다.
    + 행렬의 다항식
        + 　$$ f(A) = a_{0}E + a_{1}A + \cdots + a_{n}A^{n} (f(x) = a_{0} + a_{1}x + \cdots +a_{n}X^{n})$$
        + 　$$ = a_{0}PP^{-1} + a_{1}P\Lambda P^{-1} + \cdots + a_{n}P\Lambda^{n}P^{-1} $$
        + 　$$ = P(a_{0}P^{-1} + a_{1}\Lambda P^{-1} + \cdots + a_{n}\Lambda^{n}P^{-1})$$
        + 　$$ = P(a_{0}E + a_{1}\Lambda + \cdots + a_{n}\Lambda^{n})P^{-1} $$
        + 　$$ = P \ diag(f(\lambda_{1}), \cdots , f(\lambda_{n}))P^{-1} $$ ... `13번식` 
        + 행렬의 다항식은 고유값과 고유벡터를 이용하여 간략하게 표현할 수 있습니다.

<br>

### 4. 고유값 분해(eigen-decomposition) 가능조건 - 일차 독립

+ 모든 정방행렬이 고유값 분해가 가능한 것은 아닙니다.
+ n x n 정방행렬 $$ A $$가 고유값 문해가 가능하려면 행렬 A가 n개의 `일차독립`인 고유벡터를 가져야 합니다.
    + 일차독립(Linearly Independent)를 간단하게 설명하면
    + 어떤 벡터들의 집합 $$ \{v_{1}, \cdots , v_{n} \}$$ 이 있을 때, 이들 벡터들 중 어느 한 벡터도 다른 벡터들의 일차결합으로 표현될 수 없으면 이 벡터들은 서로 일차독립이라고 정의합니다.
    + 벡터들의 일차결합이란 $$ a_{1}v_{1} + a_{2}v_{2} + \cdots + a_{n}v_{n} $$과 같이 상수를 곱하여 합친 형태를 말합니다.
        + 만약 $$ v_{1} = (1, 0, 0), v_{2} = (0, 1, 0), v_{3} = (0, 0, 1) $$ 이라면 $$ v_{2}, v_{3} $$를 이용해서는 $$ v_{1} $$을 만들 수 없습니다.
        + 이와 같이 어떤 벡터도 다른 벡터들의 상수배 합으로 표현될 수 없으면 서로 `일차독립`이라고 합니다.
    + 또한 $$ \mathbb R^{n} $$ 공간에서는 최대 n개의 일차독립인 벡터들을 가질 수 있으며 n개의 일차독립인 벡터들은 이 공간을 생성하는 basis 역할을 합니다.
        + 　$$ v_{1} = (1, 0, 0), v_{2} = (0, 1, 0), v_{3} = (0, 0, 1), v_{4} = (-1, 3, 4) $$ 라면 $$ \{v_{1}, v_{2}, v_{3} \} $$은 일차독립 이지만 $$ \{v_{1}, v_{2}, v_{3}, v_{4} \} $$는 일차독립이 아닙니다.
            + 　$$ \because v_{4} = -v_{1}+3v_{2}+4v_{3} $$
        + 즉, 3차원 공간에서 가능한 일차독립인 벡터들의 갯수는 최대 3개입니다.
    + 또한 $$ v_{1}, v_{2}, v_{3} $$을 이용하여 3차원 공간의 모든 $$ (x, y, z) $$ 좌표를 생성할 수 있음을 알 수 있습니다.
        + 어떤 일차 독립인 벡터들의 집합의 일차결합을 통하여 어떤 공간의 모든 좌표를 생성할 수 있으면 이 벡터들을 이 공간의 `basis`라고 정의합니다.
+ 따라서 n차 정방행렬이 고유값 분해가 가능하려면 `n개의 일차독립인 고유벡터가 존재`해야 합니다.
    + 이 이유에 대해서는 고유값과 고유벡터의 계산과정을 통하여 알아보겠습니다.
   
<br>   
 
### 5. 고유값, 고유벡터의 계산

+ 고유값과 고유벡터를 정의하는 식인 $$ Av = \lambda v $$ 에서 $$ v $$ 에 대해 정리해 보면 다음과 같습니다.
    + 　$$ Av = \lambda v $$
    + 　$$ Av - \lambda v = 0 $$
    + 　$$ (A - \lambda E)v = 0 $$ ... `14번식`

+ 구하고자 하는 **고유값, 고유벡터**는 `14번식`을 풀어서 나오는 $$ \lambda $$ 와 $$ v $$ 입니다. ($$ v \neq 0 $$)
+ 그런데 `14번식`에서 $$ (A - \lambda E) $$의 역행렬이 존재하면 $$ v $$는 항상 영벡터가 됩니다.
+ 따라서 **고유값과 고유벡터가 존재** 하려면 $$ (A - \lambda E) $$ 의 역행렬이 존재하면 안됩니다.
    + 　$$ det(A - \lambda E) = 0 $$ ... `15번식` 
    + `15번식`을 행렬 A에 대하여 특성방정식(Characteristic equation)이라고 부르며 이 식을 $$ \lambda $$에 대하여 풀면 A의 고유값을 구할 수 있습니다.
    + **고유벡터**는 이렇게 구한 $$ \lambda $$를 다시 $$ (A - \lambda E)v = 0 $$ 에 대입하여 구할 수 있습니다.

<br>

+ 다음 행렬의 고유값과 고유벡터를 구해보겠습니다.
+ 　$$ \begin{bmatrix} 2 & 0 & -2 \\ 1 & 1 & -2 \\ 0 & 0 & 1 \\ \end{bmatrix} $$
+ 　$$ det(A - \lambda E) = det(\begin{bmatrix} 2 & 0 & -2 \\ 1 & 1 & -2 \\ 0 & 0 & 1 \\ \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} ) $$
+ 　$$ = \begin{vmatrix} 2-\lambda & 0 & -2 \\ 1 & 1-\lambda & -2 \\ 0 & 0 & 1-\lambda \\ \end{vmatrix} $$
+ 　$$ = (2-\lambda)( (1-\lambda)(1-\lambda) - 0 ) $$
+ 　$$ = (2 - \lambda)(1 - \lambda)^{2} $$ ... `17번식`
+ 따라서 특성방정식은 $$ (2-\lambda)(1-\lambda)^{2} = 0 $$
+ 특성방정식의 해는 $$ \lambda = 1, 2 $$이고 2인경우에는 단일근이지만 1인 경우에는 이중근입니다.
+ 고유벡터와 고유값은 쌍을 이루기 때문에 단일근에는 1개의 고유벡터가, 이중근에는 최대 2개, 삼중근에는 최대 3개의 고유벡터가 존재합니다.
+ 먼저 $$ \lambda = 2 $$를 $$ (A - \lambda E)v = 0 $$ 에 대입하여 고유벡터를 구해보겠습니다.
    + 　$$ \begin{bmatrix} 2-\lambda & 0 & -2 \\ 1 & 1-\lambda & -2 \\ 0 & 0 & 1-\lambda \\ \end{bmatrix} \begin{bmatrix} v_{1} \\ v_{2} \\ v_{3} \\ \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \end{bmatrix} $$
    + 　$$ \begin{bmatrix} 0 & 0 & -2 \\ 1 & -1 & -2 \\ 0 & 0 & -1 \\ \end{bmatrix} \begin{bmatrix} v_{1} \\ v_{2} \\ v_{3} \\ \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \end{bmatrix} $$
    + 　$$ v_{1} = v_{2}, v_{3} = 0 $$ ... `19번식`
    + 따라서 $$ \lambda = 2 $$ 일 때 대응하는 고유벡터는 $$ v = [1, 1, 0]^{T} $$ 로 잡을 수 있습니다.
+ 동일한 방법으로 $$ \lambda = 1 $$에 대하여 구해보면
    + 　$$ \begin{bmatrix} 1 & 0 & -2 \\ 1 & 0& -2 \\ 0 & 0 & 0 \\ \end{bmatrix} \begin{bmatrix} v_{1} \\ v_{2} \\ v_{3} \\ \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ \end{bmatrix} $$
    + 　$$ v_{1} = 2v_{3} $$ ... `20번식`
    + `20번식`을 만족하는 임의의 벡터는 $$ [2t, s, t] $$로 표현할 수 있습니다. (t, s는 임의의 실수)
    + 그리고 $$ [2t, s, t] = t*[2, 0, 1] + s*[0, 1, 0] $$ 으로 표현할 수 있으므로 $$ [2, 0, 1], [0, 1, 0] $$의 일차 결합으로 표현될 수 있습니다.
+ 계산 과정을 통해 확인을 해 보면 `고유값은 유일`하지만 `고유벡터는 유일하지 않습니다.`
    + 　$$ Av = \lambda v $$의 양변에 상수 $$ c $$를 곱하면
    + 　$$ A(cv) = \lambda(cv) $$
    + 만약 $$ v $$가 $$ \lambda $$에 대한 고유벡터이면 0이 아닌 임의의 상수 $$ c $$에 대하여 $$ cv $$ 또한 $$ \lambda $$에 대한 고유벡터임을 알 수 있습니다.
    + 또한 $$ v_{1}, v_{2} $$가 모두 고유값 $$ \lambda $$에 대응하는 고유벡터라고 하면 임의의 상수 $$ c_{1}, c_{2} $$에 대하여
        + 　$$ A(c_{1}v_{1} + c_{2}v_{2}) = c_{1}\lambda v_{1} + c_{2}\lambda v_{2} = \lambda(c_{1}v_{1} + c_{2}v_{2}) $$
        + 따라서 $$ c_{1}v_{1} + c_{2}v_{2} $$ 또한 $$ \lambda $$에 대한 고유벡터가 됨을 알 수 있습니다.
+ 따라서 고유벡터는 `19번식` 또는 `20번식`등과 같은 제약조건을 만족하는 벡터들 중에서 어느 벡터를 사용해도 무관하나 일반적으로 **벡터의 크기를 1로 정규화한 단위 벡터를 고유벡터로 잡는 것**이 일반적입니다.
+ `20번식`에서는 자유도가 2이기 떄문에 일차독립인 2개의 고유벡터를 잡아야만 가능한 모든 고유벡터들을 대표할 수 있습니다.
+ 위에서 구한 고유값과 고유벡터를 이용하여 행렬 대각화가 정말로 성립하는 지 확인해 보겠습니다.
    + 　$$ A = P \Lambda P^{-1} $$
    + 　$$ \begin{bmatrix} 2 & 0 & -2 \\ 1 & 1 & -2 \\ 0 & 0 & 1 \\ \end{bmatrix} = \begin{bmatrix} 1 & 0 & 2 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 2 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix} \begin{bmatrix} 1 & 0 & 2 \\ 1 & 1 & 0 \\ 0 & 0 & 1 \\ \end{bmatrix}^{-1} $$
    
### 6. 대칭행렬과 고유값 분해

    

<br><br>

## 특이값분해(SVD)

<br><br>

## 선형연립방정식의 풀이

## 주성분분석(PCA)

<br><br>
