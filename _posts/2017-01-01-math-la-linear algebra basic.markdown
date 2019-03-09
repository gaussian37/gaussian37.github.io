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

## 자주 사용하는 선형대수학 공식

<br><br>

## 역행렬과 행렬식(determinant)

<br><br>

## 고유값(eigenvalue)과 고유벡터(eigenvector)

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
    
+ ·$$ Av = \lambda v $$ ... `1번식`

$$ \begin{pmatrix} a_{11} & \cdots & a_{1n} \\     \vdots & \ddots & \vdots \\ a_{n1} & \cdots & a_{nn} \\ \end{pmatrix} \begin{pmatrix} v_{1} \\ \vdots \\ v_{n}\end{pmatrix} = \lambda \begin{pmatrix} v_{1} \\ \vdots \\ v_{n}\end{pmatrix} $$ ... `2번식` 
    
+ 좀 더 정확한 용어로 $$ \lambda $$는 **행렬 A의 고유값**이고 $$ v $$는 행렬 A의 $$\lambda$$ 에 대한 **고유벡터**이다 라고 말할 수 있습니다.
+ 고유값과 고유벡터는 행렬에 따라 정의되는 값으로서 어떤 행렬은 고유값과 고유벡터가 존재하지 않을 수도 있습니다.
    + 어떤 행렬은 하나만 존재하거나, 최대 N x N 행렬에서 N개까지 존재할 수 있습니다.

<br>

### 2. 고유값과 고유벡터의 기하학적 의미

+ 고유벡터 : 선형변환 A에 의해 방향은 보존되고 스케일만 변화되는 방향벡터
+ 고유값 : 고유벡터가 변화되는 스케일
+ 예를 들어 지구가 자전운동하는 것을 3차원 회전변환으로 생각해 보았을 때, 
    + 고유벡터 : 회전축 벡터
    + 고유값 : 1
    
<br>

### 3. 고유값 분해를 이용한 대각화 : eigen-decomposition

+ 고유값, 고유벡터는 정방행렬의 대각화와 밀접한 관련이 있습니다. 
+ 대각행렬과의 행렬곱에 대해 살펴보면, 대각행렬을 뒤에 곱하면 행렬의 열벡터들이 **대각원소의 크기만큼 상수배**가 됩니다
+ ·$$ \begin{pmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \\ v_{31} & v_{32} & v_{33} \\ \end{pmatrix} \begin{pmatrix} \lambda_{1} & 0 & 0 \\ 0 & \lambda_{1} & 0 \\ 0 & 0 & \lambda_{1} \\ \end{pmatrix} = \begin{pmatrix} \lambda_{1}v_{11} & \lambda_{2}v_{12} & \lambda_{3}v_{13} \\ \lambda_{1}v_{21} & \lambda_{2}v_{22} & \lambda_{3}v_{23} \\ \lambda_{1}v_{31} & \lambda_{2}v_{32} & \lambda_{3}v_{33} \\ \end{pmatrix} $$ ... `3번식`
+ 행렬 A의 고유값, 고유벡터들을 $$ \lambda_{i}, v_{i}, i=1,2,3, ..., n $$ 이라고 하겠습니다.
    + ·$$ \begin{matrix} Av_{1} = \lambda_{1}v_{1} \\ Av_{2} = \lambda_{2}v_{2} \\ ... \\ Av_{n} = \lambda_{n}v_{n} \\ \end{matrix} $$ ... `4번식`
+ `4번식`을 한꺼번에 표현하여 정리해보겠습니다.
    + ·$$ A[v_{1} \ v_{2} \ ... v_{n}] = [\lambda_{1}v_{1} \ \lambda_{2}v_{2} \ ... \lambda_{n}v_{n}] = [v_{1} \ v_{2} \ ... v_{n}] \begin{bmatrix} \lambda_{1} & 0 & \cdots & 0 \\ 0 & \lambda_{2} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \lambda{n} \\ \end{bmatrix} $$ ... `5번식`
+ 행렬 A의 고유벡터들을 열벡터로 하는 행렬을 $$ P $$, 고유값들을 대각원소로 하는 대각 행렬을 $$ \Lambda $$라고 하면 다음식이 성립합니다.
    + ·$$ AP = P\Lambda $$ ... `6번식`
    + ·$$ A = P\Lambda P^{-1} $$ ... `7번식`
+ 이와 같이 행렬 A는 자신의 `고유벡터들을 열벡터로 하는 행렬`과($$ P $$) `고유값을 대각원소로 하는 행렬`($$ \Lambda $$)의 곱으로 대각화 분해가 가능합니다.
    + 이러한 대각화 분해를 `eigen-decomposition` 이라고 합니다.
    + ·$$ \begin{bmatrix} 1 & 1 & 0 \\ 0 & 2 & 1 \\ 0 & 0 & 3 \\ \end{bmatrix} \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 2 \\ \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \\ \end{bmatrix} \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 2 \\ \end{bmatrix}^{-1} $$ ... `8번식`

<br>

+ 모든 정방행렬이 이런 방식의 eigen decomposition이 가능한 것은 아니지만 대각화 가능한 경우 아래 항목을 쉽게 계산할 수 있습니다.
    + det(A)
    + A의 거듭제곱
    + 역행렬
    + 대각합(trace)
    + 행렬의 다항식 
  



    
    
<br><br>

## 특이값분해(SVD)

<br><br>

## 선형연립방정식의 풀이

## 주성분분석(PCA)

<br><br>
