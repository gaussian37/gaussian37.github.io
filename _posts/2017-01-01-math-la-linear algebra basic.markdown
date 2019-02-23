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
    + 행렬 A를 선형변환으로 봤을 때, 선형변환 A에 의한 변환 결과가 자기 자신의 상수배가 되는 0이 아닌 벡터를 `고유벡터` 
    + 이 때, 상수배 값을 `고유값`이라고 합니다.
    + `고유벡터` : n x n 정방행렬(고유값, 고유벡터는 정방행렬에 대해서만 정의됨) A에 대하여 $$ Av = \lambda v $$를 만족하는 0이아닌 벡터
    + `고유값` : 상수 $$ \lambda $$
    
$$ Av = \lambda v $$ ... 1번식

$$ \begin{pmatrix} a_{11} & ... & a_{1n} \\     ... & ... & ... \\ a_{n1} & ... & a_{nn} \\ \end{pmatrix} \begin{pmatrix} v_{1} \\ ... \\ v_{n}\end{pmatrix} = \lambda \begin{pmatrix} v_{1} \\ ... \\ v_{n}\end{pmatrix} $$ ... 2번식 
    
+ 좀 더 정확한 용어로 $$ \lambda $$는 **행렬 A의 고유값**이고 v는 **행렬 A의 $$\lambda$$ 에 대한 고유벡터


    
    
<br><br>

## 특이값분해(SVD)

<br><br>

## 선형연립방정식의 풀이

## 주성분분석(PCA)

<br><br>
