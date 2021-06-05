---
layout: post
title: 블록 행렬 곱 (block matrix multiplication)
date: 2020-08-26 00:00:00
img: math/la/projection/0.png
categories: [math-la] 
tags: [Linear algebra, block matrix multiplication] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 참조 : https://ximera.osu.edu/la/LinearAlgebra/MAT-M-0023/main
- 참조 : https://math.stackexchange.com/questions/787909/block-matrix-multiplication

<br>

- 이번 글에서는 행렬 곱을 효율적으로 하기 위한 `블록 행렬 곱 연산 (block matrix multiplication)`에 대하여 간략하게 알아보도록 하겠습니다.
- `블록 행렬 곱 연산`은 간단하게 말하여 행렬의 모든 원소 값들을 한번에 연산하는 것이 아니라 **영역 별로 따로 연산**하는 것을 뜻합니다. 이와 같이 문제의 범위를 여러 단위로 분할하고 최종적으로 합치는 방법을 통해 연산량을 줄일 수 있어서 많이 사용됩니다.

<br>

- 예를 들어 다음과 같이 2 x 2 크기의 $$ A, B $$ 행렬이 있다고 가정해 보겠습니다. 이 때, 행렬의 곱은 다음과 같이 나타낼 수 있습니다.

<br>

- $$ AB = \left[ \begin{array}{cc} a_{11} & a_{12} \\ a_{21} & a_{22} \end{array} \right]\cdot \left[ \begin{array}{cc} b_{11} & b_{12} \\ b_{21} & b_{22} \end{array} \right] = \left[ \begin{array}{cc} a_{11}b_{11}+a_{12}b_{21} & a_{11}b_{12}+a_{12}b_{22} \\ a_{21}b_{11}+a_{22}b_{21} & a_{22}b_{12}+a_ {22}b_{22} \end{array} \right] $$

<br>

- 만약 위 행렬의 각 원소인 $$ a_{ij}, b_{ij} $$가 스칼라 값이 아니라 행렬이라면 어떻게 될까요?

<br>

- $$ AB = \left[ \begin{array}{c|c} A_{11} & A_{12} \\\hline A_{21} & A_{22} \end{array} \right]\cdot \left[ \begin{array}{c|c} B_{11} & B_{12} \\\hline B_{21} & B_{22} \end{array} \right] = \left[ \begin{array}{c|c} A_{11}B_{11}+A_{12}B_{21} & A_{11}B_{12}+A_{12}B_{22} \\\hline A_{21}B_{11}+A_{22}B_{21} & A_{22}B_{12}+A_{22}B_{22} \end{array} \right] $$

<br>




<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

