---
layout: post
title: orthogonal matrix  
date: 2018-09-27 15:00:00
img: math/la/overall.jpg
categories: [math-la] 
tags: [선형대수학, linear algebra, orthogonal matrix] # add tag
---

- 이번 글에서는 `orthogonal matrix`에 대하여 알아보도록 하겠습니다.
- 먼저 다음과 같이 행과 열의 위치를 바꾼 것을 전치 행렬(Transpose matrix)라고 하고 $$ T $$를 이용하여 표현합니다.
    - 　$$ A^{T}_{ij} = A_{ji} $$
- 만약 행렬 A의 열벡터 성분들이 다음 성질을 따른다고 가정해보겠습니다.
    - 　$$ a_{i} \cdot a_{j} = 0, \ i \ne j $$ : orthogonal한 상태
    - 　$$ a_{i} \cdot a_{j} = 1, \ i = j $$ : 벡터의 크기가 1인 상태
    - 참고로 위 두가지 성질을 모두 따르는 상태를 `orthonormal`하다고 합니다.
- 이 때, 행렬 $$ A $$는 $$ (a_{1}, a_{2}, \cdots, a_{n}) $$ 으로 표현될 수 있습니다.(이 때, $$ a_{i} $$는 위의 성질을 따르는 열벡터 입니다.)
- 그러면 $$ A^{T} $$는 $$ a_{i} $$를 행백터로 가지는 행렬로 표현될 수 있습니다.
- 그러면 $$ A^{T}*A = \begin{pmatrix} a_{1} \\ a_{2} \\ \cdots \\ a_{n} \end{pmatrix} * (a_{1} \ a_{2} \ \cdots \  a_{n} ) = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \cdots & \cdots & 1& \cdots  \\ 0 & 0  & \cdots & 1 \\ \end{pmatrix} = E $$ 가 됩니다.
    - 즉, $$ A^{T}A = E $$가 되어 $$ A^{T} = A^{-1} $$의 관계가 성립하게 됩니다.
- 위 식에서 $$ A $$와 $$ A^{T} $$의 곱이 identity matrix가 나올 수 있는 이유는 orthonormal하기 때문입니다.
    - 여기서 $$ A $$는 열벡터 기준으로 orthonormal 하고 $$ A^{T} $$는 행벡터 기준으로 orthonormal합니다.
    - orthonormal한 벡터들 끼리의 곱은 항상 0이고 크기는 1이기 때문에(즉, 같은 벡터의 곱은 크기이므로 1이 됩니다.) identity matrix를 결과로 얻습니다.
- 정리하면 각 열벡터의 성분이 orthonormal한 벡터들로만 이루어진 행렬을 orthogonal matrix라고 합니다.
- 이 행렬은 꽤나 유용하므로 그 사용 방법에 뒤에서는 다음 글에서 다루어 보겠습니다.  
      
 

