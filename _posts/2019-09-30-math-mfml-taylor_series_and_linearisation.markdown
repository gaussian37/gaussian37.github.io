---
layout: post
title: multivariate chain rule과 applications
date: 2019-09-30 00:00:00
img: math/mfml/mfml.jpg
categories: [math-mfml] 
tags: [calculus, multivariate chain rule, application] # add tag
---

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>

- 이번 글에서 다룰 `테일러 급수`는 함수를 `다항식 급수`로 다시 표현하는 방법입니다. 
- 테일러 급수는 간단한 선형 근사법을 복잡한 함수에 사용합니다. 이 글에서는 먼저 단일변수 테일러 급수의 공식 표현을 도출하고 기계 학습과 관련된 이 결과의 몇 가지 중요한 결과에 대해 논의해 보겠습니다. 더 나아가 다변수 사례로 확장한 후 `Jacobian`과 `Hessian`이 어떻게 적용되는 지 살펴보겠습니다. 마지막에 다루는 다변수 테일러 급수에서는 앞에서 다룬 모든 내용을 총 집합 해서 설명해 보도록 하겠습니다.

<br>

## **목차**
- ### Taylor series for approximations
- ### Multivariable Taylor Series    

<br>

## **Taylor series for approximations**

<br>

- 이번 글에서는 복잡한 함수를 좀 더 간단한 함수를 이용하여 어떻게 근사화 할 수 있는 지 배워보려고 합니다.
- 근사화 하는 방법으로는 테일러 급수(`taylor series`)를 사용하려고 합니다.

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>