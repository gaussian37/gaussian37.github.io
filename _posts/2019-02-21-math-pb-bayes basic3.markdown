---
layout: post
title: (베이즈 통계학 기초) 주관적인 숫자여도 추정이 가능하다.
date: 2019-02-21 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [통계학, 베이지안] # add tag
---

+ 출처 : [세상에서 가장 쉬운 베이즈 통계학 입문](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=103947200)

베이즈 추정은 이유가 불충분한 상황에서도 사용할 수 있습니다.

앞에 글에서 살펴본 바로는 베이즈 추정은 다음과 같은 절차를 통해 추정을 하게 됩니다.

+ 사전 확률 -> 조건부 확률 -> 관측에 의한 정보의 입수 -> 사후 확률

### 1. 주관적인 데이터 사용

+ 앞의 글에서 다루었던 [예제1](https://gaussian37.github.io/math-pb-bayes-basic1/)과 [예제2](https://gaussian37.github.io/math-pb-bayes-basic2/)에서는 `객관적`인 데이터를 사용하였었습니다.
+ 반면 이번 글에서 다루어 볼 내용은 **객관적인 사전 데이터가 없어도 추정이 가능**하다는 것을 베이즈 추정을 통해 알아보려고 합니다.
  + **사전확률을 주관적으로 설정**하여 추정을 실시할 수 있습니다.
  
```
당신이 남자라고 가정하자. 특정 여성 동료가 자신에게 호감을 가지고 있는지 알고 싶은 상황입니다. 그런 와중에 당신은 발렌타인데이에 그녀로부터 초콜릿을 받았습니다. 이 때 그녀가 당신을 진지하게 생각하고 있을 확률이 얼마라고 추정해야 할까요?
```

이 문제에서는 두가지 큰 어려움이 있습니다.
+ 첫째, 사람의 속마음을 수치화 해야한다는 것입니다.
+ 둘째, 이 문제에서 말하는 확률을 정확하게 정의하기 어렵습니다.
  + 주사위 처럼 시행을 해서 횟수를 셀 수도 없습니다.
+ 하지만, `베이즈 추정`은 이러한 문제에서도 적용할 수 있습니다.

<br>

### 2. 주관적으로 당신을 마음에 두고 있는가에 대한 사전확률을 설정합니다.

<img src="../assets/img/math/pb/bayes-basic3/3-1.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic3/3-2.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic3/3-3.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic3/3-4.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic3/3-5.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic3/3-6.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic3/3-7.PNG" alt="Drawing" style="width: 400px;"/>
