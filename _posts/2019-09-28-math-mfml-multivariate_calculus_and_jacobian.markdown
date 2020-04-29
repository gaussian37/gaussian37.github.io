---
layout: post
title: multivariate calculus와 jacobian
date: 2019-09-27 00:00:00
img: math/mfml/mfml.jpg
categories: [math-mfml] 
tags: [calculus, multivariate calculus, jacobian] # add tag
---

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>

- 이 글은 Coursera의 Mathematics for Machine Learning: Multivariate Calculus 을 보고 정리한 내용입니다.

<br>

### **목차**

<br>

- ### variables, constants & context

<br>

## **variables, constants & context**

<br>

- 앞의 글인 [basic alculus](https://gaussian37.github.io/math-mfml-basic_calculus/)에서는 어떤 점에서 함수의 미분을 하는 방법에 대하여 다루어 보았습니다.
- 특히 이전 글에서 다룬 케이스들은 variable이 1개인 single variable의 경우에 한정해서 보았었습니다.
- 이번 글에서 다루어 볼 것은 variable의 갯수가 여러개인 `multivariate system`에 대하여 다루어 보려고 합니다.
- multivariate에 대하여 다루어 보기전에 `variable`이 하는 역할에 대하여 먼저 확인을 해보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/mfml/multivariate_calculus_and_jacobian/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 예를 들어 위 그래프는 자동차의 특정 시간에 따른 속력 데이터입니다. 특정 시간에 자동차가 가질 수 있는 속력은 1개이므로 위 그래프는 명확합니다.
- 즉, 위 그래프에서 `time`이 `independent variable`이 됩니다. `speed`는 `time`에 따른 함수 값이 되는 것입니다.

<br>
<center><img src="../assets/img/math/mfml/multivariate_calculus_and_jacobian/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그러나 반대로 위와 같이 `speed`를 `independent variable`로 가져갈 수는 없습니다. 왜냐하면 함수의 정의 상 한 개의 입력 값에 대하여 한 개의 함수값이 대응 되어야 하기 때문에 함수의 정의에 맞지 않게 됩니다.
- 이러한 이유로 `speed`는 `dependent variable`이 되어야 합니다.
- 즉, **어떤 변수를 independent 또는 dependent variable로 정할 지**는 그 상황에 맞추어서 정해야 합니다.

<br>
<center><img src="../assets/img/math/mfml/multivariate_calculus_and_jacobian/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 예를 들어 위 그림과 같은 차가 있고 이 차의 엔진이 만들어내는 힘 $$ F $$는 위 식을 따른다고 가정해 보겠습니다.
- 위 식에서 $$ m $$ 은 mass, $$ a $$는 acceleration, $$ d $$는 drag coefficient, $$ v $$는 velocity라고 하겠습니다.
- 만약 내가 **운전자** 입장에서 variable들을 관찰해 본다면 $$ m $$과 $$ d $$는 고정된 상수 값입니다. 왜냐하면 운전중에 이 값은 변경할 수 없는 값이기 때문입니다.
- 운전자 입장에서 자연스러운 위 식의 해석을 하려면 $$ F $$가 `independent variable`이고 $$ a $$와 $$ v $$가 `dependent variable`이어야 합니다. 왜냐하면 엔진이 내는 힘에 따라서 속도와 가속도가 어떻게 변하는 지를 접근하는 것이 적합하기 때문입니다.
- 이렇게 해석하면 $$ m, d $$는 `constant`이고 $$ a, v $$는 `dependent variable`이고 $$ F $$는 `independent variable`이 됩니다.


<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>