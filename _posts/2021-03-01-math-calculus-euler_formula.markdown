---
layout: post
title: 오일러 공식 (Euler formula)
date: 2021-03-01 00:00:00
img: math/calculus/euler_formula/0.png
categories: [math-calculus] 
tags: [오일러 공식, euler formula, 미분 방정식] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 이번 글에서 다룰 오일러 공식은 다음 식과 같습니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 오일러 공식을 유도하는 다양한 방법 중 대표적으로 `미분 방정식`을 이용하여 유도하는 방법과 `테일러 급수`를 이용하여 유도하는 방법에 대하여 이 글에서 다루어 보도록 하겠습니다.

<br>

## **미분 방정식을 이용한 오일러 공식 유도**

<br>

- 오일러 공식을 사용할 때에는 수의 범위를 복소수의 범위에서 생각하겠습니다. 즉, 허수도 적용 가능한 공식임을 뜻합니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프는 복소 평면을 나타내며 복소 평면에서 복소수는 $$ x + iy $$ 형태의 한 점으로 나타낼 수 있습니다. 가로축은 실수(Real)을 뜻하고 세로축은 허수 (Imaginary)를 뜻합니다.
- 이 때, $$ x + iy $$를 $$ (x, y) $$와 같이 직교 좌표계 형태로 나타낼 수 있는 반면 `극 좌표계` 개념을 도입하여 $$ x + iy = r\cos{\theta} + i r \sin{\theta} $$ 형태로 나타낼 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 극 좌표계를 이용하여 위 그림과 같이 $$ r = 1 $$인 단위 원을 그리면 위 그래프와 같이 나타낼 수 있습니다. 이 경우 $$ z = \cos{\theta} + i \sin{\theta} $$로 극 좌표 $$ z $$를 나타낼 수 있습니다. 이 식과 미분 방정식을 이용하여 오일러 공식을 유도해 보겠습니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 위 식에서 유도한 바와 같이 $$ e^{i\theta} = \cos{\theta} + i\sin{\theta} $$ 관계를 가지게 됩니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 특히, $$ \theta = \pi $$를 대입하면 $$ e^{\pi i} + 1 = 0 $$으로 정리할 수 있습니다. 이와 같이 대표적으로 나타내는 오일러 공식의 포맷을 정리할 수 있습니다.

<br>

- $$ e^{i \theta} = \cos{\theta} + i\sin{\theta} $$

- $$ e^{i \pi} +1 = 0 $$

<br>

## **테일러 급수를 이용한 오일러 공식 유도**

<br>

- 먼저 테일러 급수에 대한 개념은 다음 링크를 참조하시기 바랍니다. 테일러 급수의 내용을 이해하고 있다는 전제 하에 진행하겠습니다.
    - 링크 : [https://gaussian37.github.io/math-mfml-taylor_series_and_linearisation/](https://gaussian37.github.io/math-mfml-taylor_series_and_linearisation/)

<br>
<center><img src="../assets/img/math/calculus/euler_formula/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 먼저 $$ e^{x}, \sin{x}, \cos{x} $$를 각각 테일러 급수 형태로 나타내면 위 식과 같습니다. 이 식을 이용하여 오일러 공식을 유도해 보겠습니다.
- 먼저 $$ e^{x} $$의 $$ x $$에 $$ ix $$를 대입해 보겠습니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 식의 전개를 살펴보면 자연 상수 $$ e $$의 테일러 급수 표현과 $$ \sin{x}, \cos{x} $$ 각각의 테일러 급수 표현을 이용하여 오일러 공식을 유도하였습니다.

<br>

## **오일러 공식의 기하학적 의미**

<br>

- 앞에서 다룬 내용과 다소 중복되지만 단순히 기하학적으로만 접근하여 오일러 공식을 유도하면 다음과 같습니다.

<br>
<center><img src="../assets/img/math/calculus/euler_formula/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이와 같은 다양한 방법(`미분 방정식`, `테일러 급수`, `기하학적 접근`)으로 오일러 공식을 유도할 수 있습니다.

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>