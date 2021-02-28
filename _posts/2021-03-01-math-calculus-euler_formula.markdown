---
layout: post
title: 오일러 공식 (Euler formula)
date: 2021-03-01 00:00:00
img: math/calculus/euler_formula/0.png
categories: [math-calculus] 
tags: [오일러 공식, euler formula] # add tag
---

<br>

- 이번 글에서는 오일러 공식을 유도하는 방법에 대하여 다루어 보도록 하겠습니다.
- 오일러 공식은 대표적으로 미분 방정식을 이용하여 증명하는 방법과 극한을 이용하여 증명하는 방법이 있습니다. 이 방법에 대하여 차례대로 설명해 보겠습니다.

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
<center><img src="../assets/img/math/calculus/euler_formula/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 위 식에서 유도한 바와 같이 $$ e^{i\theta} = \cos{\theta} + i\sin{\theta} $$ 관계를 가지게 됩니다.

<br>

## **극한을 이용한 오일러 공식 유도**

<br>