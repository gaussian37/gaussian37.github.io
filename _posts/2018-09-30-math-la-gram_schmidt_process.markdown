---
layout: post
title: gram schmidt process
date: 2018-09-27 16:00:00
img: math/la/overall.jpg
categories: [math-la] 
tags: [선형대수학, linear algebra, orthogonal matrix, gram schmidt] # add tag
---

- 이번 글에서는 그램 슈미트 과정(gram schmidt process)에 대하여 알아보도록 하겠습니다.
- 그램 슈미트 과정은 `orthonormal basis vector set`을 구하는 과정입니다. 
- 즉, 주어진 벡터들을 이용해서 서로 수직인 벡터들을 만드는 방법이라고 생각 할 수 있고 주어진 벡터들에 대한 직교기저(orthogonal basis) 또는 정규직교기저(orthonormal basis)를 구하는 과정이라고 생각하면 됩니다.
- 먼저 그램 슈미트 과정을 시작하기에 앞서 linearly independent한 벡터들이 있다고 가정해 보겠습니다. 예를 들어  $$ v_{1}, v_{2}, v_{3} $$ 입니다.
    - 만약 행렬이 linearly independent 하다면 determinant가 0이 아닐 것입니다.
- 자, 그러면 그램 슈미트 직교화 가정에 대하여 알아보도록 하겠습니다.

<br>

- 먼저 벡터 $$ v_{1} $$에서 부터 시작해 보겠습니다. $$ v_{1} $$에 정규화를 해주면 $$ e_{1} = \frac{v_{1}}{\vert v_{1} \vert} $$를 구할 수 있습니다.



- 먼저 $$ v_{2} $$ 에서 $$ v_{1} $$ 으로 내린 projection을 이용하면 $$ v_{2} $$를 다른 식으로 표현할 수 있습니다.
- 즉, $$ v_{2} = (v_{2} \cdot e_{1})e_{1} + u_{2} $$ 가 됩니다. 여기서 $$ u_{2} $$는 projection입니다.
- 다시 이 값을 정리하면 $$ u_{2} = v_{2}  - (v_{2} \cdot e_{1})e_{1} $$  가 됩니다.
- 그리고 $$ u_{2} $$에 normalization을 해주게 되면 $$ e_{2}  = \frac{ u_{2} }{ \vert u_{2} \vert } $$로 정의 할 수 있습니다.
- 지금 까지 한 것을 보면 비교해야 할 대상은 $$ e_{1}, e_{2} $$ 입니다. 이 두 벡터는 서로 `orthonormal`한 관계를 가집니다. 
 