---
layout: post
title: 벡터의 정사영 (projection)
date: 2020-08-26 00:00:00
img: math/la/projection/0.png
categories: [math-la] 
tags: [Linear algebra, vector, projection, 선형 대수학, 벡터, 정사영] # add tag
---

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 이번 글에서는 벡터의 정사영을 구하는 2가지 관점에 대하여 간략하게 정리하겠습니다.

<br>

- ## **scalar projection → vector projection**

<br>

- 먼저 `scalar projection`에서 `vector projection`으로 확장하는 관점에서 `vector projection`에 대하여 알아보겠습니다.
- `scalar projection`은 한 벡터에서 다른 벡터로 projection을 하였을 때 projection된 벡터의 시작점에서 projection된 지점까지의 `거리(크기)`를 나타냅니다.
- 반면 `vector projection`은 projection된 벡터의 시작점에서 projection된 지점까지의 거리만큼의 크기를 가지는 `벡터`를 나타냅니다.
- 그러면 두 벡터 $$ r, s $$가 있고 벡터 $$ s $$를 벡터 $$ r $$에 projection 시킨다는 가정하에 `scalar projection`과 `vector projection`을 구하는 방법에 대하여 알아보겠습니다. 

<br>

- 먼저 `scalar projection` 방법에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/la/projection/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- $$ \cos{\theta} = \frac{\text{adj}}{\color{blue}{\text{hyp}}} = \frac{\text{adj}}{\color{blue}{\vert s \vert}} $$

- $$ r \cdot s = \vert r \vert \vert s \vert \cos{\theta} $$

<br>
<center><img src="../assets/img/math/la/projection/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 계산 과정을 보면 `scalar projection`은 projection 된 벡터의 유닛 벡터($$ \hat{r} $$ )와 projection한 벡터($$ s $$)의 내적이 됨을 알 수 있습니다.
- `vector projection`은 벡터이기 때문에 개념적으로 스칼라 값에 유닛 벡터를 곱하면 됩니다. 따라서 위 식과 같이 유도될 수 있습니다.

<br>
<center><img src="../assets/img/math/la/projection/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>


<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

