---
layout: post
title: 벡터의 내적 (inner product)와 벡터의 정사영 (projection)
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
- `scalar projection`은 한 벡터에서 다른 벡터로 `projection`을 하였을 때 `projection`된 벡터의 시작점에서 `projection`된 지점까지의 `거리(크기)`를 나타냅니다.
- 반면 `vector projection`은 `projection`된 벡터의 시작점에서 projection된 지점까지의 거리만큼의 크기를 가지는 `벡터`를 나타냅니다.
- 그러면 두 벡터 $$ r, s $$ 가 있고 벡터 $$ s $$ 를 벡터 $$ r $$ 에 `projection` 시킨다는 가정하에 `scalar projection`과 `vector projection`을 구하는 방법에 대하여 알아보겠습니다. 

<br>

- 먼저 `scalar projection` 방법에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/la/projection/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- $$ \cos{\theta} = \frac{\text{adj}}{\color{blue}{\text{hyp}}} = \frac{\text{adj}}{\color{blue}{\vert s \vert}} $$

- $$ \text{adj} = \vert s \vert \cos{\theta} $$

<br>

- 위 식을 두 벡터의 내적의 성질에 접목시켜 보겠습니다. 두 벡터의 내적은 다음을 따릅니다.

<br>

- $$ r \cdot s = \vert r \vert \vert s \vert \cos{\theta} $$

<br>

- 따라서 앞의 식을 접목시키면 다음과 같습니다.

<br>

- $$ \text{adj} = \vert s \vert \cos{\theta} = \frac{r \cdot s}{\vert r \vert} = \hat{r} \cdot s $$

<br>

- 지금 까지가 `scalar projection`에 관한 내용이었습니다. 즉, 위 그림과 같이 두 벡터 $$ r, s $$를 이용하여 파란색의 길이를 알 수 있습니다.
- 그럼 여기서 `vector projection`으로 개념을 확장시켜 보겠습니다. 아시다시피 벡터는 크기와 방향을 가집니다. 따라서 `scalar projection`에 
방향을 추가하면 됩니다.

<br>
<center><img src="../assets/img/math/la/projection/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- $$ \text{vector projection} = \text{scalar projection} \times \text{unit vector} = \frac{r \cdot s}{\vert r \vert} \cdot \frac{r}{\vert r \vert} = \frac{r \cdot s}{\vert r \vert \cdot \vert r \vert} \cdot r= \frac{r \cdot s}{r \cdot r} \cdot r $$ 

<br>

- `unit vector`를 포함한 형태로 나타내면 다음과 같습니다.

<br>

- $$ \text{vector projection} = \text{scalar projection} \times \text{unit vector} = \frac{r \cdot s}{\vert r \vert} \cdot \frac{r}{\vert r \vert} = (\hat{r} \cdot s) \dot \hat{r} $$

<br>

- 위 식은 `scalar projection`에서 구한 길이값에 방향인 유닛 벡터를 곱하여 `vector projection`을 하는 식입니다.
- 위 계산 과정을 보면 `scalar projection`은 projection 된 벡터의 유닛 벡터($$ \hat{r} $$ )와 `projection`한 벡터($$ s $$)의 내적이 됨을 알 수 있습니다.
- `vector projection`은 벡터이기 때문에 개념적으로 스칼라 값에 유닛 벡터를 곱하면 됩니다. 따라서 위 식과 같이 유도될 수 있습니다.

<br>

- ## **vector projection 바로 구하기**

<br>

- 이번에는 vector projection을 바로 구해보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/la/projection/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서는 $$ \vec{b} $$를 $$ \vec{a} $$에 projection 시킵니다. 이것은 $$ \vec{b} $$로부터 $$ \vec{a} $$에 수직인점 까지의 길이를 가지며 $$ \vec{a} $$와 같은 방향을 갖는 벡터를 찾는것을 의미합니다.
- 그리고 $$ \vec{a} $$에서 projection 한 점 까지의 벡터를 $$ \vec{x} $$로 나타내고 변수 $$ p $$를 도입하여 $$ \vec{x} = p \vec{a} $$로 정의하겠습니다.
- 먼저 projection한 벡터와 $$ \vec{a} $$의 내적은 0입니다. 왜냐하면 사이각이 직각이기 때문에 앞에서 다룬 내적의 성질에 의해 0이 되게 됩니다.

<br>

- $$ (\vec{b} - p\vec{a})^{T} \vec{a} = 0 $$

<br>

- 위 관계식을 이용하여 $$ p $$를 정의해 보겠습니다.

<br>

- $$ \vec{b}^{T} \vec{a} - p\vec{a}^{T}\vec{a} = 0 $$

- $$ p = \frac{\vec{b}^{T}\vec{a}}{\vec{a}^{T}\vec{a}} $$

- $$ \vec{x} = p \vec{a} = \frac{\vec{b}^{T}\vec{a}}{\vec{a}^{T}\vec{a}} \vec{a} $$

<br>

- 이번 방법에서도 앞에서 정리한 방법과 동일한 결과의 vector projection을 구할 수 있었습니다.
- 특히 **두 unit vector의 내적은 1**이기 때문에 $$ \vec{a} $$가 unit vector라면 다음과 같습니다.

<br>

- $$ \vec{x} = p\vec{a} = (\vec{b}^{T}\vec{a})\vec{a} $$

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

