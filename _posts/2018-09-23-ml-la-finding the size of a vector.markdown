---
layout: post
title: Finding the size of a vector, its angle and projection  
date: 2018-09-15 03:49:00
img: math/la/overall.jpg
categories: [math-la] 
tags: [Linear algebra] # add tag
---

<br>

## **목차**

<br>

- ### Inner Product
- ### Cosine & dot product
- ### Projection

<br>

## **Inner product**

<br>

- 먼저 $$ i $$ 와 $$ j $$를 각 축의 기본이 되는 unit vector 라고 하겠습니다.
- 예를 들어 2차원에서 unit vector는 각각 $$ i = [1, 0] $$ 과 $$ j = [0, 1] $$의 값을 가집니다.

<br>
<center><img src="../assets/img/math/la/Finding the size of a vector,angle,projection/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>   

- unit vector는 각 축과 일치하는 방향으로 기본값인 1을 가지므로 어떤 vector는 unit vector를 이용하여 나타낼 수 있습니다.
- 위 그림에서 $$ r $$ vector 또한 unit vector인 $$ i $$와 $$ j $$를 이용하여 나타낼 수 있습니다. 즉 $$ r $$은 $$ i $$ 벡터의 방향과 $$ j $$ 벡터의 방향을 가지고 있다는 뜻입니다.
- 그러면 $$ r $$은 $$ ai + bj $$로 표현할 수 있습니다. vector 형태로 표현하면 $$ [a, b] $$가 됩니다.
- 여기서 $$ \vert a \vert $$는 무엇을 나타낼까요? 이것은 벡터의 크기를 나타냅니다. 따라서 $$ ai $$는 unit vector를 a배 한 것이라고 말할 수 있습니다.
- 그러면 vector $$ r $$의 크기는 어떻게 될까요? 이것은 피타고라스의 정리를 통하여 $$ \sqrt{a + b} $$임을 알 수 있습니다.
    - 물론 바로 뒤에서 쉽게 구하는 방법을 확인할 예정입니다.   

<br>

- 그러면 unit vector를 이용하여 vector의 여러가지 기본 연산에 대하여 다루어보도록 하겠습니다.
- 아래 이미지에서 다루는 것은 vector의 기본 연산 성질인 `Commutative`, `Distributive`, `Associative` 그리고 vector의 사이즈를 구하는 방법 입니다.

<br>
<center><img src="../assets/img/math/la/Finding the size of a vector,angle,projection/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>   


## Cosine & dot product

![cosine](../assets/img/math/la/Finding the size of a vector,angle,projection/cosine.png)

## Projection

![projection](../assets/img/math/la/Finding the size of a vector,angle,projection/projection.png)


 
 