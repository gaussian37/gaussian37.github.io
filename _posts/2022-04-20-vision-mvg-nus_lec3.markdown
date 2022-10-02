---
layout: post
title: Lecture 3. Circular points and Absolute conic
date: 2022-04-20 00:00:03
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [Multiple View Geometry, Circular points and Absolute conic] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/T-p6d7av32Y?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/tsO6VO1s_x8?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision

<br>

- 이번 글에서는 **Circular points and Absolute conic** 내용의 강의를 듣고 정리해 보도록 하겠습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/T-p6d7av32Y" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번 강의에서는 크게 위 3가지 내용을 배울 예정입니다.
- ① `line at infinity` 와 `circular points` 개념을 배우고 이 개념을 이용하여 `affine` 또는 `projective` distortion을 제거하는 방법에 대하여 배워보도록 하겠습니다.
- ② 개념을 확장하여 `plane at infinity`를 배우고 `affine transformation`에서 불변한 성질에 대하여 배워보도록 하곘습니다.
- ③ `absolute conic`과 `absolute dual quadrics`를 배우고 `similarity transformation`에서 불변한 성질에 대하여 배워보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

## **Recovery of Affine Properties from Images**

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 슬라이드에서 $$ v_{1}, v_{2} $$ 2개의 점의 `cross product`를 이용하여 $$ l $$ 을 구하는 방법은 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/9_1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같은 선 $$ l $$, $$ v_{1}, v_{2} $$ 점의 관계에서 두 점이 선 $$ l $$ 위에 있으므로 아래 식을 만족합니다.

<br>

- $$ l \cdot v_{1} = 0 $$

- $$ l \cdot v_{2} = 0 $$

<br>

- 벡터 $$ (a, b, c) $$ 는 $$ (x_{1}, y_{1}, z_{1}) $$ 과 $$ (x_{2}, y_{2}, z_{2}) $$ 에 모두 수직인 벡터입니다. 따라서 $$ v_{1} $$ 과 $$ v_{2} $$ 모두에 수직인 벡터를 구하는 방법이 `cross product`이므로 다음과 같이 $$ l $$ 을 구할 수 있습니다.

<br>

- $$ l = v_{1} \times v_{2}  $$

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

## **Computing a Vanishing Point from a Length Ratio**

<br>

- 이번에는 `Vanishing Point`를 어떻게 계산하는 지 살펴보도록 하겠습니다. 

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>


<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

 
<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/16.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/17.png" alt="Drawing" style="width: 800px;"/></center>
<br>



<br>

## **Circular Points and Their Dual**

<br>

<br>

- 지금 부터는 **Circular points and Absolute conic** 강의의 후반부 내용을 살펴보도록 하겠습니다.

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/tsO6VO1s_x8" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

- 



<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>