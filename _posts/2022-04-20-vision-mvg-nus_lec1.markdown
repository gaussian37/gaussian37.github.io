---
layout: post
title: Lecture 1. 2D and 1D projective geometry
date: 2022-04-20 00:00:01
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [Multiple View Geometry] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/LAHQ_qIzNGU?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/gQ7IUS8NKCI?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision
- 참조 : https://www.cuemath.com/learn/mathematics/conics-in-real-life/

<br>

- 이번 글에서는 **2D and 1D projective geometry** 내용의 강의를 듣고 정리해 보도록 하겠습니다.

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/T-p6d7av32Y" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

- 지금 부터는 **2D and 1D projective geometry** 강의의 후반부로 conics, dual conics와 관련된 내용과 transform 관련 내용에 대하여 다루어 보도록 하겠습니다.

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/gQ7IUS8NKCI" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>

<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec1/46.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- point $$ x_{i} $$ 가 line $$ l $$ 위에 있을 때, $$ l^{T} x_{i} = 0 $$ 으로 표현할 수 있습니다. 만약 transformed point (`projective transformation`) 인 $$ x_{i}' = H x_{i} $$ 가 $$ l' $$ 위에 있다면 $$ {l'}^{T} x_{i}' = 0 $$ 이 되고 $$ l $$ 과 $$ l' $$ 두 line의 관계로 나타내면 $$ l' = H^{-T} l $$ 으로 표현할 수 있습니다. 수식 전개 과정은 아래와 같습니다.

<br>

- $$ x_{i}' = H x_{i} $$

- $$ {l'}^{T} x_{i}' = 0 $$

- $$ \therefore \quad {l'}^{T} H x_{i} = 0 $$

- $$ {l'}^{T} H x_{i} = l^{t} x_{i} $$

- $$ {l'}^{T} H = l^{t} $$

- $$ H^{T} l' = l $$

- $$ \therefore \quad l' = H^{-T} l $$

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>