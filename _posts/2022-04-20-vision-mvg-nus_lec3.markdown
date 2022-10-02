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

## **Circular Points and Their Dual**

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>


- 어떤 두 개의 점이 $$ l_{\infty} $$ (`line at infinity`) 상에서 `similarity transformation`에 대하여 `fixed point` 라고 생각해 보겠습니다.
- 먼저 `fixed point`라고 하면 **정의역과 공역이 공간 $$ X $$ 인 함수 $$ f : X \to X $$ 에 대하여 $$ x_{0} \in X $$ 가 $$ f(x_{0}) = x_{0} $$ 을 만족할 때, 이 점 $$ x_{0} $$ 를 `fixed points`**라고 합니다.
- 이 때, $$ l_{\infty} $$ 상에 존재하는 `fixed points`인 두 개의 점을 `circula (absolute) points` 라고 하며 아래와 같이 복소수 형태로 나타냅니다. (이름의 의미는 이후 슬라이드에서 설명합니다.)

<br>

- $$ I = \begin{pmatrix} 1 \\ i \\ 0 \end{pmatrix} $$

- $$ J = \begin{pmatrix} 1 \\ -i \\ 0 \end{pmatrix} $$

<br>

- 두 점 $$ I, J $$ 는 $$ l_{\infty} $$ 상에 있으므로 마지막 차원은 0이 되고 보시는 바와 같이 `complex conjugate`를 만족하는 `ideal points (point at infinity)`입니다. ($$ i $$ 는 허수를 의미합니다.)

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 슬라이드에서는 `circular points`인 $$ I, J $$ 가 `fixed point` 일 때, `projective transformation` $$ H $$ 는 `similarity`임이 `필요 충분 조건`임을 설명합니다.
- 따라서 위 슬라이드 조건과 앞에서 설명한 `fixed point`의 정의와 같이 $$ I' = H_{x}I = I $$ 임을 통하여 정의역 $$ I $$ 가 $$ H_{x} $$ 를 거치더라도 $$ I $$ 됨을 통하여 `fixed point` 임을 보입니다. (이해 필요...)

<br>

- 위 식에서 $$ H_{s} $$ 는 `similarity transformation` 행렬로 아래 식을 따릅니다.

<br>

- $$ H_{s} = \begin{bmatrix} s\cos{(\theta)} & -s\sin{(\theta)} & t_{x} \\ s\sin{(\theta)} & s\cos{(\theta)} & t_{y} \\ 0 & 0 & 1 \end{bmatrix} $$

<br>

- 따라서 위 식과 슬라이드의 오일러 공식을 이용하면 다음과 같이 정리할 수 있습니다.

<br>

- $$ I' = H_{s}I = \begin{bmatrix} s\cos{(\theta)} & -s\sin{(\theta)} & t_{x} \\ s\sin{(\theta)} & s\cos{(\theta)} & t_{y} \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ i \\ 0 \end{bmatrix} $$

- $$ = \begin{bmatrix} s \cdot \cos{(\theta)} - s \cdot i \cdot \sin{(\theta)} \\ s \cdot \sin{(\theta)} + s \cdot i \cdot \cos{(\theta)} \end{bmatrix} $$

- $$ = s \begin{bmatrix} \cos{(\theta)} - i \cdot \sin{(\theta)} \\ \sin{(\theta)} + i \cdot \cos{(\theta)} \\ 0 \end{bmatrix} $$

- $$ = s \begin{bmatrix} e^{-i\theta} \\ i(\cos{(\theta)} - i\sin{(\theta)}) \\ 0 \end{bmatrix} $$

- $$ = s \begin{bmatrix} e^{-i\theta} \\ i(e^{-i\theta}) \\ 0 \end{bmatrix} $$

- $$ = s e^{-i\theta} \begin{bmatrix} 1 \\ i \\ 0 \end{bmatrix} $$

<br>

- **그런데 이게 왜 $$ I $$ 가 되는 지 확인이 필요하다...**

<br>

- 위 식과 같은 전개 과정을 $$ J $$ 에 대하여 적용하면 유사하게 유도할 수 있습니다.
- 따라서 `circular points`가 `fixed points`이면 이 때 사용된 `transformation matrix`는 `similarity transformation`임을 확인할 수 있었습니다.


<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/16.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번 슬라이드에서는 `circular points`의 의미에 대하여 설명합니다.

<br>

- 이전 강의에서 배운 `conics` 관련 식은 아래와 같습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/16_1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/16_2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 먼저 문제를 간단하게 살펴보기 위해 원 (circle)인 형태를 살펴보면 `conics` 식에서 `a = c = 1`로 가정하고 `b = 0`으로 두겠습니다. 그러면 위 슬라이드와 같이 식이 정리됩니다.ㄴ

<br>

- $$ x_{1}^{2} + x_{2}^{2} + dx_{1}x_{3} + ex_{1}x_{3} + fx_{3}^{2} = 0 $$

<br>

- 이 때, $$ x_{3} = 0 $$ 은 0 인 상태에서 $$ l_{\infty} $$ 와 `conic`이 교차하는 지점의 `ideal points`는 슬라이드와 같이 $$ I, J $$ 에서 만나게 됨을 알 수 있으며 이 때 `ideal points`의 좌표가 앞에서 다룬 $$ I, J $$ 가 됩니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec3/17.png" alt="Drawing" style="width: 800px;"/></center>
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