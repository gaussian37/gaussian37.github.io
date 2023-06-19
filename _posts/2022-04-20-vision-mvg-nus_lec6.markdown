---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 6. Single view metrology
date: 2022-04-20 00:00:05
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [멀티플 뷰 지오메트리, Multiple View Geometry, Single view metrology] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/D6Pm5id_LK4?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/XSflleJr4qM?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/9TuPY67JBpo?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/D6Pm5id_LK4" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/5.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/6.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/7.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/8.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/9.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/10.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/11.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/13.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/14.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/15.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/16.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/17.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/18.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/19.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/20.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/21.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 이번 챕터의 이름은 `The Importance of the Camera Centre`로 카메라 센터점에 관하여 다룹니다.
- 앞에서 다룬 `cone`은 이상적인 기하 형태의 `cone`을 나타내므로 하나의 꼭지점과 매끈한 면을 가진을 `cone`을 생각하게 할 수 있습니다.
- 반면 위 슬라이드에서 보여주는 형태의 `cone`은 실제 이미지가 형성되는 형태와 연관되어 있습니다.
- 카메라 센터점 $$ C $$ 를 꼭지점으로 시작하여 `ray`를 쏘았을 때, 형성되는 `cone`과 같은 형태를 위 슬라이드에서는 `cone of rays`라고 표현하며 위 슬라이드의 그림과 같습니다. `cone of rays`는 실제 `object`가 존재하는 위치 까지 형성됩니다.
- 이 때, 이미지는 `cone of rays`에서 어떤 위치에 어떤 방향으로 교차하도록 평면을 위치시키면 `ray`들이 평면에 투영되는 데, 그 결과를 이미지 라고 볼 수 있습니다.
- 위 슬라이드에서 표시된 것 처럼 같은 `cone of rays`에 표시되어 있으나 `focal length`의 차이를 통하여 물체를 점점 더 확대하여 표현할 수 있음을 보여줍니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/22.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 같은 카메라 센터점을 통하여 얻어진 이미지들은 어떤 평면에서 다른 평면으로의 `projective transformation (Homography)` 관계를 가집니다. 예를 들어 이전 슬라이드에서 집 형태의 물체를 다양한 위치 및 방향으로 평면을 집어 넣어서 조금씩 다른 형상의 집 형태의 상이 맺히도록 이미지를 형성하였습니다.
- 이 때, 서로 다른 형태의 집은 `projective transformation` 관계를 가지게 되고 그 관계만 알 수 있다면 같은 형태로 만들 수 있습니다. `projective transformation` 관계이기 때문에 `line`은 그대로 `line`으로 유지됩니다.
- 서로 다른 평면의 이미지 간에는 `projective transformation` 관계를 가지므로 `projective properties`는 유지 됩니다. 이와 관련된 속성으로는 `incidence`, `cross ratio`, `conic section`이 있습니다. (이전 글에서 다룬 내용입니다.)
    - `incidence`는 2개의 점이 하나의 선에 있었다면 `projective transformation`을 하더라도 2개의 점은 같은 하나의 선에 있다는 것입니다. 즉, 선은 그대로 선으로 유지됨을 의미합니다.
    - `cross ratio` :  4개의 동일 선상에 존재하는 점들 간의 거리 비율은 `projective transformation`을 하더라도 유지됨을 의미합니다.
    - `conic section` : 어떤 이미지 상에서의 `conic section`은 `projective transformation`을 하더라도 `conic section`으로 유지된다는 점입니다. 물론 어떤 변환 관계인 지에 따라서 다른 형태의 `conic section`이 됩니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/23.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 같은 카메라 센터점에서 촬영한 2개의 이미지 $$ I $$ 와 $$ I' $$ 에 대하여 `projective transformation` 즉, `homography` 관계를 가짐을 앞에서 살펴보았습니다. 이번 슬라이드에서는 이 부분에 대하여 좀 더 수식적으로 살펴봅니다.
- 위 슬라이드에서 $$ K $$ 는 `intrinsic parameter` ( $$ 3 \times 3 $$ 크기의 행렬)을 의미하고 $$ R $$ 은 `extrinsic parameter` ( $$ 3 \times 3 $$ 크기의 행렬)을 의미합니다. $$ -\tilde{C} $$ 는 원점에서 카메라 중점 까지의 거리를 나타냅니다. (`translation`의미로 이 부분도 이전 강의에서 다루었습니다.)
- 같은 카메라 중점을 고려하기 때문에 $$ P $$ 와 $$ P' $$ 모두 $$ -\tilde{C} $$ 을 사용하였습니다. 나머지 $$ K, R $$ 은 서로 다른 카메라를 고려하였기 때문에 $$ K, R $$ 과 $$ K', R' $$ 로 구분하였습니다.

<br>

- 이미지에 표현된 3D 공간 상에서의 점 $$ X $$ 는 위 슬라이드에 있는 식을 통하여 2개의 카메라에서 다음과 같은 관계로 표현됩니다.

<br>

- $$ x' = P'X = (K'R')(KR)^{-1}PX = (k'R')(KR)^{-1}x $$

<br>

- 따라서 앞에서 언급한 `homography` $$ H $$ 는 $$ H = (K'R')(KR)^{-1} $$ 로 정의할 수 있습니다.
- 3D 공간 상의 점들을 변환하는 관계이지만 2D 이미지 상에서 변환이 발생하므로 `planar homography`라는 용어로 표현합니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/24.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 이번 슬라이드에서는 `focal length`의 증가에 따라서 이미지 평면에서의 변화와 관련된 내용을 다룹니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/24.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/25.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


- 슬라이드의 $$ K'K^{-1} $$ 의 행렬에서 $$ (1 - k)\tilde{x}_{0} $$ 가 사용된 이유를 살펴보면 다음과 같습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/25_1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 $$ \tilde{x}_{0} $$ 는 `principal point`를 뜻하고 `magnification factor`인 $$ k = f'/f $$ 가 됩니다.
- 위 그림에서 $$ \tilde{x}' $$ 는 다음과 같이 정의할 수 있습니다.

<br>

- $$ \begin{align} \tilde{x}' &= \frac{f'}{f}(\tilde{x} - \tilde{x}_{0}) + \tilde{x}_{0} \\ &= \frac{f'}{f}\tilde{x} + (1 - \frac{f'}{f})\tilde{x}_{0} \\ &= k\tilde{x} + (1 - k)\tilde{x}_{0} \end{align} $$

<br>

- 따라서 $$ \tilde{x}' = k\tilde{x} + (1-k)\tilde{x}_{0} $$ 로 표현할 수 있음을 확인하였습니다. 즉, $$ I $$ 이미지에서의 점 $$ \tilde{x} $$ 를 $$ I' $$ 이미지의 점 $$ \tilde{x}' $$ 로 변환하기 위해서는 $$ k $$ 배를 한 후 $$ (1-k)\tilde{x}_{0} $$ 만큼 이동해 주면 됩니다.
- 이 내용을 이미지의 $$ u, v $$ 방향에 대하여 모두 적용해 주면 `focal length`의 차이를 반영하는 행렬 $$ K'K^{-1} $$ 은 다음과 같이 정의할 수 있습니다.

<br>

- $$ K'K^{-1} = \begin{bmatrix} k & 0 & (1-k)c_{x} \\ 0 & k & (1-k)c_{y} \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} kI & (1-k)\tilde{x}_{0} \\ 0^{T} & 1 \end{bmatrix} $$

<br>


<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/26.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/27.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/28.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>
