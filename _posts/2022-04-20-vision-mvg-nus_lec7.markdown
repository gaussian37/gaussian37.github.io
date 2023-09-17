---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 7. The fundamental and essential matrices
date: 2022-04-20 00:00:07
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [멀티플 뷰 지오메트리, Multiple View Geometry, The fundamental and essential matrices] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/eJnG1vwGJkE?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/mNThwULGR-g?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/7QYq7qNkmtg?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/WQvJICS3Ecc?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/eJnG1vwGJkE" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 먼저 본 강의 내용에 앞서 다음 내용을 간략히 숙지하면 이해하는 데 도움이 됩니다.

<br>

- `epipolar geometry`란 2개이상의 카메라에서 카메라 간의 관계를 추정하는 것으로 생각할 수 있습니다.
- 만약 카메라가 2개라면 스테레오 비전 또는 2-view라고 하며 스테레오 비전에서의 두 카메라의 관계를 `epipolar geometry`로 표현할 수 있습니다. 아래 그림과 같습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림에서 동일한 3차원의 점인 $$ P $$ 를 서로 다른 두 카메라에서 획득하였을 때, 매칭 쌍 $$ (p, p') $$ 사이의 기하학적 관계를 다룹니다. 먼저 위 그림에 각 기호에 대한 설명을 하면 다음과 같습니다.
- 　$$ P(P') $$ : 이미지 상에 맺힐 3차원 공간 상의 점을 의미합니다.
- 　$$ C, C' $$ : 각 영상의 카메라 센터점을 의미합니다.
- 　$$ \text{base line} $$ : 카메라의 센터점을 이은 선을 `base line`이라고 합니다. $$ C, C' $$ 의 거리가 가까운 경우를 `narrow base line`이라고 하고 거리가 먼 경우를 `wide base line`이라고 합니다.
- 　$$ p, p' $$ : 각 영상에서 $$ P(P') $$ 가 투영된 점을 의미합니다.
- 　$$ e, e' $$ : 반대 영상의 카메라 센터점에서 해당 영상의 이미지 상에 맺힌 점을 의미합니다. 이 점을 `epipole` 이라고 합니다.
- 　$$ l, l' $$ : 각 영상에서 `epipole` ( $$ e, e' $$ )과 이미지 상의 점 ( $$ p, p' $$ )를 이은 선을 의미합니다. 이 선을 `epolar line` 이라고 합니다.

<br>

- 위 그림에서 $$ C $$ 와 $$ C' $$ 간의 3차원 위치 관계인 $$ [R \vert T] $$ 와 $$ p $$ 를 알더라도 3차원 공간 상의 점 $$ P $$ 에 대한 실제 깊이 (`depth`) 를 알지 못하면 유일한 $$ p' $$ 를 결정하지 못합니다.
- 반면에 $$ P $$ 는 $$ C $$ 와 $$ p $$ 를 잇는 `ray` 상에 존재하므로 이 선이 반대 영상에 투영된 `epipolar line` $$ l' $$ 은 유일하게 존재합니다.
- 이 때, $$ A $$ 이미지에서 $$ B $$ 이미지로 대응되는 `epipolar line` ( $$ l' $$ ) 의 관계를 나타내는 행렬이 $$ F, E $$ 이며 각각 `Fundamental Matrix`, `Essential Matrix`라고 합니다.

<br>

- `Essential Matrix` $$ E $$ 는 `normalized image plane` 에서의 매칭쌍들 사이의 기하학적 관계를 설명하는 행렬을 의미하고 `Fundamental Matrix` $$ F $$ 는 카메라 파라미터 까지 포함한 두 이미지의 실제 픽셀 좌표 사이의 기하학적 관계를 표현하는 행렬을 의미합니다. 따라서 $$ E $$ 는 $$ F $$ 의 특수한 형태라고 생각할 수 있습니다.

<br>

- 두 이미지 평면 간의 기하학적 관계가 $$ E, F $$ 가 주어지고 두 이미지 평면상의 매칭쌍 $$ p, p' $$ 가 주어질 때, 3D 공간 상의 좌표 $$ P $$ 를 결정할 수 있습니다. 따라서 스테레오 비전에서의 깊이 ('depth')를 구할 수 있습니다.

<br>

- 상세 내용은 본 강의 내용을 통하여 자세하게 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/5.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/6.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/7.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/8.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/9.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/10.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/11.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/12.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/13.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/14.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/15.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/16.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec7/17.png" alt="Drawing" style="width: 1000px;"/></center>
<br>



<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/mNThwULGR-g" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/7QYq7qNkmtg" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/WQvJICS3Ecc" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>
