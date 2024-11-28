---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 12. Generalized cameras
date: 2022-04-20 00:00:10
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [멀티플 뷰 지오메트리, Generalized cameras] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/PUgr2VKlNbc?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/HuxLHhvdBLY?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/E-YXFI5xzNM?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/PUgr2VKlNbc" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/4.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/5.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/6.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/7.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/8.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/9.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- `plucker vector` 설명: https://en.wikipedia.org/wiki/Pl%C3%BCcker_coordinates

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/9_1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림에서 $$ d $$ 는 `direction vector`이고 $$ m $$ 은 `moment vector`를 의미합니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/10.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/11.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/12.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/13.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/14.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/15.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- `Plucker Vector`의 `Rigid Transformation`을 적용하면 `direction vector` $$ q $$ 와 `moment vector` $$ q' $$ 는 다음과 같이 변환될 수 있습니다.

<br>

- $$ p' = Rp + t $$

- $$ q_{\text{transformed}} = Rq $$

- $$ \begin{align} q'_{\text{transformed}} &= p' \times q_{\text{transformed}} \\ &= (Rp + t) \times (Rq) \\ &= (Rp) \times (Rq) + t \times (Rq) \\ &= R(p \times q) + t \times (Rq) = Rq' + t \times Rq \\ &= Rq' + [t]_{\times} Rq \end{align} $$

<br>

- 따라서 위 슬라이드의 행렬과 같이 표현할 수 있습니다.

- $$ \begin{bmatrix} R & 0 \\ [t]_{\times} R & R \end{bmatrix} \begin{bmatrix} q_{1} \\ q'_{1} \end{bmatrix} = \begin{bmatrix} Rq_{1} \\ [t]_{\times} Rq_{1} + Rq'_{1} \end{bmatrix} $$

<br>

<br>

- 한 쌍의 선이 교차할 때의 필요 충분 조건에 대하여 다루어 보도록 하겠습니다. 필요 충분 조건에 대한 수식 증명을 위해서는 아래 식을 증명해야 합니다.

<br>

- $$ q_{b}^{T}q'_{a} + q_{b}'^{T}q_{a} = 0 $$

<br>

- 위 식을 증명하기 위하여 두 선 $$ L_{a} $$ 와 $$ L_{b} $$ 를 다음과 같이 표현해 보겠습니다.

<br>

- $$ L_{a} = p_{a} + t_{a}q_{a} \quad (t_{a} \text{ is scalar.}) $$

- $$ L_{b} = p_{b} + t_{b}q_{b} \quad (t_{b} \text{ is scalar.}) $$

<br>

- 이 때, 두 선 $$ L_{a}, L_{b} $$ 가 교차한다면 다음과 같이 식을 적을 수 있습니다.

<br>

- $$ p_{a} - p_{b} = t_{b}q_{b} - t_{a}q_{a} $$

<br>

- 따라서 벡터 $$ p_{a} - p_{b} $$ 는 $$ q_{a}, q_{b} $$ 벡터의 `span`된 `plane` 상에 존재함을 알 수 있습니다.

<br>

- 지금부터 $$ q_{b}^{T}q'_{a} + q_{b}'^{T}q_{a} = q_{a}q'_{b}^{T} + q_{b}^{T}q'_{a} = 0 $$ 식에 대한 유도를 진행해 보도록 하겠습니다.

<br>

- $$ q'_{a} = p_{a} \times q_{a} $$

- $$ q'_{b} = p_{b} \times q_{b} $$

- $$ q_{a}q'_{b}^{T} + q_{b}^{T}q'_{a} = q_{a} \cdot (p_{b} \times q_{b}) + q_{b} \cdot (p_{a} \times q_{a}) $$

<br>

- 스칼라 삼중적 성질을 이용하면 $$ a \cdot (b \times c) = b \cdot (c \times a) $$ 이므로 위 식을 다음과 같이 변형할 수 있습니다.

<br>

- $$ \begin{align} q_{a}q'_{b}^{T} + q_{b}^{T}q'_{a} &=q_{a} \cdot (p_{b} \times q_{b}) + q_{b} \cdot (p_{a} \times q_{a}) \\ &= p_{b} \cdot  (q_{b} \times q_{a}) + p_{a} \cdot (q_{a} \times q_{b}) \\ &= p_{b} \cdot (q_{b} \times q_{a}) - p_{a} \cdot (q_{b} \times q_{a}) \\ &= (p_{b} - p_{a}) \cdot (q_{b} \times q_{a}) \end{align} $$

<br>

- 이 때, $$ p_{a} - p_{b} $$ 는 $$ q_{a}, q_{b} $$ 의 선형 결합으로 `span` 된 평면 상에 존재하는 벡터임을 앞에서 확인하였습니다. 그리고 $$ q_{b} \times q_{a} $$ 는 $$ q_{a}, q_{b} $$ 에 의해 만들어진 평면과 직교합니다.
- 따라서 $$ p_{a} - p_{b} $$ 는 $$ q_{a}, q_{b} $$ 에 형성되는 평면상에 존재하고 $$ (q_{b} \times q_{a}) $$ 는 평면과 직교하므로 $$ p_{a} - p_{b} $$ 와 $$ (q_{b} \times q_{a}) $$ 또한 직교합니다. 따라서 다음 식을 만족해야 합니다.

<br>

- $$ q_{a}q'_{b}^{T} + q_{b}^{T}q'_{a} = (p_{b} - p_{a}) \cdot (q_{b} \times q_{a}) = 0 $$

<br>

- 앞에서 두 선이 교차한다는 조건인 $$ p_{a} - p_{b} = t_{b}q_{b} - t_{a}q_{a} $$ 식을 사용하였으므로 두 선이 교차할 때, $$ q_{a}q'_{b}^{T} + q_{b}^{T}q'_{a} = 0 $$ 이 만족함을 확인할 수 있습니다.

<br>

- 선들이 교차하는 경우, 선 위의 점들 사이의 벡터는 `direction vector`에 의해 형성되는 평면에 있기 때문에 스칼라 삼중곱은 0이 되어야 합니다. 따라서 필요 조건을 만족함을 확인하였습니다. 반대로 $$ q_{a}q'_{b} + q_{b}^{T}q'_{a} = 0 $$ 인 경우에 앞에서 보인 바와 같이 선들이 교차하므로 충분 조건을 만족합니다. 따라서 필요 충분 조건을 만족함을 알 수 있습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec12/16.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 앞에서 유도한 식을 이용하여 위 슬라이드의 식을 전개해 보도록 하겠습니다.

<br>

- $$ q_{a}q'_{b}^{T} + q_{b}^{T}q'_{a} = 0 $$

- $$ \Rightarrow q_{b}^{T}q'_{a} + q_{a}q'_{b}^{T} = q_{2}^{T}q'_{1} + q_{1}q'_{2}^{T} = 0 $$

- $$ \Rightarrow q_{2}^{T}([t]_{\times} Rq_{1} + Rq'_{1}) + q'_{2}^{T}q_{1} = 0 \quad (\because q'_{1} = [t]_{\times} Rq_{1} + Rq'_{1}) $$

- $$ \therefore \quad q_{2}^{T}[t]_{\times} Rq_{1} + q_{2}^{T}Rq'_{1} q'_{2}^{T}q_{1} = 0 $$

<br>

- 따라서 행렬로 표현하면 다음과 같이 표현할 수 있습니다.

<br>

- $$ \begin{bmatrix}q_{2} \\ q'_{2} \end{bmatrix}^{T} \begin{bmatrix} E & R \\ R & 0 \end{bmatrix} \begin{bmatrix} q_{1} \\ q'_{1} \end{bmatrix} = 0 $$

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/HuxLHhvdBLY" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/E-YXFI5xzNM" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>



<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>
