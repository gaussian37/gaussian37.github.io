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
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/XSflleJr4qM" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/29.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 이전 강의에서는 **카메라의 센터점이 동일**한 상태에서의 서로 다른 상태로 취득한 이미지에 대하여 `homography`를 이용하면 이론적으로 이미지들을 서로 복원할 수 있음을 확인하였습니다.
- 이번 슬라이드에서 부터는 그 이론적인 내용을 이용하여 적용할 수 있는 대표적인 Application인 `Synthetic Views`와 `Planar Panoramic Mosaicing`입니다.

<br>

- 먼저 `Synthetic Views`는 위 그림과 같이 `source image`로 부터 `planar homography`를 이용하여 새로운 형태의 `target image`로 생성하는 것을 의미합니다. `planar homography`를 이용한 이미지 변환은 카메라읜 센터점은 고정한 체 카매라의 촬영 방향을 변화하는 효과를 줄 수 있도록 할 수 있습니다.
- `planar homography`는 3 x 3 크기의 행렬로 어떤 이미지의 점들을 다른 이미지의 점들로 변환할 수 있도록 하며 3 x 3 행렬의 성분을 통하여 `rotation`, `translation`, `scale` 및 `perspective distortion`이 이미지 변환 시 어떻게 바뀌는 지 설명해 줄 수 있습니다.
- `systhetic view`를 생성하기 위해서는 기본적으로 다음의 절차를 따릅니다.
- ① `source image`와 `target image`에 서로 대응되는 (같은 위치를 가리키는) 점들을 확인합니다.
- ② 이 점들 쌍을 이용하여 `homograpy matrix`를 계산합니다.
- ③ `homography`를 적용하여 `source image`를 `target image`로 생성합니다. (`synthetic view`)

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/30.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 앞에서 설명한 바와 같이 `source image`에서 임의의 사각형을 지정하고 `target image`에서 이 사각형이 어디에 위치해 있는 지 알수 있으면 `homography`를 계산할 수 있습니다. `homography`를 계산하는 일반적인 방법은 `source image`와 `target image`에 대응되는 점들의 쌍을 선형 방정식의 해를 푸는 방식으로 구합니다. 자세한 내용은 아래 글을 참조하시면 됩니다.
    - 참조 : [image transformation](https://gaussian37.github.io/vision-concept-image_transformation/)
    - 참조 : [direct linear transformation](https://gaussian37.github.io/vision-concept-direct_linear_transformation/)

<br>

- 이와 같은 방법은 통해 $$ H $$ 를 구하면 `source image` 에서 $$ x $$ 점을 `target image`에서 $$ x' = Hx $$ 로 변환할 수 있으며 필요한 영역의 모든 점을 이와 같이 변환하면 필요한 영역의 이미지를 구할 수 있습니다. 이와 같은 작업은 `warping` 이라고 하기도 합니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/31.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/32.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/33.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/34.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 이번에는 카메라 캘리브레이션이 우리에게 어떤 정보를 주는 지 살펴보도록 하겠습니다.
- 이전 강의에서 다루었듯이 카메라 캘리브레이션을 통하여 카메라의 `intrinsic`와 `extrinsic` 파라미터를 얻을 수 있었고 `intrinsic` 파라미터를 통해 `focal length`, `optical center`, `lens distortion` 등을 구할 수 있으며 `extrinsic` 파라미터를 통하여 카메라와 world의 관계를 구할 수 있었습니다.
- 이전 강의에서는 이와 같은 의미를 가지는 `intrinsic`과 `extrinsic` 파라미터를 이용하여 2D 이미지와 3D 공간 상의 정보를 변환하는 `projection matrix`를 구하여 사용하였고 이 행렬은 $$ 3 \times 4 $$ 크기를 가졌습니다. 또한 `projection matrix`를 역으로 분해를 하면 `intrinsic`과 `extrinsic`을 구할 수 있었습니다.

<br>

- 위 슬라이드에서의 $$ \tilde{X} = \lambda d $$ 는 카메라 좌표계 상의 3D 점을 의미하고 $$ d $$ 는 카메라 센터점으로 부터 뻗어나가는 방향 벡터에 해당하며 $$ \lambda $$ 는 그 방향의 `depth`를 의미합니다.
- 위 슬라이드의 $$ x = K[I \vert 0] ( \lambda d^{T}, 1) ^{T} = Kd $$ 식은 카메라 좌표계의 3D 포인트가 어떻게 2D 이미지에 투영되는 지 나타냅니다. 여기서 $$ K $$ 는 카메라 `intrinsic` 파라미터이고 $$ [I \vert 0] $$ 은 `extrinsic` 파라미터를 나타내며 여기서는 world의 원점에서 간단히 `z` 축을 바라보는 카메라를 의미하도록 하여 간략하게 나타내었습니다.
- 식 $$ d = K^{-1} x $$ 를 통하여 이미지의 점 $$ x $$ 는 다시 $$ d $$ 벡터로 나타낼 수 있으며 이와 관련된 내용은 앞선 강의에서 다루었습니다. `homogeneous` 좌표계로 나타내고 있기 때문에 $$ d $$ 는 반드시 `unit vector`일 필요는 없으며 scale은 달라질 수 있습니다.
- 정리하면 카메라 캘리브레이션 정보는 2D 이미지와 3D 공간 상의 정보를 연결해 주는 역할을 하고 있으며 이 정보는 다양한 task에 사용될 수 있습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/35.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 2D 이미지의 두 점 $$ x_{1}, x_{2} $$ 는 $$ d_{1} = K^{-1}x_{1} $$ 과 $$ d_{2} = K^{-1}x_{2} $$ 로 변형하여 두 벡터 간의 사이각을 구할 수 있습니다. 다음과 같습니다.

<br>

- $$ \begin{align} \cos{\theta} &= \frac{d_{1}^{T} d_{2} }{ \sqrt{d_{1}^{T}d_{1}} \sqrt{d_{2}^{T}d_{2}} } \\ &= \frac{ (K^{-1}x_{1})^{T}(K^{-1}x_{2}) }{ \sqrt{ (K^{-1}x_{1})^{T}(K^{-1}x_{1}) }  \sqrt{ (K^{-1}x_{2})^{T}(K^{-1}x_{2}) } } \\ &= \frac{ x_{1}^{T} (K^{-T}K^{-1})x_{2} } { \sqrt{ x_{1}(K^{-T}K^{-1})x_{1} } \sqrt{ x_{2}^{T}(K^{-T}K^{-1})x_{2} } } \end{align} $$

<br>

- 이 방법은 두 카메라 간의 상대적인 `pose`를 구하거나 `multiple view`를 고려한 3D 점의 `triangulation`을 구할 때 사용되곤 합니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/36.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 카메라 캘리브레이션을 통하여 카메라 `intrinsic` 파라미터 $$ K $$ 를 알게 되었다면 $$ K^{-T}K^{-1} $$ 또한 알 수 있습니다.
- 이 값은 꽤 유용하게 사용될 수 있는데 이 값을 통하여 **이미지의 점**들을 3D 공간 상의 `ray`로 보낼 수 있으며 반대로도 적용할 수 있습니다.
- 정리하면 파라미터 $$ K $$ 만 알 수 있으면 **이미지 상의 점들 만**으로도 3D 공간 상의 `ray`의 방향 또한 알 수 있고 이미지 점들 사이의 `각도` 또한 알 수 있습니다.

<br>
<center><img src="../assets/img/vision/mvg/nus_lec6/37.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 위 슬라이드에서는 이미지 상의 선 $$ l $$ 을 구성하기 위해 카메라 센터 $$ c $$ 로 부터 나온 `ray`인 $$ d $$ 전체와 직교한 방향의 벡터 $$ n $$ 은 $$ n = K^{T} l $$ 로 정의 됨을 보여줍니다.

<br>

- 먼저 앞에서 살펴본 바와 같이 이미지 상의 선 $$ l $$ 의 점을 $$ x $$ 는 다음과 같이 `back-project` 됩니다.

<br>

- $$ d = K^{-1}x $$

<br>

- `ray` $$ d $$ 는 위 그림과 같이 $$ n $$ 벡터에 직교하므로 다음과 같이 전개할 수 있습니다. 아래 식은 직교한 벡터의 `dot product`는 0임을 이용한 것입니다.

<br>

- $$ d^{T}n =  x^{T}K^{-T}n = 0 $$

<br>

- 선 $$ l $$ 상에 존재하는 점 $$ x $$ 에 대하여 $$ x^{T}l = 0 $$ 을 만족합니다. 따라서 식을 다음과 같이 전개할 수 있습니다.

<br>

- $$  x^{T}K^{-T}n = x^{T}l = 0  $$

- $$ K^{-T}n = l $$

- $$ \therefore n = K^{T}l $$

<br>

- 따라서 2D 이미지에서 선을 찾았을 때, 그 선을 구성하는 ray 전체와 직교하는 3D 공간상의 `normal vector`를 찾는 데 $$ K $$ 를 사용할 수 있음을 확인하였습니다. 이와 같은 정보를 이용하면 `2D 이미지 상의 선`을 통해 3D 공간 상의 기하 정보를 추정할 수 있습니다.

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>
