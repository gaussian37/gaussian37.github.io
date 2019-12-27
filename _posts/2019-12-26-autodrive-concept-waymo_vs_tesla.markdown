---
layout: post
title: 자율주행에서의 waymo와 tesla
date: 2019-12-26 00:00:00
img: autodrive/concept/waymo_vs_tesla/0.png
categories: [autodrive-concept] 
tags: [자율주행, autonomous drive, waymo tesla] # add tag
---

<br>

<br>
<div style="text-align: center;">
    <iframe src="https://youtu.be/6SCj3S3ZoOU" frameborder="0" allowfullscreen="true" width="600px" height="400px"> </iframe>
</div>
<br>

<br>

- 참조한 영상을 살펴보면 `waymo`와 `tesla`의 자율주행 컨셉의 차이는 어떤 센서를 주로 사용하는 지에 따라 나뉘어 있습니다.
- 결론적으로 말하면 `waymo`는 라이다를 주로 사용하고 `tesla`는 카메라를 주로 사용하여 자율주행을 합니다.
- 먼저 tesla의 입장은 카메라를 사용하는 것이 좀 더 저렴한 것이 이유이고 비싼 라이다를 사용할 필요가 없다는 입장이고
- 영상에 occlusion이나 날씨로 인한 문제가 발생하는 경우에는 레이더를 이용하여 보완하겠다는 생각입니다.
- 영상을 이용하면 사람이 할 수 있는 것들, 예를 들어 교통 표지판을 읽는것 등을 할 수 있는 장점도 있습니다. 
- 물론 카메라를 통한 영상 데이터를 이용하였을 때의 단점은 deep learning 학습에 필요한 많은 데이터가 필요하고 실제 차에서 동작할 때 power가 많이 들어간다는 것입니다.
- 특히 tesla에는 8개의 카메라가 들어가니 power가 큰 문제가 될 것 같습니다.
- 또한 문제가 발생하였을 때, 새로운 데이터를 이용하여 재학습하는 방식으로 문제들을 풀어나아가야하는 불편함도 있습니다. 

<br>
<center><img src="../assets/img/autodrive/concept/waymo_vs_tesla/1.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- `waymo`에서의 메인 센서는 라이다 입니다. 물론 테슬라에서 사용하는 카메라와 레이더도 있고 기타 센서들도 더 있지만 메인 센서는 라이다 입니다.

<br>
<center><img src="../assets/img/autodrive/concept/waymo_vs_tesla/2.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- `tesla`는 8대의 카메라가 메인 센서가 됩니다. 전방 레이더와 12개의 초음파 센서는 거리를 측정할 때 사용됩니다. 

<br>
<center><img src="../assets/img/autodrive/concept/waymo_vs_tesla/3.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- `waymo`의 접근 방법 중에 또 중요한 것은 정밀 지도 입니다. 현재 자율주행을 시범적으로 하고 있는 곳에는 정밀 지도가 있어서 데모가 가능한 상태이나 그 이외의 지역은 아직 확장해나아갈 계획이 있는 상태이지 당장 운행할 수 있는 상황은 아닙니다. 

<br>
<center><img src="../assets/img/autodrive/concept/waymo_vs_tesla/4.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- 반면에 `tesla`는 영상 데이터를 기반으로 움직이기 때문에 deep learning 학습을 완료하는 데 꽤 수고스럽지만 한번 학습이 완료되면 비슷한 환경에서는 자율주행을 할 수 있습니다. 이것이 `waymo`와의 접근 방법의 차이입니다.

<br>

- **가격** 측면에서 살펴보면 차 값을 제외한 센서값을 비교하였을 때, `tesla`가 저렴합니다. 그 목적을 두고 설계를 한 만큼 비용 측면에서는 tesla에 우위가 있습니다. 

<br>

- 일단 `tesla`는 양산 모델들이 있기 때문에 `waymo`에 비해서 실제 차에서 수집할 수 있는 데이터의 양이 훨씬 많습니다.
- 성능을 끌어올리기 위해 데이터가 중요하기 때문에 이 점에서 또한 `tesla`가 우위에 있습니다. 
- 위에서 설명한 바와 같이 `waymo`는 현재 제한된 지역에서 tesla에 비해 턱없이 작은 수의 차만 도로위에서 굴러가고 있기 때문에 데이터 취득 측면에서는 좋지 못한 상황입니다. 
- 하지만 모든 것이 `tesla`가 좋은 것이 아닙니다. 사실 `waymo`와 자율주행단계가 다르기 때문에 비교하기도 애매한 것이 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/waymo_vs_tesla/4.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- 하지만 `waymo`는 고가의 라이다 기반이기 때문에 현재 tesla와 자율주행단계가 다릅니다. 어찌보면 각 회사의 타겟이 다르다고 볼 수 있습니다.
- `waymo`는 현재 제한된 지역에서 자율주행4단계를 하고 있는 반면 `tesla`의 기능은 자율주행 2단계 입니다. 
- 즉, `waymo`는 제한된 지역에서 라이다와 정밀지도를 이용하여 좀 더 제대로 해보겠다는 상태이고, `tesla`는 범용적으로 낮은 단계의 자율주행을 해보겠다는 상태입니다.
