---
layout: post
title: 2023 Wayve 자율주행 영상 분석
date: 2023-04-03 00:00:00
img: autodrive/concept/wayve/0.png
categories: [autodrive-concept] 
tags: [자율주행, 자율주행 자동차, autodrive, self-driving] # add tag
---

<br>

[자율주행 관련 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>

- 출처 : https://www.gatesnotes.com/Autonomous-Vehicles?WT.mc_id=20230329100000_Autonomous-Vehicles_BG-LI_&WT.tsrc=BGLI

<br>

- [Alex Kendall](https://alexgkendall.com/)이 창립한 [Wayve](https://wayve.ai/)의 자율 주행 영상을 빌 게이츠와의 인터뷰를 통하여 공개하였습니다.
- Alex Kendall은 Computer Vision 분야에서 유명한 논문을 많이 쓴 만큼 큰 기대를 하며 영상을 살펴보았습니다. 

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/ruKJCiAOmfg" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 시연 장소는 `wayve` 회사가 있는 런던입니다. 중간에 빌 게이츠가 먹는 피쉬 앤 칩스 가게를 통해 위치를 추정해보면 아래 지역입니다. 한적하면서도 골목이 좁은 자율 주행 테스트를 하기에는 꽤 복잡한 구간입니다.

<br>
<center><img src="../assets/img/autodrive/concept/wayve/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/wayve/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 영상의 내용을 간략하게만 살펴보면 2가지 포인트에서 약간 놀라운 점이 있었습니다.

<br>

- ① `Object Detection`에 대한 시연 영상이 없었습니다. 그 흔한 자동차나 보행자를 2D 또는 3D 형태로 찾은 결과를 따로 보여주지 않았습니다. 실제 구현을 하지 않아도 된다고 판단한 것인 지 또는 동작하지만 데모에 나타나지 않은 것인지는 알 수 없지만 보여주지 않았다는 점에서 구현하지 않은 것으로 생각됩니다.
- 대신에 `Free Space Detection`을 위한 `Semantic Segmentation`과 `Depth Estimation`만 구현하여 데모로 보여주었습니다.

<br>

- ② `Free Space Detection`에 대한 인식 성능이 매우 높지 않음에도 자율주행이 가능해 보였다는 점입니다.

<br>
<center><img src="../assets/img/autodrive/concept/wayve/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 영상에서 Free Space Detection을 위한 Semantic Segmentation과 Depth Estimation의 출력을 나타낸 것입니다. 데모 영상인데도 인식 성능이 깔끔하지 않은 것을 볼 수 있습니다.

<br>

- 아래는 Depth Estimation 정보를 이용하여 Semantic Segmentation 컬러 (+ RGB 컬러 블렌딩)한 픽셀을 2D → 3D 로 변환하여 시각화한 결과 입니다. 

<br>
<center><img src="../assets/img/autodrive/concept/wayve/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/wayve/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/wayve/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/wayve/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 영상에 자세히 소개 되지는 않았지만, `wayve`의 [공식 홈페이지 소개](https://wayve.ai/technology/av2-0/)에서는 전방, 좌측방, 우측방 각 1개씩의 카메라를 사용하여 총 3개의 카메라로 영상 인식을 하는 것으로 소개합니다.

<br>
<center><img src="../assets/img/autodrive/concept/wayve/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림과 같이 밴과 세단 2가지 타입의 차를 운용하고 있으며 카메라 센서의 위치는 비슷한 위치에 장착한 것을 볼 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/wayve/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 전방 카메라와 좌측방, 우측방 카메라의 중첩 영역을 확인해 보려고 했지만, 데모로 올라온 영상은 카메라가 동시에 촬영되지 않은 영상을 업로드 하여 확인할 수 없었습니다. (중간 중간 캡쳐 마다 중첩 영역이 다릅니다.) 따라서 영상 뷰 정도만 확인할 수 있었습니다.

<br>

- 지금 까지 살펴본 내용이 짧은 영상에서 확인할 수 있는 `wayve`의 자율주행 관련 내용입니다. 가장 인상 깊은 점은 단순한 영상 인식 성능을 이용하여 자율주행을 하고 있다는 점입니다.

<br>

___

<br>

- 런던에 방문하면서 Wayve 본사가 어떤 곳인 지 궁금하여 입구 까지 한번 가 보았습니다. (23년 12월 10일, 오후 4시 30분)
- 런던의 겨울에는 밤이 굉장히 길어서 오후 4시만 되어도 해가 지기 시작합니다.

<br>
<center><img src="../assets/img/autodrive/concept/wayve/11.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/wayve/10.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

- 회사는 산업 단지가 약간 밀집 되어 있는 곳에 위치하였습니다. 주말이라서 회사내에 차고 처럼 생긴 곳에 대부분의 차량이 주차되어 있는 것 같았고 일부 차량들만 밖에 세워져 있었습니다.

<br>
<center><img src="../assets/img/autodrive/concept/wayve/12.jpg" alt="Drawing" style="width: 800px;"/></center>
<br>

- Wayve 홈페이지에 별도 설명되어 있진 않지만 야외 주차장에 있는 차량을 보면 라이다가 달려있는 것을 볼 수 있습니다. Wayve 또한 다른 회사와 마찬가지로 라이다 데이터를 취득하여 개발에 활용하는 것으로 추정됩니다.

<br>

[자율주행 관련 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>