---
layout: post
title: Tesla 카메라 센서 (2023)
date: 2023-05-20 00:00:00
img: autodrive/concept/tesla_cameras/0.png
categories: [autodrive-concept] 
tags: [tesla, 테슬라, camera, 카메라] # add tag
---

<br>

- 참조 링크 : https://www.tesla.com/autopilot

<br>

- 이번 글에서는 `Tesla`에서 사용하는 카메라 센서의 환경에 대하여 살펴보도록 하겠습니다. 이 글은 2023년 기준으로 작성된 것이며 이후에는 달라질 수 있습니다.
- 아래 영상은 `Tesla AI Day 2022`에서 발표된 영상이며 본 글에서 사용하는 이미지는 아래 영상에서 참조하였습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/ODSJsviD_SU" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

- 테슬라는 8개의 카메라를 이용하여 인식을 하고 있으며 카메라의 종류는 다음과 같이 6개 입니다. `Forward Looking Side Camera`와 `Rearward looking Side Camera`가 각 2개이므로 총 8개의 카메라를 사용합니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 이미지에서 각 카메라 별 화각을 구한 방법은 아래와 같이 각도기를 이용해서 구하였습니다. 그림의 화각이 실제 카메라 화각과 의미가 없을 수 있겠지만, 공식 사이트에 적혀져 있는 `Wide Forward Camera`가 120도, `Forward looking Side Camera`가 90도로 소개된 것이 실제 각도를 아래와 같이 측정한 것과 동일한 것으로 보았을 때, 유사하다고 추정해 보았습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 일반적으로 카메라 화각이 클수록 넓은 영역을 볼 수 있으나 멀리보지 못하고 카메라 화각이 작을수록 좁은 영역만 볼 수 있으나 먼 영역까지 볼 수 있습니다.
- 이 내용은 `focal length`와 연관이 있으며 상세 내용은 [카메라 캘리브레이션](https://gaussian37.github.io/vision-concept-calibration/)을 참조하시면 됩니다.
- 위 내용을 통하여 확인할 수 있는 `카메라 화각`과 `최대 인식 거리`의 상관 관계를 살펴보면 다음과 같습니다.
- `카메라 화각`과 더불어 해상도도 중요하지만 해상도 관련 상세 정보는 부족하여 생략하였습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 각 카메라의 역할 설명을 공식 홈페이지에서 찾으면 다음과 같습니다.

<br>

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/8.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `Tesla AI Day 2022`에서 살펴본 카메라 별 실제 출력 영상을 살펴보면 다음과 같습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서는 조금 잘려 있지만 각 방향의 카메라를 일부 확인할 수 있습니다. 전방 카메라는 카메라 커버가 보일 정도로 넓은 영역을 볼 수 있으므로 `Wide Front Camera`가 아닐까 생각이 듭니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서는 `Wide Front Camera`와 `Main Forward Camera` (또는 `Narrow Forward Camera` 일 지도 모릅니다.)를 비교하여 보여줍니다. 확실히 좁은 영역을 더 자세하게 보여줄 수 있다는 점에서 `Wide Front Camera`와의 차이점이 있습니다.
- 카메라 화각이 줄어서 검은색 영역의 카메라 커버 부분도 없어진 것을 볼 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cameras/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 통하여 각 카메라의 실제 영상을 대략 관찰할 수 있었습니다.

<br>

- 아쉬운 점은 `Narrow Forward Camera`의 영상을 볼 수 없었다는 점입니다. 표현되지 않은 `Rear View Camera`는 실제 테슬라 차량에서 후진을 통해 확인할 수 있습니다.

<br>

## **HW3.0 Vs. HW4.0**

<br>

- 해외 테슬라 유저가 2018년 모델 S 차량과 2023년 모델 Y 차량으로 `HW3.0`을 사용하는 모델 S의 카메라 이미지와 `HW4.0`을 사용하는 모델 Y의 카메라 이미지를 비교한 유튜브 영상입니다.
- 시간이 지남에 따라 카메라의 화질이 개선된 점이 보입니다. AI Day의 발표 영상에 따르면 테슬라 FSD의 입력 이미지는 Raw 영상 그대로를 사용하는 것으로 밝혀져서 카메라 이미지의 개선이 실제 사람이 보는 이미지의 개선 만큼 직접적이지는 않을 수 있습니다. 하지만 시간이 지난 만큼 카메라 해상도의 개선과 그것을 처리할 수 있는 칩의 개선도 있을 것으로 추정됩니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/zcpfeMXM344" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>