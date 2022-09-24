---
layout: post
title: 스트라드비전 (StradVision) 영상 인식 결과 분석 
date: 2022-09-20 00:00:00
img: autodrive/concept/stradvision/2.png
categories: [autodrive-concept] 
tags: [스트라드비전, stradvision] # add tag
---

<br>

- 출처 : https://youtu.be/QS0cRlKNYvM ([MOCAR](https://www.youtube.com/user/digitrio))

<br>

- 본 글은 22년도 8월에 MOCAR 채널에서 업로드한 스트라드비전의 인터뷰 영상에서 데모 영상에서 발췌하였습니다. 
- 최근에 테슬라를 선두로 영상 인식 기술 회사의 상세한 개발 이력을 공유하는 경향이 있습니다. 스트라드비전에서도 MOCAR와의 인터뷰에서 상세한 출력 사양을 공개하면서 기술력을 자랑하였고 인상 깊은 점들에 대하여 정리하였습니다.

<br>

- 스트라드비전에서는 9개의 카메라를 사용하고 있으며 그 중 60도 화각의 전방 카메라에 대하여 출력 영상을 실제 차량에서 보여줍니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 위 그림과 같이 5개의 화면을 분할하여 출력 결과를 확인합니다. 
- ① 결과 : 왼쪽 상단의 ① 영역에서는 `2D/3D Object Detection`, `Line Detection`의 결과를 2D 이미지 상에서 보여줍니다.
- ② 결과 : ② 영역에서는 ① 영역에서 보여준 결과를 Bird Eye View 형식으로 보여줍니다. 인식된 결과의 색상이 ① 과 동일한 것을 볼 수 있습니다.
- ③ 결과 : 전방 카메라가 인식한 Depth Estimation 결과를 3D 상에서 보여줍니다. 60도 화각의 카메라를 사용 하였기 때문에 화각이 60도 형태로 출력됨을 알 수 있습니다.
- ④ 결과 : 전방 카메라가 인식한 Depth Estimation 결과를 2D 이미지 상에서 보여줍니다. 2D 이미지와 대응하여 볼 수 있습니다.
- ⑤ 결과 : Semantic Segmentation의 결과 입니다. 픽셀 별로 어떤 분류의 물체 또는 배경인 지 결과를 보여줍니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 먼저 2D/3D Object Detection과 Line Detection 결과를 살펴보겠습니다.
- 2D, 3D Object Detection과 Line Detection을 한번에 출력하는 딥러닝 모델을 사용하는 지 또는 2D, 3D Object Detection은 한번에 하고 Linde Detection을 별개로 처리하는 지 궁금증이 남습니다.
- 먼저 Object Detection 결과를 살펴보면 `3D Detection`은 자동차 (승용차, 트럭, 버스 등)를 위주로 인식하고 있습니다. 반면 `2D Detection`에는 차 이외의 물체인 보행자, 오토바이, 신호등 (+ 신호등 색), 표지판 등을 위주로 인식합니다. (오토바이의 사람은 보행자로 별도 표시하지 않는 것으로 확인됩니다.)
- 각 Bounding Box의 출력을 살펴보면 박스 상단에 클래스 이름과 박스 하단에는 거리값이 적혀져 있습니다. ② 영역을 참조하였을 때, 원점 기준은 카메라 좌표계의 원점인 것으로 추정되며 부호의 방향으로 보았을 때, 오른손 좌표계를 따르는 것으로 보입니다. 오른손 좌표계 기준으로 Boundinb Box 아래의 (X, Y) 좌표에서 `X`는 검지 손가락에 해당하는 차량 앞쪽을 나타내고 `Y`는 차량 왼쪽을 나타내는 중지 손가락에 해당합니다. 위 그림의 `MOTB`인 오토바이는 카메라 기준으로 5.6 m 전방에 있고 1.5 m 왼쪽에 있는 것으로 추정됩니다.
- 오른쪽 하단에 `FPS` : 14.00 (~ 16.00) 정도 확인되며 ① 결과인 지 ① ~ ⑤ 결과인 지 확인은 되지 않으나 15 FPS를 목표로 개발된 것으로 추정됩니다.

<br>

- `Line Detection`의 출력 결과를 살펴보면 `차선의 유형`, `차선의 갯수`, `차선의 위치`에 따라 구분하여 표시합니다.
- `차선의 유형`은 실선(Solid)인 지, 점선(Dashed)인 지 차선의 유형을 표시하고 `차선의 갯수`는 1줄 차선인 지 2줄 차선인 지 등을 구분합니다. `차선의 위치`는 `EGO LEFT`, `EGO RIGHT`클래스로 자차 바로 주변의 차선을 찾고 그 옆차선을 찾기 위하여 `NEXT LEFT`, `NEXT RIGHT` 또한 구분합니다. 그리고 그 이외의 차선은 `OTHER`로 구분하여 위치 별 총 5개 차선으로 구분합니다.
- 자차의 양쪽 차선인 `EGO LEFT`, `EGO RIGHT`의 영역은 색이 칠해진 것으로 표시됩니다. 이 색은 자차의 앞에 있는 차 까지 색이 칠해져서 자차와 앞차와의 간격을 확인할 수 있도록 표시합니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- ③, ④ 결과는 Depth Estimation 모델을 통하여 얻은 Depthmap 입니다. Depth Estimation 모델을 이용하면 ④와 같은 결과를 얻을 수 있고 위 색상은 거리값에 따라 색상을 달리하여 보여준 것입니다.
- ④ 결과와 카메라 intrinsic을 이용하면 ③을 얻을 수 있습니다. 즉, 이미지의 픽셀 (u, v)를 3차원 좌표계 (X, Y, Z)로 변환할 수 있습니다. (참조 : https://gaussian37.github.io/vision-depth-pcd_depthmap/)
- Depth Estimation의 인식 가능 거리는 최대 80 ~ 100 m로 추정합니다. Depth Estimation이 통상적으로 50 ~ 80 m 정도 인식하고 ②에서 최대 거리도 100 m 미만이기 때문입니다.

<br>

- 마지막으로 ⑤ 의 Semantic Segmentation에서는 픽셀 별 클래스를 구분하는 역할을 하며 영상을 참조하면 실제 출력 클래스를 살펴볼 수 있습니다. (영상에는 나오지 않은 잘 등장하지 않는 클래스 또한 존재할 것으로 추정합니다.)
    - 1) 진한 보라색 : 도로
    - 2) 노란색 : 자동차
    - 3) 밝은 보라색 : 구조물
    - 4) 연두색 : 가로등 및 표지판
    - 5) 밝은 초록색 : 나무
    - 6) 진한 초록색 : 건물
    - 7) 청록색 : 하늘
    - 8) 회색 : 사람, 오토바이
    - 9) 하늘색 : 인도
    - 10) 어두운 보라색 : ? (추정 어려움)

<br>

- 영상에 나온 화면 출력을 보면서 출력 결과를 살펴보겠습니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/1.gif" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 자차와 앞차 사이의 영역을 별도 표시하는 것을 확인할 수 있습니다.
- Depth Estimation의 결과를 보았을 때, 경계면에서의 구분이 가능한 정도로 잘 나오고 있으나 움직이는 물체에선 일부 뭉게지는 것이 있음을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/2.gif" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 사람을 인식할 때 우산, 가방 등의 물건에서 Depth Estimation과 Segmentation 결과가 오인식 하는 것을 볼 수 있습니다. 
- Depth Estimation 결과에서 Depth의 오인식이 있는 것으로 보아 우산이 있는 데이터가 많지 않은 것으로 보이며 Segmentation 결과를 봐서 보행자의 물건에 대한 별도 클래스는 없는 것으로 확인됩니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/3.gif" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 오토바이에 대한 오인식이 확인 됩니다. 상대 속도의 영향도 있겠지만 뛰어가는 사람에 비해 인식 성능이 떨어지는 것을 보면 데이터가 많지 않을 것으로 생각됩니다.
- 횡단보도 및 차선 등에 대한 별도 Segmentation을 하지 않는 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/stradvision/4.gif" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 도심에 일반 차선이 아닌 유도선이나 안전지대와 같은 경우에서 차선에 대한 오인식이 많이 발생하는 것을 확인할 수 있습니다. Segmentation에서는 도로의 어떤 노면 정보도 별도 인식하지 않고 모두 도로로 인식하는 점도 확인할 수 있습니다.
- 펜스 너머로 보이는 원거리 영역의 차들과 도로 등에서 오인식이 확인 되며 구조물 등에서도 오인식이 발생하는 것으로 확인 됩니다.

<br>

- 지금까지 스트라드비전의 출력 영상을 보고 추정한 것들을 적어보았습니다. MOCAR 덕분에 좋은 영상을 볼 수 있어서 감사합니다.