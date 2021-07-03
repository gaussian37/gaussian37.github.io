---
layout: post
title: Andrej Karpathy (Tesla) CVPR 2021 Workshop on Autonomous Vehicles 정리
date: 2021-06-30 00:00:00
img: autodrive/concept/tesla_cvpr_2021/0.png
categories: [autodrive-concept] 
tags: [자율주행, 자율주행 자동차, 테슬라, cvpr, cvpr 2021, workshop] # add tag
---

<br>

[Autonomous Driving 관련 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>

- 출처 : https://www.youtube.com/watch?v=NSDTZQdo6H8

<br>

- 이번 글은 Andrej Karpathy가 CVPR 2021에서 발표한 Vision만을 이용하여 어떻게 테슬라에서 자율주행을 하는 지에 대한 간략한 설명을 정리하였습니다.
- 영상의 주요 내용은 `Vision`만으로도 자율주행을 위한 인식이 사용 가능하다는 것과 vision의 성능을 향상하기 위하여 어떻게 데이터를 구축하였는 지에 대한 내용입니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 약간 표현이 어색하지만 사람을 `Meat Computer` 라고 표현하였습니다. 사람은 80mph가 넘는 속도로 움직이는 1톤의 물체를 정교하게 제어해야 합니다. 
- 하지만 사람은 250ms의 latency를 가지고 주변 상황을 인식하기 위해서는 고개를 돌려서 상황을 직접 봐야 하는 한계가 있습니다. 심지어 인스타그램을 하는 딴 짓을 하기도 합니다. 이러한 단점들로 인하여 전세계적으로 약 3700명의 사람이 교통사고로 죽어가고 있고 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 컴퓨터는 사람의 이러한 단점을 개선할 수 있습니다. (컴퓨터를 `Silicon Computer`로 표현하였습니다.)
- 컴퓨터는 먼저 latency가 100ms 이하이고 360도 전방을 감지할 수 있습니다.
- 또한 사람처럼 딴 짓을 하지 않고 상시 전 집중을 하여 주변 상황을 인지합니다.
- 이러한 장점들로 인하여 교통 사고 문제를 개선할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 현재 테슬라에서 제공하는 자율주행 시스템은 `AEB (Autonomous Emergency Braking, 긴급 제동)`, `Traffic Control Warning`, `PMM (Pedal Misapplication Mitigation, 운전자 페달 오조작 완화)`와 같은 기능을 제공하고 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 2021년 기준 현재 테슬라의 FSD Beta는 약 2,000명의 고객에게 임시로 Full Self-Driving 을 제공하고 있으며 인식 성능은 Display를 통하여 확인할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 테슬라와 같은 기능은 테슬라 뿐 아니라 웨이모와 같은 다른 회사에서도 만들어 서비스를 하고 있지만 주변 상황을 인식하는 데 많은 차이점이 있습니다.
- 위 그림은 웨이모와 같은 회사에서 주변 상황을 인식하기 위한 센서를 종합한 것입니다. 즉, Vision 뿐 아니라 Lidar와 HD Map 까지 사용하고 있습니다.
- 라이다는 현재 상대적으로 가격이 비싸며 차량 360도 주변의 거리 검색을 하여 포인트 클라우드를 생성합니다. 이 때, 라이다 센서로 주의 환경을 미리 매핑을 해야 하는 전제 조건이 있습니다. 즉, 고해상도 지도(HD Map)를 만들어야 합니다. HD Map에서는 주변 인프라의 위치, 교통 신호, 차선 등도 포함하고 있습니다. 이렇게 고비용으로 만든 HD Map을 이용하여 자차의 위치를 추정하고 주행을 하는 것이 웨이모의 방식입니다.
- 하지만 테슬라는 라이다, HD Map 모두를 사용하지 않고 단순히 Vision만 사용합니다. 즉, 기존에 알고 있는 도로를 가는 것이 아니라 매번 새로운 도로를 가는 것으로 비유할 수 있습니다. (물론 학습 데이터에 사용된 도로에 갈 수도 있습니다. 하지만 HD Map 처럼 정보를 가지고 있지는 않습니다.)
- 테슬라에서 이와 같은 Vision 기반의 방식을 이용하려는 이유는 전 세계 곳곳에 서비스를 제공하기 위함입니다. 전 세계의 HD Map을 만들고 유지 보수하는 것은 굉장히 많은 비용이 발생하여 현실성이 없기 떄문입니다.
- **물론 딥러닝 기반의 Vision 시스템을 이용하여 HD Map 없이 이 문제를 해결하는 것은 굉장히 어렵습니다. 하지만 이 문제를 테슬라에서는 Vision 시스템으로 해결하고자 하였습니다.** 만약 Vision 시스템으로 이 문제를 해결하면 어느 지역에서는 일반화 시킬 수 있기 때문입니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림은 테슬라에서 사용하는 센서 현황입니다. 라이다, 레이더, HD Map은 없으며 `카메라`, `초음파 센서`, `오도메터리 센서`, `GPS` 입니다. 주요 센서는 `카메라`라고 말할 수 있습니다.
- 기존에 사용하던 레이더도 이제 삭제되어 양산되고 있습니다. 테슬라 내부적으로 검토하였을 떄, 카메라의 인식이 오히려 레이더 보다 더 낫다고 판단하고 있으며 카메라와 레이더의 판단이 서로 다를 때, 카메라의 결과를 믿고 쓰고 있다고 합니다. 따라서 redundancy로 사용 하는 레이더를 삭제하고 그 리소스를 카메라를 이용한 Vision에 더 주는 것이 낫다고 판단하였습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 테슬라는 현재 8개의 카메라를 사용하고 있으며, 각각 전방 3개, 측방 2개, 후측방 2개, 후방 1개의 영역을 살피고 있고 각각의 카메라 영상을 뉴럴넷을 통하여 물체의 범위 및 깊이를 예측하고 있습니다.
- 사실 인간이 시각만을 이용하여 굉장히 정확하게 파악하지는 못하지만 주변 모든 물체의 깊이와 속도를 이해하면서 운전하고 있습니다. 뉴럴넷도 이와 비슷하게 동작할 수 있습니다. 심지어 사람의 느낌적으로 느끼는 거리보다는 더 정량적으로 우수하게 인식할 수 있습니다.

<br>
<center><img src="../assets/img/autodrive/concept/tesla_cvpr_2021/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서는 `레이더`가 인지한 앞의 차량의 거리, 속도, 가속도를 나타냅니다. 레이더는 위 그림과 같이 인식하고자 하는 물체의 거리, 속도, 가속도를 계산하는 데 좋은 성능을 가집니다.
- 하지만 레이더는 때때로 오인식이나 노이즈를 발생시킵니다. 예를 들어 전방에 다리가 있는 상황에서 다리를 차로 인식하여 정차한 차로 오인식 하는 경우가 발생하곤 합니다.
- 이와 같은 상황에서 영상에서 인식한 거리, 속도 등의 정보와 레이더의 정보를 센서 퓨전하는 데 어려움이 있습니다.
- 다행이도 대량의 데이터로 학습한 영상 기반의 뉴럴넷에서 인식한 거리와 속도가 레이더에서 인식한 것과 유사한 수준에서 확인할 수 있었습니다. 따라서 레이더 없이 영상 기반으로 거리, 속도를 인식할 수 있었습니다. 여기서 핵심은 `데이터 수집`입니다.


<br>

[Autonomous Driving 관련 글 목록](https://gaussian37.github.io/autodrive-concept-table/)

<br>