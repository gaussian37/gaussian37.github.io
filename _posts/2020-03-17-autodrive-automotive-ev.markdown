---
layout: post
title: 전기 자동차에 관하여
date: 2020-03-14 00:00:00
img: autodrive/automotive/automotive.png
categories: [autodrive-automotive] 
tags: [자동차 공학, 전기 자동차] # add tag
---

<br>

- 참조 : 자동차 에코기술 교과서
- 이번 글에서는 전기 자동차에 관하여 여러 방면으로 알아보겠습니다. 전기자동차 하면 배터리가 떠올릴 수도 있지만 **강력한 토크와 가속력**이 떠오를 수도 있습니다.

<br>

## **모터는 내연기관 보다 에너지 효율이 좋다.**

<br>

- 가솔린 엔진이나 디젤 엔진의 경우 마찰에 따른 손실이 발생하고 연소로 발생한 열의 대부분을 그냥 버리기 떄문에 열효율이 낮습니다.
- 반면 모터는 전력의 대부분을 구동력으로 변환하기 때문에 에너지 효율의 측면에서는 압도적으로 좋습니다. 이 이유로는 **모터 구조가 단순하여 손실이 적기 때문**입니다.
- 물론 전기를 생산하는 발전소에서의 에너지 손실 또는 송전을 할 때의 에너지 손실 들이 있지만 그것 또한 엔진의 손실에 비하면 매우 낮은 수준입니다.
- 일단 전기차의 에너지 효율은 매우 높은 상태이고 앞으로 점점 더 개선해야 할 문제는 **배터리의 에너지 밀도나 충전 시스템**입니다. 즉 에너지를 저장하는 것이 개선해야 할 문제인 것이죠.

<br>
<center><img src="../assets/img/autodrive/automotive/ev/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 모터는 인휠 모터라고 하면 자동차의 휠 속에 모터를 장치해서 직접 타이어를 구동하기 때문에 전기 자동차 중에서도 특히 에너지 효율이 높다고 알려져 있습니다.

<br>

## **보통의 전기 자동차는 변속기와 클러치가 없다.**

<br>

- 엔진에는 효율적인 회전수라는 것이 있습니다. 이 효율적인 회전수를 위해 자동차에서는 상황에 맞춰 최적의 회전수를 얻기위해 변속기를 사용합니다.
- 예를 들어 정지 상태에서 발진을 할 떄는 적은 회전수로도 커다란 토크를 얻을 수 있는 기어를 사용하지만, 고속 주행을 할 떄는 관성의 힘이 작용하므로 엔진의 회전을 억제해 매끄러운 주행과 연비의 향상을 할 수 있는 감속비가 적은 기어를 사용합니다.
- 또한 후진할 때는 타이어에 전달되는 회전 방향을 정반대로 만들 필요가 있는데, 이 역회전 또한 기어를 이용하여 만들어냅니다.
- 반면 모터는 정지 상태에서 자력의 흡인과 반발의 힘을 가장 크게 발휘하기 때문에 발진할 떄부터 강력하게 가속할 수 있습니다.
- 또한 모터의 전력 소비는 주로 부하 크기의 영향을 받으므로 회전수가 상승해도 전력 소비는 그다지 증가하지 않습니다. 발진할 떄는 전력 소비가 증가하지만 그 후에는 속도를 높여도 전력 소비가 증가하지 않는 것입니다.
- 이같은 이유로 **전기 자동차에는 대개 변속기를 탑재하지 않습니다.** 특수 목적으로 변속기를 사용하는 경우도 있지만 중량 절감이나 에너지 전달 효율의 측면에서 변속기를 탑재하지 않는 것이 일반적입니다.
- 변속기가 없으므로 클러치 또한 없습니다. 클러치는 회전 중인 엔진으로부터 변속기로 동력을 전달하거나 차단하는 기구입니다. 엔진도 없고 변속기도 없으니 클러치는 당연히 필요 없습니다.
- 모터의 사용으로 에너지 효율도 올라가고 변속기와 클러치도 빠져서 중량도 절감할 수 있습니다. (물론 배터리의 무게로 인하여 전기차의 무게가 더 나가긴 합니다.)

<br>

## **회생 제동 장치**

<br>

- 모터가 전력을 구동력으로 변환하면 발전기는 구동력을 전력으로 변환하는 기계입니다. 
- 전기자동차의 모터는 감속할 때 발전기로 사용합니다. 특히 발전 중에는 저항으로 작용되어 감속이 더 잘됩니다. 마치 가솔린 자동차의 엔진 브레이크와 같은 역할도 하게 됩니다.
- 물론 전기자동차 뿐만 아니라 하이브리드 자동차에도 회생 제동 장치를 이용한 충전을 하여 연비를 높입니다.
- 회생 제동 장치를 통항 가솔린 자동차에서는 제동 과정 중 버려지던 열에너지를 회수 할 수 있으므로 상당히 효율이 좋아집니다.

<br>
<center><img src="../assets/img/autodrive/automotive/ev/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 회생 제동 장치는 달리는 기세를 이용하여 모터를 발전기로 돌리면서 그 저항을 제동력으로 삼습니다. 이 때 만든 전기는 배터리나 다른 장치에 공급합니다.

<br>

## **전기 자동차의 배터리**

<br>

- 전기 자동차나 하이브리드 자동차에는 2종류의 배터리가 적용됩니다.
- 일반적인 모든 차에 들어가서 시동을 켜는 목적의 납산 배터리와 주행에 사용되는 리튬 이온 배터리입니다.
- 납산 배터리는 말 그대로 납과 묽은 황산을 반응시켜서 사용하는 배터리입니다. 이 배터리는 시동시에만 사용합니다. 가격은 낮을지라도 무게 및 용적에 비해 효율이 낮기 때문입니다.
- 주행에 사용되는 배터리는 리튬 이온 배터리로 고성능 동작이 필요한 스마트폰, 노트북 등에도 들어가는 배터리입니다.
- 리튬 이온을 배터리 용액 속에 녹이고 이것을 양극과 음극 사이로 이동시켜 전기 흐름을 만듭니다.
- 이온을 주고 받는 전해질이 액체 상태이면 액체가 셀 우려가 있기 때문에 안정성을 높이기 위하여 배터리 용액을 겔 형태로 만들기도 합니다.
- 리튬 이온 배터리가 고성능이라고 할지라도 셀 한개로는 전압이 낮아서 힘이 부족합니다.
- 그래서 여러 셀을 직렬로 연결하여 고전압의 팩 이라는 단위로 만들고 이 팩들을 병렬로 연결하여 사용합니다.
- 각각의 셀들은 성능의 편차가 있기 때문에 사용하다보면 충전이 불가능한 셀이 발생하곤 합니다.
- 따라서 몇 개의 셀이 사용 불가가 되더라도 전압이 떨어지지 않도록 배터리 조합에 여유를 두고 설계를 합니다.

<br>

- 리튬 이온 배터리는 저온에서 효율이 떨어지는 특성이 있습니다. 이를 개선하기 위해 전기 자동차에는 배터리의 온도 저하를 방지하기 위한 배터리 워머가 있지만 그럼에도 불구하고 겨울철에는 연속 주행 거리가 짧아지는 경향이 있긴합니다.
- 특히 엔진 자동차 같은 경우 겨울철 히터를 엔진 열을 이용하여 만들어 내지만 전기 자동차 같은 경우는 난방에도 전력을 사용해야 하는 단점이 있습니다.

<br>

- 전기 자동차를 충전할 때 급속 충전을 하면 90% 까지 충전하는 데 빠르게 충전되지만 완전 충전 되지 않는 이유는 급속 충전을 위한 대전류로 100%까지 충전하면 배터리 셀이 손상될 위험이 있기 때문입니다.
- 전기 자동차는 발진이나 가속 등의 고부하 상황에서 전류를 단번에 방전하고, 충전량이 줄어들면 한꺼번에 충전하는 사례가 많습니다. 이 때문에 다량의 전기를 충전하고 방전하는 일을 반복하게 됩니다. 이렇게 되면 배터리 셀의 개체차가 서서히 커져서 전압차가 발생하기 때문에 이를 방지하기 위해 충전 종료 직전에 완전히 충전되지 않은 셀을 충전하는 조정을 실시합니다.
- 리튬 이온 배터리의 특성상 충전 초기에는 전기를 빨아들이 듯이 충전을 진행하지만 충전량이 일정 이상이 되면 전압이 거의 상승하지 않게 됩니다. 이 특성을 이용하여 과충전을 막습니다.

<br>

- 리튬 이온 배터리는 밀도가 높고 무겁기 때문에 차의 무게 중심을 낮추기 위해서 가급적 낮은 위치에 탑재하는 것이 일반적입니다. 타이어 높이와 비슷하게 바닥에 늘어 놓도록 설치하는 데 이렇게 하면 무게 중심이 낮아져서 주행 안정성이 좋아집니다.