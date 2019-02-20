---
layout: post
title: (베이즈 통계학 기초) 정보를 얻으면 확률이 바뀐다.
date: 2019-02-20 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [통계학, 베이지안] # add tag
---

+ 출처 : [세상에서 가장 쉬운 베이즈 통계학 입문](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=103947200)

+ 베이즈 이론에 대하여 간략하게 알아보기 위하여 손님이 쇼핑족인지 아이쇼핑족인지 분류하는 예제를 다루어 보도록 하겠습니다.

<br>

### 1단계. 경험에서 사전확률을 설정합니다.

+ 추측을 위해 가장 먼저 해야 할 일은 손님의 두 가지 타입 - 쇼핑족과 아이쇼핑족에 대해 그 `비율`이 각각 몇인지 수치로 배정하는 것입니다.
  + 이러한 타입을 클래스 라고도 하며 이런 타입의 확률(비율)을 `사전확률`이라고 합니다.
  + `사전`이란 어떤 정보가 들어오기 전을 뜻하는 말이고 이때 `정보`란 손님이 말을 거는 행동을 했다 와 같은 **추가적인 상황**을 뜻합니다.
+ 추가적인 상황을 통하여 타입에 대한 추측을 업데이트 하게 되는데, `사전확률`이라고 하면 추가정인 정보 즉, `관측`이 이루어지기 전의 상태 입니다.

<br>

+ 만약 경험에 의하여 쇼핑족의 비율이 20% 즉, 0.2임을 알고 있다고 가정해 보겠습니다. 그러면 아이쇼핑족은 0.8이 됩니다.
  + 이 비율이 `사전확률`에 해당하고 **타입에 대한 사전분포** 라고 합니다.

<img src="../assets/img/math/pb/bayes-basic/1.1.PNG" alt="Drawing" style="width: 400px;"/>

+ 큰 직사각형을 2개의 직사각형으로 분할하는데, 면적의 비율이 각각 0.2와 0.8이 되도록 분할합니다.
  + `면적`이 베이즈 확률을 다루는 데 중요한 역할을 합니다.
+ 여기서 면적을 0.1과 0.4 또는 1과 4등으로 쓰지 않고 0.2와 0.8로 사용한 이유는 **확률은 전부 더해서 1이 되도록 설정한다**는 성질 때문입니다.
  + 이것을 `정규화 조건`이라고 합니다.

<br>

### 2단계. 타입별로 말을 거는 행동을 하는 조건부 확률을 설정합니다.

+ 다음 단계로 쇼핑족에 속하는 손님과 아이쇼핑족에 속하는 손님이 각기 어느 정도의 확률로 점원에게 말거기 행동을 하는가를 설정해야 합니다.
  + 이 때 확률은 어떠한 경험, 실증, 실험에 기반한 수치가 필요합니다. 

<img src="../assets/img/math/pb/bayes-basic/1.2.PNG" alt="Drawing" style="width: 400px;"/>

+ 위 행동에 대한 조건부 확률은 계산이 간단해지도록 임의로 설정하였습니다.
+ 여기서 주의할 점은 표를 가로 방향으로 보면 `정규화 조건`이 충족됩니다.
  + 가로 방향은 특정 타입 즉, 클래스에 따라서 분류한 것으로 각 클래스별 발생 확률의 총합은 1이 됩니다.
+ 반면 세로 방향으로 보면 정규화 조건이 충족되지 않습니다.
  + 각기 다른 타입의 사람에 대한 행동을 나타내고 있는 것으로 행동 전체를 아우르는 확률적 사건이 아니므로 더해도 1이 안됩니다.
+ 이 표는 `조건부 확률`을 나타냅니다. 즉, 타입을 한정한 경우 각 행동의 확률에 해당합니다.
+ 만약 `타입`을 행동의 `원인`으로 본다면 **원인을 알고 있을 때의 결과의 확률**이라고 해석할 수 있습니다.

<img src="../assets/img/math/pb/bayes-basic/1.3.PNG" alt="Drawing" style="width: 400px;"/>

+ 두가지 타입(쇼핑족과 아이쇼핑족)과 행동(말을 건다, 안건다)를 가지고 4가지 경우로 분리를 하면 위 표와 같습니다.
+ 각 구역에 나타나는 사항의 확률이 각 직사각형의 면적과 같습니다.

<img src="../assets/img/math/pb/bayes-basic/1.4.PNG" alt="Drawing" style="width: 400px;"/>

+ 면적을 구하는 방법은 위와 같고 면적의 총합은 1이 됩니다.

<br>

### 3단계. 관측한 행동에서 가능성이 사라진 세계를 제거합니다.

<img src="../assets/img/math/pb/bayes-basic/1.5.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic/1.6.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic/1.7.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic/1.8.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic/1.9.PNG" alt="Drawing" style="width: 400px;"/>

<img src="../assets/img/math/pb/bayes-basic/1.10.PNG" alt="Drawing" style="width: 400px;"/>

