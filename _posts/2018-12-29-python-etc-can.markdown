---
layout: post
title: python으로 CAN 데이터 읽기 
date: 2018-12-19 13:46:00
img: python/etc/can/can.jpg
categories: [python-etc] 
tags: [python, can, mdf] # add tag
---

<br>

- 이번 글에서는 python을 이용하여 CAN 데이터를 저장하는 방법에 대하여 간략하게 알아보겠습니다.
- 정확히는 `MDF(Measurement Data Format)`로 저장된 데이터를 읽는 방법이라고 할 수 있습니다.
    - 참고로 `MDF`는 벡터와 보쉬가 자동차 산업에서 사용하기 위해 만든 계측 데이터 용도의 바이너리 파일입니다.
    - 참조 : https://www.vector.com/kr/ko/products/application-areas/ecu-calibration/measurement/mdf/
- 일반적으로 많이 사용하는 `Vector` 회사의 장비를 이용하여 CAN 데이터를 수집한다면 `MDF`파일로 CAN 값을 저장할 수 있습니다.
    - CAN Logger 장비(GL 시리즈), CANoe, CANalyzer 등은 모두 지원합니다.
- MDF 파일로 저장할 때, 반드시 `CAN DB`를 CAN 데이터와 연동하여 저장해야 파이썬에서 읽을 때 CAN에 대한 설명값을 얻을 수 있습니다.
    - CAN DB를 연동하는 것은 벡터 소프트웨어에서 `MDF` 형식으로 저장할 때 할 수 있습니다.  

<br>

## **필요한 라이브러리 설치**

- `pip install twisted`
- `pip install asammdf`

<br>

- 참고로 `ASAM`은 ASAM (Association for Standardisation of Automation-and Measuring System)의 약자로 차량용 제어기 개발에 대한 시간과 비용을 줄이기 위해 설립된 단체입니다.

<br>

## **CAN 데이터 읽고 실행하는 방법**

- [실습 예제 파일](https://drive.google.com/open?id=1kbmElexO_jwdm60WXp_lOsUQDTsTWXvz)
    - 벡터 사에서 제공하는 데모 소프트웨어 파일로 벡터사 홈페이지에서도 받을 수 있습니다.
- MDF 파일을 불러오는 방법은 아래 코드와 같습니다. `asammdf`의 MDF를 이용하여 파일을 불러올 수 있습니다.
- MDF 파일 내의 CAN 신호 리스트는 `.channels_db`에서 확인 가능합니다. 

<img src="../assets/img/python/etc/can/1.png" alt="Drawing" style="width: 600px;"/>

<br>

- CAN 신호는 dictionary 형태로 저장되어 있으므로 (key, value)로 값 접근이 가능합니다.
- `.channels_db`에서 확인한 값을 Key 값으로 하면 각 신호의 값을 알 수 있습니다.
- `t`를 Key 값으로 하면 신호가 기록된 시간을 알 수 있습니다.
- 따라서 `t`에 해당하는 신호값을 찾으면 샘플링 시간에 기록된 신호값을 확인할 수 있습니다.
- 아래 코드는 x축을 시간, y축을 속도로 나타낸 차속 그래프 입니다.

<img src="../assets/img/python/etc/can/2.png" alt="Drawing" style="width: 600px;"/>

<br>

- CAN 신호값을 파이썬으로 불러왔기 때문에 엑셀로 저장도 가능합니다.
- 아래 코드는 Pandas를 이용하여 처리하기 좋은 엑셀로 저장하는 코드 입니다.

<img src="../assets/img/python/etc/can/3.png" alt="Drawing" style="width: 600px;"/>
