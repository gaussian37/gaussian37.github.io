---
layout: post
title: FMEA(Failure Mode Effeact Analysis)에 대한 이해
date: 2020-09-25 00:00:00
img: etc/phm/0.png
categories: [etc-etc] 
tags: [phm, prognostics, health management, FMEA, FMECA] # add tag
---

<br>

[PHM 관련 글 목록](https://gaussian37.github.io/etc-phm-table/)

<br>

- 참조 : http://blog.naver.com/gics17/221719761513
- 참조 : https://ridibooks.com/books/2709000117?_s=search&_q=%EC%8B%A0%EB%A2%B0%EC%84%B1+%EA%B3%B5%ED%95%99

<br>

## **목차**

<br>

- ### [고장(failure)이란?](#고장(failure)이란?-1)

<br>

## **고장(failure)이란?**

<br>


- 이번 글에서는 `FMEA(Failure Mode Effects Analysis)`에 대하여 알아보도록 하겠습니다.
- 먼저 FMEA는 신뢰성 공학에서 나온 개념이며 신뢰성 공학은 **시스템의 고장과 관련된 학문**입니다. 따라서 FMEA에 접근하기 이전에 `고장(Failure)`와 관련된 용어들에 대하여 알아보도록 하겠습니다.

<br>

- `고장물리(Physics of Failure)` : 잠재적인 고장 메커니즘, 고장부위 및 고장모드의 확인과 관련된 제조와 강건설계를 통해 고장방지를 위해 **근본적인 고장발생과정(root-cause failure process)의 지식을 이용**하여 설계하고, 신뢰성 평가, 시험, 설계 마진을 정하는 접근 방법을 뜻합니다.
- 고장은 내부의 잠재적인 고장원인과 외부의 스트레스가 복합적으로 결합하여 발생하며 대표적으로 제품에 인가되는 스트레스, 재료, 구조, 형상등이 부적합할 때 발생합니다.
- 고장은 `파국 고장(catastrophic failure)`과 `열화 고장(degration failure)`으로 분류할 수 있습니다. 파국고장은 이론적으로 견딜 수 있는 외부 충격 
스펙 이상으로 스트레스가 가해져서 고장난 경우를 뜻하며 열화 고장은 제품의 성능이 시간에 따라 점진적으로 저하되어 발생하는 고장을 말합니다.

<br>
<center><img src="../assets/img/etc/phm/fmea/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 파국 고장, 열화 고장 모두에 가해지는 스트레스의 대표적인 종류는 위 도표와 같습니다. 위 스트레스가 서서히 시간이 가해져서 고장이 발생하면 열화 고장이고 순간적으로 강한 스트레스가 가해져서 고장이 나면 파국 고장이라고 이해하시면 됩니다.

<br>

- 그러면 고장이 발생하였을 때, 어떤 스트레스가 가해졌는 지 분석을 해야 합니다. 이 때, 고장의 분석절차는 다음과 같습니다.
    - ① `사전 조사` : 고장품 종류, 제조시기, 사용조건, 고장발생 상황등을 조사합니다.
    - ② `외관 관찰` : 광학현미경이나 Hi-Scope를 이용하여 외관 및 형상을 관찰합니다.
    - ③ `비파괴분석` : 커브 트레이서, LSI 테스터 등을 이용하여 전기적 특성을 측정하고 X-ray를 이용하여 내부 관찰을 합니다.
    - ④ `반파괴분석` : 고품에 화학 처리등을 통하여 관찰할 수 있는 분석을 하며 물리적인 분해를 하는 파괴분석 이전에 할 수 있는 분석을 합니다.
    - ⑤ `파괴분석` : 물리적인 분해를 통해 단면 절단, 연마, 제거 등을 통해 고장이 의심되는 곳을 분해하여 분석합니다.
    - ⑥ `고장 메커니즘 규명` : 어떤 스트레스가 어떤 과정을 거쳐 고장 부위에 특정 고장모드를 발생시키는 지 규명하는 단계입니다. 고장 모드는 고장 메커니즘의 결과이고, 고장 모드와 스트레스의 상관관계를 밝히는 것이 고장 분석입니다.
    - ⑦ 재현 실험
    - ⑧ 시정 조치

<br>

- 고장 메커니즘은 크게 우발 고장(overstress failure)과 마모 고장(wearout failure)으로 나뉩니다. 아래 도표를 참조하시기 바랍니다.

<br>
<center><img src="../assets/img/etc/phm/fmea/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 분류를 이용하여 고장 분석을 하면 카테고리화 하여 분석을 할 수 있습니다.



<br>

[PHM 관련 글 목록](https://gaussian37.github.io/etc-phm-table/)

<br>