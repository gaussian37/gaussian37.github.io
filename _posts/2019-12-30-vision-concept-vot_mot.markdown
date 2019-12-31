---
layout: post
title: VOT(Visual Object Tracking)와 MOT(Multiple Object Tracking)
date: 2019-12-30 00:00:00
img: vision/concept/vot_mot/0.jpg
categories: [vision-concept] 
tags: [vision, vot, mot, tracking] # add tag
---

<br>

- 이번 글에서는 Tracking의 개념에 대하여 간략하게 다루어 볼 예정입니다.
- 특히 Tracking의 두 종류인 **Visual Object Tracking**과 **Multiple Object Tracking**이 무엇인지 다루어 보겠습니다.

<br>

## **목차**

<br>

- ### VOT(Visual Object Tracking) 이란
- ### VOT의 예
- ### MOT(Multiple Object Tracking) 이란
- ### MOT의 예

<br>

## **Visual Object Tracking 이란**

<br>

## **Visual Object Tracking의 예**

<br>

## **MOT(Multiple Object Tracking) 이란**

<br>
<center><img src="../assets/img/vision/concept/vot_mot/0.jpg" alt="Drawing" style="width: 600px;"/></center>
<br> 

<br>

- 이번에는 MOT(Multiple Object Tracking)에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/vot_mot/1.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- 앞에서 다룬 VOT와 비교하여 MOT는 어떻게 다른지 알아보겠습니다.
- MOT는 말 그대로 여러개의 객체를 트래킹 하는 것을 말합니다.
- MOT의 목적은 크게 2가지가 있습니다. 첫번째로 `여러 객체를 동시에 처리`할 수 있어야 한다는 것과 두번째로 단기간의 시간이 아닌 `장기간(long-term) 트래킹`이 가능해야 한다는 것입니다.
- 물론 트래킹을 하기 위해서는 센서값인 디텍션 좌표가 필요합니다.
- 디텍션을 이용한 트래킹에는 크게 2가지 방법이 있습니다. 첫번째가 DBT(Detection Based Tracking)이고 두번째는 DFT(Detection Free Tracking) 입니다.
- DBT는 일반적으로 감지해야할 객체가 정의가 되어 있고 그 객체에 대한 좌표 값을 매 프레임 마다 얻는 것을 말합니다.
- 반면 DFT는 시작 프레임의 좌표 또는 바운딩 박스의 좌표를 가지고 객체를 트래킹 하는 방법을 말합니다. 즉, 시작 프레임에서 특정 객체에 바운딩 박스를 주고 그 객체만을 계속 트래킹 하는 것으로 이해할 수 있습니다.
- 이번 글에서는 `DBT`를 기준으로 설명해 보겠습니다.
- 즉, 트래킹 해야 할 대상은 영상 속에서 디텍션 알고리즘에 의해 검출된 좌표 또는 바운딩 박스들이 됩니다.

<br>
<center><img src="../assets/img/vision/concept/vot_mot/2.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- 트래킹에서 발생하는 에러에는 대표적으로 2가지가 있습니다. 첫번째로 `ID Switch` 문제가 있고 두번째로 `Fragmentation` 문제가 있습니다.
- `ID Switch`문제는 위 그림과 같이 Ground Truth 하나에 2개의 trajectory(자취)가 생기는 것을 말합니다.
    - 여기서 Ground Truth는 검은색 점들의 trajectory로 정답에 해당합니다.
    - 처음에 빨간색 trajectory가 GT에 근사하게 표시되다가 파란색 trajectory로 바뀌는 오류를 범하였습니다.
    - 이것은 트래킹 알고리즘에 새로운 객체로 인식해서 새로운 trajectory로 형성한 것입니다. 그래서 객체의 ID가 변화는 `ID Switch` 문제가 발생한 것입니다.
- 두번째는 `Fragmentation` 문제입니다.
    - 이 오류는 센서값인 디텍션 정보가 중간에 끊어졌기 때문에 발생하는 것입니다. 즉, 실제는 있어야 할 trajectory가 일정 시간 형성되지 않아 False Negative 가 발생한 것입니다.
    - 트래킹 알고리즘이 fragmentation 보완을 잘 해줘서 trajectory가 Ground Truth를 잘 쫓아가도록 해주어야 합니다. 그렇지 않으면 위의 오른쪽 그림과 같이 trajectory가 끊겼다가 새로 시작하는 지점에서 새로운 객체로 인지하여 또다시 `ID Switch`가 발생하게 됩니다.  



<br>

## **Multiple Object Tracking의 예**

<br>
