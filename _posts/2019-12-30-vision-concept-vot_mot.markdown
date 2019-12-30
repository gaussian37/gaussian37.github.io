---
layout: post
title: VOT(Visual Object Tracking)와 MOT(Multiple Object Tracking)
date: 2019-12-30 00:00:00
img: vision/concept/vot_mot/1.png
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

- 이번에는 MOT(Multiple Object Tracking)에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/vot_mot/1.png" alt="Drawing" style="width: 600px;"/></center>
<br> 

- 앞에서 다룬 VOT와 비교하여 MOT는 어떻게 다른지 알아보겠습니다.
- MOT는 말 그대로 여러개의 객체를 트래킹 하는 것을 말합니다.
- MOT의 목적은 크게 2가지가 있습니다. 첫번째로 `여러 객체를 동시에 처리`할 수 있어야 한다는 것과 두번째로 단기간의 시간이 아닌 `장기간(long-term) 트래킹`이 가능해야 한다는 것입니다.
- 물론 트래킹을 하기 위해서는 센서값인 디텍션 좌표가 필요합니다.
- 디텍션을 이용한 트래킹에는 크게 2가지 방법이 있습니다. 첫번째가 DBT(Detection Based Tracking)과 DFT(Detection Free Tracking) 입니다.
- DBT는 일반적으로 감지해야할 객체가 정의가 되어 있고 그 객체에 대한 좌표 값을 매 프레임 마다 얻는 것을 말합니다.
- 반면 DFT는 시작 프레임의 좌표 또는 바운딩 박스의 좌표를 가지고 객체를 트래킹 하는 방법을 말합니다. 즉, 시작 프레임에서 특정 객체에 바운딩 박스를 주고 그 객체만을 계속 트래킹 하는 것으로 이해할 수 있습니다.
- 이번 글에서는 `DBT`를 기준으로 설명해 보겠습니다.







<br>

## **Multiple Object Tracking의 예**

<br>
