---
layout: post
title: 이미지들 중에서 이미지 선택하기 
date: 2020-03-19 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [opencv, 이미지, 이미지 선택] # add tag
---

<br>

- 이번 글의 응용 사례는 다음과 같은 상황입니다.
- 현황 : N개의 이미지가 있는 상태
- 필요 사항 : N개의 이미지에 다양한 이미지 프로세싱을 적용하여 변형을 하였을 때, 각 이미지마다 어떤 결과가 좋은 지 정성적으로 선택이 필요한 경우 GUI 상에서 클릭해서 선택할 수 있어야 합니다.

<br>

- 예를 들어 다음과 같이 바다, 산, 도시 사진이 있다고 가정해 보겠습니다.

<br>
<center><img src="../assets/img/vision/opencv/images_selection/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 각 사진에 어떤 영상 처리를 해주어서 사진이 조금 변형되었을 때, 어떤 사진이 좋은 지 선택 하려고 합니다.
- 예를 들어 바다 사진을 다음과 같이 5장으로 변형해 보겠습니다.

<br>
<center><img src="../assets/img/vision/opencv/images_selection/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 사진 중 어떤 사진이 좋은 지 클릭을 하여 기록해 놓고 싶을 수 있습니다.
- 이 요구사항을 반영하여 어플리케이션을 한번 만들어 보겠습니다.

<br>

## **Input 데이터 준비**

<br>
<center><img src="../assets/img/vision/opencv/images_selection/3.png" alt="Drawing" style="width: 300;"/></center>
<br>

- 위에서 다룬 5가지 이미지 프로세싱 처리한 결과를 각 폴더에 따로 저장해 보겠습니다.
- 예를 들어 image1 폴더는 1번 프로세싱, image2 폴더는 2번 프로세싱, ... 이렇게 처리한 결과를 각 폴더에 저장해 놓습니다.

<br>

## **실행 방법**

<br>






