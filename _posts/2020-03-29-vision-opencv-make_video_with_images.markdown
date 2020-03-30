---
layout: post
title: 이미지를 이용하여 비디오를 만들기
date: 2020-03-19 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [opencv, 이미지, 비디오] # add tag
---

<br>

- 이번 글에서는 opencv-python을 이용하여 이미지들을 비디오로 만드는 방법에 대하여 다루어 보겠습니다.

<br>

- 다음 코드는 위에서 다룬 이미지들을 비디오로 만드는 코드입니다.
- 입력 ① `path1`은 필수 `path2`는 선택 : 디렉토리 경로
    - 입력은 디렉토리 경로를 최소 1개 또는 2개를 입력 받고 각 디렉토리의 이미지들을 이름 오름차순으로 정렬된 순서로 합쳐서 프레임을 만듭니다.
    - 각 디렉토리의 이미지 갯수가 다르면 가장 작은 이미지를 갖고 있는 디렉토리 갯수를 기준으로 영상의 프레임을 만듭니다.
    - 디렉토리1의 이미지가 50개, 디렉토리2의 이미지가 40개이면 총 40개의 프레임만 만듭니다.
- 입력 ② `fps` : FPS를 받으며 기본 값은 30입니다.
- 입력 ③ `method` : 두 이미지를 concatenate 할 수 있도록 사이즈를 맞춘다.
    - method == -1 : 작은 사이즈에 맞추어 resize 한다.
    - method == 0 : 두 이미지 중 큰 height와 width를 선택하여 max_height와 max_width로 만든 두 공간을 만들고 그 공간에 중앙 정렬한다.
    - method == 1 : 큰 사이즈에 맞추어 resize 한다.
