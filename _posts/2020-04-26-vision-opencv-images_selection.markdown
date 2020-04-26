---
layout: post
title: 이미지들 중에서 이미지 선택하기 
date: 2020-03-19 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [opencv, 이미지, 이미지 선택] # add tag
---

<br>

## **목차**

<br>

- ### 어플리케이션 소개
- ### Input 데이터 준비
- ### 실행 방법
- ### 동작 방법
- ### 출력 결과


<br>

## **어플리케이션 소개**

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
<center><img src="../assets/img/vision/opencv/images_selection/3.png" alt="Drawing" style="width: 200;"/></center>
<br>

- 위에서 다룬 5가지 이미지 프로세싱 처리한 결과를 각 폴더에 따로 저장해 보겠습니다.
- 예를 들어 image1 폴더는 1번 프로세싱, image2 폴더는 2번 프로세싱, ... 이렇게 처리한 결과를 각 폴더에 저장해 놓습니다.

<br>

## **실행 방법**

<br>

- 입력은 다음 3개를 받습니다.
- `--path` : 각 폴더들이 저장된 경로를 받습니다. 위 tree 구조에서 images에 해당하는 경로를 입력하면 됩니다.
- `--row` : 이미지들을 한번에 표시할 때, 표시 할 행의 갯수를 나타냅니다.
- `--col` : 이미지들을 한번에 표시할 때, 표시 할 열의 갯수를 나타냅니다.

<br>
<center><img src="../assets/img/vision/opencv/images_selection/4.png" alt="Drawing" style="width: 300;"/></center>
<br>

- 예를 들어 위 실행 결과의 경우 `row = 2`, `col = 3`의 옵션을 주어서 실행한 결과 입니다.

<br>

## **동작 방법**

<br>

- 아래 그림과 같이 어떤 영역에 왼쪽 마우스 버튼을 클릭하면 선택된 그림의 테두리가 빨간색 경계선이 만들어 집니다.

<br>
<center><img src="../assets/img/vision/opencv/images_selection/4.png" alt="Drawing" style="width: 300;"/></center>
<br>

- 만약 취소 하고 싶으면 다시 그 영역을 클릭하면 테두리가 사라집니다.
- 선택이 끝나면 키보드의 `n`을 눌러서 다음 그림으로 넘어가면 됩니다.
- 중간에 끝내고 싶으면 `esc`를 누르면 됩니다.
- 선택된 결과는 실행 파일이 있는 위치에 `csv` 파일로 저장되어 있습니다.

<br>

## **출력 결과**

<br>

- 출력 결과는 `csv` 파일에 저장되어 있고 출력 형식은 다음과 같습니다.

<br>

```
frame_number,selected_folder
1,images3,images4
2,images2,images3
3,images2
```

<br>

- 출력 결과는 `,`로 구분 되어 있고 첫 열은 Frame의 숫자이고 두번째 열부터는 각 프레임에서 선택된 폴더의 이름이 입력됩니다.

<br>

## **파이썬 코드**

<br>

```python


```


