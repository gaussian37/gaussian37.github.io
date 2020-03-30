---
layout: post
title: 작은 이미지가 큰 이미지의 중심에 오도록 적용
date: 2020-03-29 00:00:00
img: vision/opencv/opencv-python.png
categories: [vision-opencv] 
tags: [opencv, 이미지, 중앙 정렬] # add tag
---

<br>

- 작은 이미지가 큰 이미지의 중심에 오도록 적용 하는 방법에 대하여 알아보도록 하겠습니다.
- 이 방법을 적용하면 큰 이미지안에 작은 이미지를 넣을 수도 있고 큰 배경 안에 어떤 이미지를 중앙 정렬하여 넣을 수 있습니다.

<br>
<iframe height="800px" width="100%" src="https://repl.it/@gaussian37/CenteringImage?lite=true" scrolling="no" frameborder="no" allowtransparency="true" allowfullscreen="true" sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts allow-modals"></iframe>
<br>


- 위 코드의 `CenteringImage` 함수를 이용하면 첫 번째 인자로 들어가는 큰 이미지에 두번째 인자인 작은 이미지를 중앙 정렬하여 삽입합니다.

<br>
<center><img src="../assets/img/vision/opencv/centering_image/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>
