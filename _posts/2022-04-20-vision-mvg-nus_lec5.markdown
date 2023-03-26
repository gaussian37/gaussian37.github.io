---
layout: post
title: (멀티플 뷰 지오메트리) Lecture 5. Camera models and calibration
date: 2022-04-20 00:00:05
img: vision/mvg/mvg.png
categories: [vision-mvg] 
tags: [멀티플 뷰 지오메트리, Multiple View Geometry, Robust homography estimation] # add tag
---

<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>

- 참조 : https://youtu.be/sae97GVUXcg?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/oyGCk4idsaU?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : https://youtu.be/77kpTQUfIBg?list=PLxg0CGqViygP47ERvqHw_v7FVnUovJeaz
- 참조 : Multiple View Geometry in Computer Vision

<br>

- 이번 강의에어슨 카메라 모델과 카메라 캘리브레이션에 관한 내용을 다루어 보도록 하겠습니다.

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/sae97GVUXcg" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/mvg/nus_lec5/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>





<br>

[Multiple View Geometry 글 목차](https://gaussian37.github.io/vision-mvg-table/)

<br>
