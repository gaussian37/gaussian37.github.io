---
layout: post
title: 스케일 불변 특징점 검출 (SIFT, SURF)
date: 2021-02-17 00:00:00
img: vision/concept/scale_invariant_feature_extraction/0.png
categories: [vision-concept] 
tags: [SIFR, SURF, scale invariant feature extraction] # add tag
---

<br>

- 참조 : 컴퓨터 비전 (오일석)
- 참조 : https://medium.com/data-breach/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40
- 참조 : https://ballentain.tistory.com/47
- 참조 : https://bskyvision.com/21
- 참조 : https://www.koreascience.or.kr/article/JAKO201310457144649.pdf
- 참조 : https://darkpgmr.tistory.com/131
- 참조 : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=laonple&logNo=220879911249

<br>

## **목차**

<br>

- ### 스케일 공간
- ### 해리스 라플라스 특징 검출
- ### SIFR 검출
- ### SURF 검출

<br>

## **스케일 공간**

<br>


<br>

## **해리스 라플라스 특징 검출**

<br>


<br>

## **SIFR 검출**

<br>

- 지금 부터 설명할 SIFT와 이후에 설명할 SURF는 `scale & rotation invariant`한 feature를 찾기 위한 대표적인 알고리즘입니다. `scale`과 `rotation`에 강건하기 때문에, 이미지의 크기 변화 또는 이미지의 위치 변화 또는 회전이 되었다고 하더라고 이미지의 feature를 찾기 용이합니다. 이 성질을 이용해서 두 이미지에서 같은 물체의 feature를 찾은 뒤 두 물체를 매칭하는 데 사용됩니다.

<br>
<center><img src="../assets/img/vision/concept/scale_invariant_feature_extraction/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 이미지에서는 같은 자동차를 다른 뷰에서 촬영한 2개의 사진에서 feature extraction을 한 것을 볼 수 있습니다. 이 때, 두 자동차가 같은 차 인 지 알기 위하여 feature간 매칭이 된 것을 참조할 수 있습니다.
- 위 사진을 통해 확인할 점은 두 이미지가 `scale`, `rotation` 모두가 변경이 되었지만 비슷한 부분의 feature를 잡아낼 수 있다는 점입니다.

<br>
<center><img src="../assets/img/vision/concept/scale_invariant_feature_extraction/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>




<br>

## **SURF 검출**

<br>


<br>


