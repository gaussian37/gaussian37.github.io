---
layout: post
title: AI for Full-Self Driving at Tesla
date: 2020-08-18 00:00:00
img: autodrive/concept/ai_for_full_self_driving_at_tesla/0.png
categories: [autodrive-concept] 
tags: [테슬라, tesla, 자율주행, 자율주행 자동차, autodrive, self-driving] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/hx7BXih7zx8" frameborder="0" allowfullscreen="true" width="800px" height="800px"> </iframe>
</div>
<br>

- 이 글은 안드레이 카파시가 [scaledml2020](http://scaledml.org/2020/)에서 테슬라의 자율주행에 관해 발표한 내용을 정리한 것입니다.

<br>

## **목차**

<br>

- ### 테슬라의 오토파일럿이란?
- ### (라이다 방식이 아닌) 컴퓨터 비전 기반의 테슬라 방식
- ### 양산을 위한 뉴럴 네트워크
- ### fleet 으로 부터 까다로운 케이스에 해당하는 이미지 획득
- ### 테스트를 위해선 loss function과 accruracy 평균만으로는 부족함
- ### HydraNet (48 network, 1,000 prediction, 70,000 hours train)
- ### Full self-driving을 위한 neural network
- ### Self-supervised learning을 이용하여 이미지에서 depth를 예측하고 실제 거리를 측정하는 방법
- ### 다른 self-supervised learning의 사용 사례
- ### Q & A

<br>

