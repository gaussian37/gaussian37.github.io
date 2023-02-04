---
layout: post
title: RegNet, Designing Network Design Spaces
date: 2021-07-19 00:00:00
img: dl/concept/regnet/0.png
categories: [dl-concept]
tags: [deep learning, regnet, tesla, designing network design space] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 논문 : https://arxiv.org/abs/2003.13678
- 참조 : https://2-chae.github.io/category/2.papers/31
- 참조 : https://jackyoon5737.tistory.com/245
- 참조 : https://cocopambag.tistory.com/47
- 참조 : https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249
- 참조 : https://github.com/signatrix/regnet
- 참조 : https://www.youtube.com/watch?v=bnbKQRae_u4

<br>

- 이번 글에서는 `Designing Network Design Spaces` 이라는 논문에서 다루는 `RegNet`에 대하여 살펴보도록 하겠습니다.
- `RegNet`은 `Regular Network`의 줄임말로 논문을 낸 곳인 Facebook에서 **일반적인 용도로 사용**할 수 있는 `backbone`의 의미를 가지도록 이름을 명명한 것으로 보이며 컴퓨터 비전에서 사용 중인 대표적인 `backbone`인 `EfficientNet`과 유사한 성향을 가집니다.
- 두 네트워크는 모두 `width (layer의 channel 수)`, `depth (layer의 깊이 수)`, `resolution (입력 해상도)`에 따른 네트워크 변경의 자유도를 가지므로 리소스를 고려하여 3가지 항목 + α 의 변경을 통해 네트워크의 크기를 쉽게 조절할 수 있습니다.
- 컴퓨터 비전 분야에서는 테슬라가 `RegNet`을 이용하여 `Perception`을 구현한 것을 공개 하였기 때문에 집중되기도 하였습니다. 관련 내용은 아래 링크에서 참조하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/autodrive-concept-tesla_ai_day/](https://gaussian37.github.io/autodrive-concept-tesla_ai_day/)

<br>




 
<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
