---
layout: post
title: What is a marginal probability
date: 2019-08-04 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [베이지안, bayes, 확률, marginal probability, 주변 확률] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/r27mouuyFQk" frameborder="0" allowfullscreen="true" width="600px" height="400px"> </iframe>
</div>
<br>

- 이번 글에서는 `marginal probability`에 대하여 알아보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/pb/bayes_stat_06/1.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이번 글의 주제인 **marginal probability**는 확률 변수가 한 개가 아니라 다변수 일 때 나오는 개념입니다.
- 위 슬라이드 처럼 $$ X $$ 라는 확률 변수와 $$ Y $$ 라는 확률 변수가 각각 존재한다면, 확률을 접근할 때 다음과 같이 접근할 수 있습니다.
- 먼저 $$ X , Y $$  값이 각각 정해진 경우, 이 때에는 $$ X = 0, Y = 0 $$과 같이 특정 케이스로 한정이 됩니다.
- 이런 확률을 `Joint probability`라고 하는데 확률 변수 들이 결합되어서 확률을 만들어 내기 때문입니다. 결합 확률이라고도 합니다.
- 이 경우에는 격자에서 특정 영역 한 칸을 나타냅니다.

<br>

- 반면에 고정된 확률 변수의 갯수가 전체 확률 갯수 보다 작으면 어떻게 될까요?
- 예를 들어 $$ X = 0 $$으로 정해진 반면에 $$ Y $$값은 정해지지 않아서 0 또는 1이 될 수 있다고 가정하겠습니다.
- 그러면 $$ X $$는 0인 모든 경우의 확률이 됩니다. 즉, 특정 영역 한 칸의 확률이 아니라 범위가 됩니다.
- 위 슬라이드 기준으로 $$ X = 0 $$일 때의, **marginal probability**는 0.5 + 0.1 = 0.6이 됩니다.

<br>

- 위 내용을 일반화 시키면 $$ P(X = x) = \sum_{y} P(X = x, Y = y) $$가 됩니다. 고정된 변수를 제외하고 나머지 변수는 모두 선택되어 합해지는 구조입니다.
- 다시 한번 말하면 다변수 확률 변수에서 특정 값으로 고정된 변수와 그렇지 않은 변수가 있을 때, **marginal probability**를 정의할 수 있습니다.  