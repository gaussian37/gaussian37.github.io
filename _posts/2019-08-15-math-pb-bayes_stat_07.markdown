---
layout: post
title: What is a conditional probability
date: 2019-08-015 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [베이지안, bayes, 확률, conditional probability, 조건부 확률] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/_Y_xMTTmt-Q" frameborder="0" allowfullscreen="true" width="600px" height="800px"> </iframe>
</div>
<br>

- 이번 글에서는 `conditional probability` 즉, 조건부 확률에 대하여 다루어 보겠습니다.
- 슬라이드의 왼쪽 상단과 같이 두 개의 확률 변수 $$ X, Y $$ 가 있을 때, 확률이 표와 같이 구성되어 있다고 가정해 보겠습니다. 앞의 글 `marginal probability`에서와 같습니다.
- 이 때, 모든 확률 변수의 값이 고정이 되었을 때, 격자 한칸에 대한 확률을 정의할 수 있습니다. 이 경우 여러 확률 변수가 `AND` 조건으로 묶이게 되는데 이것을 `Joint probability`라고 하였습니다.
- 반면 `marginal probability`는 확률 변수의 값 중 고정되지 않은 값이 있어서 영역으로 표시되는 구간을 뜻하였습니다.
- 예를 들어 $$ P(X = 1) = 0.1 + 0.3 = 0.4 $$ 처럼 일부 변수만 고정이 되어 위 표 기준으로는 한 개의 열 또는 행의 확률이었습니다.
- 이번 글에서 다루는 `conditional probability`는 간단하게 말하면 `joint probability / marginal probability` 라고 말할 수 있습니다.

<br>

- 위 슬라이드를 보면 $$ P(X = 1 \vert Y = 1) = \frac{0.3}{0.1 + 0.3} $$과 $$ P(X = 0 \vert Y = 1) = \frac{0.1}{0.1 + 0.3} $$ 이 설명되어 있습니다.
- 이 뜻은 $$ Y = 1 $$ 일 때, $$ X = 1 $$ 또는 $$ X = 0 $$이라는 뜻을 의미합니다.
- 여기서 분모의 영역을 모든 영역이 아닌 일부 영역으로 한정지었는데, 이것을 `conditional`이라고 합니다. 즉 전체 영역인 분모가 달라집니다.
    - 이 때, 분모의 영역을 보면 앞에서 본 `marginal probability`에 해당하는 영역입니다. 한 행의 영역이라고 할 수 있습니다.
- 분자를 보면 확률 변수 모두가 고정되어 있기 때문에 `joint probability`가 됩니다.
- 다시 말하면 `conditional probability`는 표 전체 영역이었던 확률 공간을 조건을 주어 일부 영역으로 줄여습니다. 여기서는 2개의 확률 변수(2차원)가 있으니 영역이 줄어 1차원인 행 또는 열이된 셈입니다.
    

