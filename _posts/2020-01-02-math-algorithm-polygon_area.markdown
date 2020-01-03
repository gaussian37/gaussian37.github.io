---
layout: post
title: n차원 다각형의 넓이 계산
date: 2020-01-02 00:00:00
img: math/algorithm/polygon_area/0.png
categories: [math-algorithm] 
tags: [algorithm, 알고리즘, 다각형 넓이] # add tag
---

<br>

- 출처 : 
- https://www.mathopenref.com/coordpolygonarea2.html
- http://mathworld.wolfram.com/PolygonArea.html

<br>

- 이번 글에서는 n각형의 볼록 또는 오목 다각형의 좌표를 모두 알고 있을 때, **n각형의 넓이 계산**을 하는 방법에 대하여 알아보도록 하겠습니다.
- 이 내용은 중, 고등학교 과정에서 한번씩은 사용해 보았을 방법인데, 7차 교육 과정을 겪은 저 기준으로 교육 과정에는 없었지만 원리는 모른 체 편법으로 배웠었던 것 같습니다.
- 그러면 내용을 한번 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### n각형의 넓이 계산 방법
- ### 원리 이해
- ### 한계 상황
- ### python 코드
- ### c 코드

<br>

## **n각형의 넓이 계산 방법**

<br>

- 먼저 계산하는 방법 부터 알아보도록 하겠습니다.
- 예를 들어 $$ (x_{1}, y_{1}) , (x_{2}, y_{2}), ... , (x_{n}, y_{n}) $$의 n각형 꼭지점의 좌표가 있다고 한다면 넓이는 다음과 같습니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_area/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 여기서 $$ \vert M \vert $$ 는 `determinant`를 뜻하므로 풀어서 쓰면 다음과 같습니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_area/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이 식을 좀더 시각적으로 기억하기 좋게 표현하면 다음과 같습니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_area/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉, $$ (x_{1}, y_{1}) $$ 부터 $$ (x_{n}, y_{n}) $$ 까지 쓰고 마지막에 다시 한번 더 $$ (x_{1}, y_{1}) $$을 쓴 다음에, 오른쪽 아래로 대각선 성분끼리 곱한 것은 더하고 오른쪽 위로 대각선 성분끼리 곱한 것을 뺀 다음에 2로 나누면 면적이 됩니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_area/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서 면적은 128이 됩니다. 그러면 이것을 위 식에 대입해서 한번 구해보겠습니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_area/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위에서 설명한 방법대로 구하면 넓이를 구할 수 있음을 확인하였습니다.

<br>

## **원리 이해**

<br>


