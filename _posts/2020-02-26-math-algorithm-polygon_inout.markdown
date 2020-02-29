---
layout: post
title: 점의 다각형의 내부 또는 외부 위치 확인
date: 2020-01-02 00:00:00
img: math/algorithm/algorithm.png
categories: [math-algorithm] 
tags: [algorithm, 다각형의 내부 외부] # add tag
---

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>
<center><img src="../assets/img/math/algorithm/polygon_inout/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같은 다각형이 있고 위 다각형의 내부에 점이 있습니다. 이 점들이 다각형의 내부에 있는 지 확인할 수 있는 방법에 대하여 다루어 보겠습니다.
- 이번 글을 이해하기 위해서는 [ccw](https://gaussian37.github.io/math-algorithm-ccw/)와 [선분의 교차](https://gaussian37.github.io/math-algorithm-line_intersection/)를 반드시 이해하고 와야 정확히 이해할 수 있습니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_inout/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 선분의 교차를 이해하셨다면 점의 다각형 내부 또는 외부 위치 확인 방법은 간단하게 이해하실 수 있습니다.
- 위 그림처럼 다각형의 외부에 임의의 점을 하나 찍은 다음에 다각형 내부의 어떤 점과 이은 선분을 하나 만듭니다.
- 그 다음 그 선분과 다각형의 선분들(위 그림에서는 8개의 선분이 있습니다.)이 교차하는 지 모두 살펴보고 교차하는 갯수가 홀수이면 다각형의 내부에 있고 짝수이면 (0 포함)다각형의 외부에 있다고 판단할 수 있습니다.
- 예를 들어 위 그림의 내부에 있는 점들은 교차점이 각각 1개가 있습니다.
- 이렇게 되는 이유는 다각형 밖의 임의의 점에서 교차를 할 때 마다 외부 → 내부로 바뀌게 되고 다시 한번 더 교차하면 내부 → 외부로 바뀌게 됩니다.
- 시작점이 다각형의 외부였기 때문에, 교차 점의 갯수가 홀수가 되면 탐색하는 점은 다각형의 내부에 있다고 말할 수 있습니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_inout/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이 때, 중요한 것은 어느 위치에 외부의 점을 정하느냐 입니다.
- 위 그림처럼 애매한 위치에 점을 찍게 되면 (특히 다각형의 선분과 collinear 하거나 다각형의 꼭지점과 겹치는 경우) 교차 점의 갯수를 셀 때 잘못 계산될 수 있습니다.
- 따라서 **다각형 외부의 점과 연결한 선분이 다각형의 변과 일치하는 경우가 없도록 외부에 점을 정해야** 합니다.

<br>
<center><img src="../assets/img/math/algorithm/polygon_inout/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그 다음으로 고려해야 할 상황은 점이 다각형의 선분 위에 있는 경우 입니다.
- 다각형의 선분에 걸쳐 있는 점 또한 다각형의 내부로 봐야 한다는 가정하에 설명하면 위와 같은 경우에 교차점은 2개이므로 오검출을 하게 됩니다.
- 따라서 앞에서 설명한 과정을 하기 전에 점이 선분 위에 있는 지 먼저 살펴보고 선분 위에 있다면 다각형의 내부로 판단하면 됩니다.


<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>
