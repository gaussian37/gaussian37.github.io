---
layout: post
title: 선분의 교차 여부 확인
date: 2020-01-02 00:00:00
img: math/algorithm/algorithm.png
categories: [math-algorithm] 
tags: [algorithm, 선분의 교차] # add tag
---

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

- 이번 글에서는 선분의 교차 여부를 확인하는 방법에 대하여 알아보도록 하겠습니다.
- 두 선분의 교차 여부를 확인할 때 사용하는 알고리즘은 [CCW(Counter ClockWise)](https://gaussian37.github.io/math-algorithm-ccw/)입니다. 이 알고리즘을 기반으로 설명해 보도록 하겠습니다.

<br>

- 먼저 한 선분을 나타낼 때 필요한 것은 두 점입니다. 따라서 두 선분의 교차 여부를 확인하기 위해서는 두 선분이 필요하고 각 선분당 두 점이 필요하므로 총 4개의 점이 필요 합니다.

<br>
<center><img src="../assets/img/math/algorithm/line_intersection/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위와 같이 2개의 선분이 있고 선분이 교차할 때 위와 같은 방식으로 교차될 수 있습니다.
- 여기서 $$ p_{1} , p_{2} $$를 이은 선분을 $$ p_{1} \to p_{2} $$ 방향으로의 벡터 $$ v_{1} $$ 이라고 하면 $$ v_{1} = p_{2} - p_{1} $$이 됩니다.
- 같은 원리로 $$ v_{2} = p_{3} - p_{1} $$ 이라고 하고 $$ v_{3} = p_{4} - p_{1} $$ 이라고 하겠습니다. 이 두 벡터는 위 그림의 선분은 아니나 방향을 확인하기 위하여 도입하였습니다..
- 이 때, 확인해 볼 것은 $$ v_{1}, v_{2} $$의 방향과 $$ v_{1}, v_{3} $$의 방향 입니다.

<br>
<center><img src="../assets/img/math/algorithm/line_intersection/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림에서 $$ v_{1} $$은 $$ v_{2} $$의 시계 방향에 있으므로 ccw의 결과는 음수가 나옵니다.
- 반면 $$ v_{1} $$은 $$ v_{3} $$의 반시계 방향에 있으므로 ccw의 결과는 양수가 나옵니다.
- 위와 같은 케이스의 선분의 교차에서는 위 방법으로 벡터 쌍들의 ccw를 구하고 그 방향이 다르면 선분이 교차한다고 말할 수 있습니다.
- 위 케이스는 가장 일반적인 경우이고 이제 한계 상황들에 대하여 다루어 보도록 하겠습니다.






<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>
