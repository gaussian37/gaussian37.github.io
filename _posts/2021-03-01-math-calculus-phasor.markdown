---
layout: post
title: 페이저 (Phasor)
date: 2021-03-01 00:00:00
img: math/calculus/phasor/0.png
categories: [math-calculus] 
tags: [phasor, 페이저] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

- 참조 : https://angeloyeo.github.io/2019/06/18/phasor.html

<br.>

- 이번 글에서는 물리학 또는 신호 처리에서 사용되는 `Phasor`의 정의에 대하여 간략하게 알아보도록 하겠습니다.
- 먼저 `정현파(sinusoidal wave)`의 의미에 대하여 알아보겠습니다. 정현파는 **막대기의 회전 운동**을 통하여 유도할 수 있습니다. 다음 그림을 참조하면 쉽게 이해할 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/phasor/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- cos 함수의 궤적은 위 그림과 같이 어떤 선을 반시계 방향으로 각도 $$ \theta $$ 만큼 회전할 때, 위에서 바라본 그림자(빨간색 선) 길이의 자취로 생각할 수 있습니다.
- 반면 sin 함수의 궤적은 오른쪽에서 바라본 그림자 길이의 자취로 생각할 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/phasor/1.gif" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 cos 함수를 나타낸 것입니다. 막대의 움직임이 반시계 방향으로 점점 증가함에 따라서 위에서 바라본 그림자의 길이를 자취로 나타낸 형태입니다.
- 중요한 점은 sin, cos 모두 같은 형태 및 방향으로 움직이는 막대를 보는 방향에 따라서 다르게 값을 나타낸다는 점입니다. 즉, **보는 방향만 90도 만큼 이동하면 같은 관점**에서 바라보므로 같은 값을 가지게 됩니다. 이 점이 cos, sin 함수의 관계라고 말할 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/phasor/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 즉, 위 식과 같이 sin이나 cos 모두 **cos 함수로 환원**해서 생각할 수 있다는 점입니다. `phasor`의 중요한 전제는 모든 정현파를 cos으로 환원해서 생각하는 것입니다. 따라서 sin은 cos이 $$ \pi / 2 $$ 만큼 이동한 것으로 생각하면 됩니다.

<br>
<center><img src="../assets/img/math/calculus/phasor/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 그러면, 좀 더 개념을 확장해 보도록 하겠습니다. 사용할 변수는 $$ A, \theta, f, t $$ 입니다. 
- ① $$ A $$ : 위 그림에서는 막대의 길이에 해당하며 **진폭**과 관련 있습니다.
- ② $$ \theta $$ : 위 그림에서 막대가 움직이는 각도에 해당

위 그림과 같이 막대의 길이가 `A`이고 $$ \theta $$ 의 크기를 변경해가면서 반시계 방향으로 움직여 보겠습니다.








<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>