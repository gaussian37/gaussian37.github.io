---
layout: post
title: Calculus 기초
date: 2019-09-27 00:00:00
img: math/calculus/overall.jpg
categories: [math-calculus] 
tags: [calculus] # add tag
---

- 이 글은 Coursera의 Mathematics for Machine Learning: Multivariate Calculus 을 보고 정리한 내용입니다.

<br>

- 먼저 `Calculus`의 간단한 정의를 내리면 **어떤 함수가 입력 변수에 관하여 어떻게 변하는지**를 나타낸 것이라고 말할 수 있습니다.

 
<br>
<center><img src="../assets/img/math/calculus/basic_calculus/1.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 그러면 위 그림을 어떻게 해석하면 될까요? **입력 변수에 관하여 함수값이 어떻게 바뀌는지를 살피는 것**이 목적입니다.
- 위 입력값은 `X축`인 시간입니다. 그러면 시간에 따라 함수값인 `Speed`가 어떻게 바뀌는지 살펴보면 시간에 따라 다양하게 변화하고 있습니다.
- 이 때의 변화량을 알고 싶습니다. 변화량을 확인하   고 싶으면 곡선에 접하는 선의 기울기를 이용하여 확인할 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/2.gif" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 이미지를 보면 입력값인 시간에 따라서 속도가 바뀌고 그 순간 순간의 변화가 얼만큼인지에 해당하는 변화율 또한 나타낼 수 있음을 알 수 있습니다.
- 변화율은 속도가 증가할때는 양의 방향의 직선을 나타내다가 속도가 정체하면 수평방향의 직선 그리고 속도가 감소하면 음의 방향의 직선 모양을 가집니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞에서 계속 다룬 `gradient` 라는 개념은 변화율을 뜻하는 것이었고 이것은 `Rise Over Run`이라고 표현하기도 합니다.
- 위 그림을 보면 `Run`은 입력값 `x`의 증감양을 나타내고 `Rise`는 함수값 `f(x)`의 증감양을 나타냅니다. 따라서 입력값 대비 함수값의 변화양이 `gradient`가 됩니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 예를 들어 위와 같은 경우에 입력이 4인 지점의 `gradient`를 보면 `run`인 입력값의 변화량은 `6 - 2 = 4`이고 `rise`의 변화량은 `4 - 0 = 4`이므로 `gradient`는 **1**이 됩니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위와 같은 그래프에서 `gradient`만을 그래프로 표현하면 어떻게 될까요?
- 위 그래프의 `gradient`의 추이를 보면 입력값이 증가할수록 `gradient`가 점점 감소하고 있습니다.
- 그리고 `gradient`는 양의 값부터 시작해서 음의 값으로 변하고 있습니다.
- 또한 `gradient`가 0이되는 지점 즉, **turning point**는 한 곳에만 있음을 알 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/6.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 그래프를 표현하면 위와 같이 표현할 수 있습니다.
- 앞에서 살펴본 바와 같이 입력값이 증가할수록 `gradient`가 감소하고 **turning point** 즉, `gradient = 0`인 지점은 한번 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 만약 위와 같은 그래프의 `gradient`를 표현하면 어떻게 될까요?

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/8.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위와 같이 `gradient`를 표현할 수 있습니다.
- `gradient`는 음의 값부터 시작해서 양의 값으로 다시 음의 값으로 변했다가 최종 양의 값이 됩니다.
- 이 때 **turning point**는 3번 생깁니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 또 다른 예시를 살펴보겠습니다. 이번 그래프는 앞의 그래프에 비교해서 y축으로 평행이동한 상태입니다.
- 이 경우의 `gradient`는 어떻게 될까요?

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/10.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- `gradient`는 변화량과 관계가 있지 실제 함수값이 가지는 값의 크기와는 관련이 없습니다. 변화량만 같다면 동일한 패턴의 `gradient`그래프를 그릴 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 그러면 반대로 위와 같은 `gradient` 그래프가 있으면 어떤 형태의 그래프에서 도출될 수 있을까요?
- 위 `gradient`를 보면 먼저 4번의 0 값을 가지기 때문에 4번의 **turnning point**를 가집니다.
- 그리고 `gradient`가 범위에 따라 음의 값인지 양의 값인지를 확인하여 살펴보면 다음 그래프를 그릴 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/12.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 예를 들면 위와 같이 4번의 변곡점을 가지는 그래프를 그릴 수 있습니다.




 



