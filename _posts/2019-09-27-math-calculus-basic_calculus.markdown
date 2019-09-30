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

### **Rise Over Run**

<br>

- 먼저 `Calculus`의 간단한 정의를 내리면 **어떤 함수가 입력 변수에 관하여 어떻게 변하는지**를 나타낸 것이라고 말할 수 있습니다.

 
<br>
<center><img src="../assets/img/math/calculus/basic_calculus/1.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 그러면 위 그림을 어떻게 해석하면 될까요? **입력 변수에 관하여 함수값이 어떻게 바뀌는지를 살피는 것**이 목적입니다.
- 위 입력값은 `X축`인 시간입니다. 그러면 시간에 따라 함수값인 `Speed`가 어떻게 바뀌는지 살펴보면 시간에 따라 다양하게 변화하고 있습니다.
- 이 때의 변화량을 알고 싶습니다. 변화량을 확인하   고 싶으면 곡선에 접하는 선의 기울기를 이용하여 확인할 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/2.gif" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 이미지를 보면 입력값인 시간에 따라서 속도가 바뀌고 그 순간 순간의 변화가 얼만큼인지에 해당하는 변화율 또한 나타낼 수 있음을 알 수 있습니다.
- 변화율은 속도가 증가할때는 양의 방향의 직선을 나타내다가 속도가 정체하면 수평방향의 직선 그리고 속도가 감소하면 음의 방향의 직선 모양을 가집니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 계속 다룬 `gradient` 라는 개념은 변화율을 뜻하는 것이었고 이것은 `Rise Over Run`이라고 표현하기도 합니다.
- 위 그림을 보면 `Run`은 입력값 `x`의 증감양을 나타내고 `Rise`는 함수값 `f(x)`의 증감양을 나타냅니다. 따라서 입력값 대비 함수값의 변화양이 `gradient`가 됩니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 예를 들어 위와 같은 경우에 입력이 4인 지점의 `gradient`를 보면 `run`인 입력값의 변화량은 `6 - 2 = 4`이고 `rise`의 변화량은 `4 - 0 = 4`이므로 `gradient`는 **1**이 됩니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위와 같은 그래프에서 `gradient`만을 그래프로 표현하면 어떻게 될까요?
- 위 그래프의 `gradient`의 추이를 보면 입력값이 증가할수록 `gradient`가 점점 감소하고 있습니다.
- 그리고 `gradient`는 양의 값부터 시작해서 음의 값으로 변하고 있습니다.
- 또한 `gradient`가 0이되는 지점 즉, **turning point**는 한 곳에만 있음을 알 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/6.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 그래프를 표현하면 위와 같이 표현할 수 있습니다.
- 앞에서 살펴본 바와 같이 입력값이 증가할수록 `gradient`가 감소하고 **turning point** 즉, `gradient = 0`인 지점은 한번 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 만약 위와 같은 그래프의 `gradient`를 표현하면 어떻게 될까요?

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/8.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위와 같이 `gradient`를 표현할 수 있습니다.
- `gradient`는 음의 값부터 시작해서 양의 값으로 다시 음의 값으로 변했다가 최종 양의 값이 됩니다.
- 이 때 **turning point**는 3번 생깁니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/9.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 또 다른 예시를 살펴보겠습니다. 이번 그래프는 앞의 그래프에 비교해서 y축으로 평행이동한 상태입니다.
- 이 경우의 `gradient`는 어떻게 될까요?

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/10.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- `gradient`는 변화량과 관계가 있지 실제 함수값이 가지는 값의 크기와는 관련이 없습니다. 변화량만 같다면 동일한 패턴의 `gradient`그래프를 그릴 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/11.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 그러면 반대로 위와 같은 `gradient` 그래프가 있으면 어떤 형태의 그래프에서 도출될 수 있을까요?
- 위 `gradient`를 보면 먼저 4번의 0 값을 가지기 때문에 4번의 **turnning point**를 가집니다.
- 그리고 `gradient`가 범위에 따라 음의 값인지 양의 값인지를 확인하여 살펴보면 다음 그래프를 그릴 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/12.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 예를 들면 위와 같이 4번의 변곡점을 가지는 그래프를 그릴 수 있습니다.

<br>

### **Definition Of Derivative**

<br>

- 그러면 앞에서 배운 개념을 좀 더 formal한 방법으로 한번 다루어 보겠습니다.
- 앞에서 다룬 `gradient = Rise Over Run`을 생각해보면서 좀 더 정교한 식으로 한번 나타내 보려고 합니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/13.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 먼저 `gradient`는 순간 순간마다 다릅니다. 즉, 위 그림과 같이 어떤 점을 잡느냐에 따라서 `gradient`는 다릅니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/14.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이런 `gradient`를 계산하기 위해서는 `Rise Over Run`을 좀 더 수식적으로 나타낼 필요가 있습니다.
- 먼저 수학에서 사용하는 기호인 $$ \Delta $$는 **작은 변화량**을 나타낼 때 사용합니다.
- `Run`에 해당하는 가로축의 변화량을 보면 $$ \Delta x $$ 만큼 변화하였고 `Rise`인 세로축을 보면 $$ f(x + \Delta x) - f(x) $$ 만큼 변한것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/15.gif" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이 때 `Run`에 해당하는 입력값의 변화량 $$ \Delta x $$를 줄이면 줄일수록 더 나은 `gradient` 근사값을 찾을 수 있습니다. 

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/16.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이 표현 방법을 위와 같이 `limitation`을 이용하여 식으로 나타낼 수 있습니다.
- `gradient`라는 용어 대신 식으로 $$ \frac{df}{dx} $$로 나타내기도 하고 $$ f'(x) $$로 나타내기도 합니다.
- 다시 한번 정리해 보겠습니다. `gradient`를 나타내는 식은 다음과 같습니다.

<br>

$$ f'(x) = \lim_{\Delta x \to 0}\Bigl( \frac{ f(x + \Delta x) - f(x) }{\Delta x} \Bigr) $$

<br>

- 이 식을 이용하여 $$ f(x) = 3x + 2 $$일 때를 예로 한번 `gradient`를 살펴보겠습니다.

<br>

$$ f'(x) = \lim_{\Delta x \to 0} = \Bigl( \frac{ 3(x + \Delta x) + 2 - (3x + 2) }{\Delta x} \Bigr) = 3 $$

<br>

- 또 다른 예로 $$ f(x) = 5x^{2} $$ 일 때를 살펴보겠습니다.

<br>

$$ f'(x) = \lim_{\Delta x \to 0}\Bigl( \frac{ 5(x + \Delta x)^{2} - 5x^{2} }{\Delta x} \Bigr) = \lim_{\Delta x \to 0}\Bigl( \frac{5x^{2} + 10x\Delta x + 5\delta x^{2} - 5x^{2} }{\Delta x} \Bigr) = \lim_{\Delta x \to 0}(10x + 5\Delta x) = 10x $$

<br>

- 매번 `gradient`를 구할 때, 이 과정을 반복하기는 번거로우므로 다음 `Power Rule`을 대신 사용하겠습니다.

<br>
<center><img src="../assets/img/math/calculus/basic_calculus/17.PNG" alt="Drawing" style="width: 600px;"/></center>
<br>

  



 



