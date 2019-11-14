---
layout: post
title: 선형 칼만 필터
date: 2019-11-07 00:00:00
img: ad/kalmanfilter/kalman_filter.jpg
categories: [ad-kalmanfilter] 
tags: [칼만 필터, kalman filter, 선형 칼만 필터] # add tag
---

<br>

- 참조: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mathybits

<br>

- 이번 글에서는 선형 칼만 필터를 통하여 칼만 필터의 기본 컨셉에 대하여 자세하게 알아보도록 하겠습니다.

<br>

## **목차**

<br>

- ### 1. 칼만 필터란 무엇일까?
- ### 2. 칼만 필터로 무엇을 할 수 있을까?
- ### 3. 어떻게 칼만필터가 문제를 다루는 지 살펴보자
- ### 4. 행렬을 통하여 문제 다루어 보기
- ### 5. measurement로 estimate 에 반영해 보기
- ### 6. gaussian 결합
- ### 7. 앞에서 다룬 내용 종합

<br>

## **1. 칼만 필터란 무엇일까?**

<br>

- 칼만 필터는 일부 동적 시스템에 대한 정보가 확실하지 않은 곳에서 사용할 수 있으며 시스템이 다음에 수행 할 작업에 대한 정확한 추측을 할 수 있습니다.
- 칼만 필터는 센서를 통해 추측한 움직임에 노이즈가 들어오더라도 노이즈 제거에 좋은 역할을 합니다.
- 칼만 필터는 지속적으로 변화하는 시스템에 이상적입니다. 왜냐하면 어떤 연산 환경에서는 메모리가 부족할 수 있는데 칼만 필터에서는 이전 상태 이외의 기록을 유지할 필요가 없기 때문입니다. 
- 또한 연산 과정 또한 빠르기 때문에 `실시간 문제` 및 `임베디드 시스템`에 적합합니다.
- 이 글에서 살펴볼 `Linear` 칼만 필터는 기본적인 확률과 행렬에 대한 지식만 있으면 이해 가능합니다. 여기까지 읽고 관심이 있으시면 아래 내용을 한번 살펴보시기 바랍니다.

<br>

## **2. 칼만 필터로 무엇을 할 수 있을까?**

<br>

- 먼저 칼만 필터로 어떤것을 할 수 있는 지 한번 살펴보겠습니다.
- 만약 어떤 로봇이 있고 로봇이 이동하기 위해서는 `어디에 있는지` 정확히 알아야합니다.
- 그러면 로봇의 위치를 어떻게 나타낼 것인지 먼저 정해보겠습니다.
    - 로봇의 상태를 $$ \vec{x_{k}} = (\vec{p}, \vec{v}) $$ 이라고 정의하겠습니다.
    - 여기서 $$ \vec{p} $$는 위치이고 $$ \vec{v} $$는 속도입니다.
- 이 예에서는 위치와 속도이지만 탱크의 유체 양, 자동차 엔진의 온도, 터치 패드에서 사용자의 손가락 위치 또는 추적해야 할 여러 항목에 대한 데이터 일 수 있습니다.
- 만약 이 로봇의 GPS 센서의 정확도가 약 10m라고 가정해 보겠습니다. 만약 이 GPS 센서만을 이용하여 이동을 한다면 정확하게 움직일 수 있을까요? 아마도 절벽에 떨어질 수도 있을 것입니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/robot_ohnoes.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 따라서 GPS라는 센서만으로 위치를 파악하기에는 충분하지가 않습니다. 이것을 좀 더 개선을 해야하는데, 개선점을 `칼만 필터`를 통해서 찾아보겠습니다.

<br>

## **3. 어떻게 칼만필터가 문제를 다루는 지 살펴보자**

<br>

- 먼저 어떤 로봇의 상태부터 정의해 보도록 하겠습니다.

<br>

$$ \vec{x} = \begin{bmatrix} p\\ v \end{bmatrix} $$

<br>

- 여기서 $$ p $$는 위치를 나타내고 $$ v $$는 속도를 나타냅니다.
- 이 때, 위치와 속도를 정확하게 알아내기는 어렵습니다. 대신 실제 속도와 위치가 `어떤 범위`안에 속할 것이라는 것 정도는 예측할 수 있습니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/0.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 칼만 필터에서는 위치와 속도 두 변수를 랜덤 `가우시안 분포`로 가정합니다. 
- 각 변수는 평균과 분산의 특성을 가지고 있습니다. 평균 $$ \mu $$는 랜덤 분포의 중심이 되고 분산 $$ \sigma^{2} $$은 불확실성(uncertainty)가 됩니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/1.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그래프를 보면 가로축은 속도이고 세로 축은 위치 입니다.
- 가로축과 세로축의 가운데에 하얀 점이 있는데 그것은 평균 $$ \mu $$ 라고 하고 그 중심을 기준으로 분포가 되어 있음을 알 수 있습니다.
- 이 때, 가로축은 속도이므로 속도의 분산에 의해 분포가 되어 있고 세로축은 위치이므로 위치의 분산에 의해 분포가 되어 있습니다.
- 그런데 분포의 패턴을 보면 가로축으로만 넓게 분포되어 있고 세로축으로는 넓게 분포되어 있지 않습니다.
- 중요한 것은 가로축이 증가하면 세로축이 증가한는 등의 상관관계가 없다는 것에 있습니다.
    - 즉, `uncorrelated` 하다고 말할 수 있습니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/2.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 반면에 위 예시는 좀 더 재밌는데요, 위치와 속도가 서로 상관관계를 가지는 것으로 보입니다. 즉, `correlated` 합니다.
- 이런 경우에는 특정 위치를 관측하는 것이 특정 속도와 연관이 있다고 해석할 수도 있습니다.

- 사실 현실 세계에서도 위치와 속도의 관계를 보면 이와 유사한 관계를 가지고 있습니다.
    - 예를 들어 속도가 높으면 더 멀리까지 이동할 수 있고 속도가 느리면 그리 멀리 가지 못하는 것과 같습니다.
- 이러한 관계를 주목하는 것이 상당히 중요한데, 왜냐하면 이런 관계를 파악하게 되면 `더 많은 정보`를 얻을 수 있기 때문입니다.
    - 앞선 위치와 속도 관계와 같이 한 가지를 측정하게 되면 다른 값이 어떻게 나오게 되는 지 예측 할 수 있는 것과 같습니다.
- 이러한 관계를 파악하는 것이 칼만 필터의 한가지 목적이 됩니다. 
- 불확실환 관측으로 부터 가능한한 더 많은 정보를 얻어내는 것이 우리의 목적입니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/3.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이러한 상관관계는 `covariance matrix, 공분산`을 통하여 정의될 수 있습니다.
- 공분산에서 각각의 원소 $$ \Sigma_{ij} $$는 $$ i $$번 째 상태 변수와 $$ j $$ 번째 상태 변수의 상관관계를 나타냅니다.
    - 당연히 i,j 와의 관계와 j,i와의 관계는 같기 때문에 공분산 행렬은 대칭행렬이 됩니다.

<br>

## **4. 행렬을 통하여 문제 다루어 보기**

<br>

- 앞에서 다룬 내용들을 이제 가우시안을 통하여 한번 모델링 해보려고 합니다.
- 먼저 $$ k $$번째 타임에 필요한 정보는 2가지 입니다. 최적의 예측치라고 가정하는 $$ \hat{x_{k}} $$ (앞에서 설명한 평균 $$ \mu $$에 해당)과 공분산 행렬 $$ P_{k} $$에 해당합니다.

<br>

$$
\begin{equation} \label{eq:statevars} 
\begin{aligned} 
\mathbf{\hat{x}}_k &= \begin{bmatrix} 
\text{position}\\ 
\text{velocity} 
\end{bmatrix}\\ 
\mathbf{P}_k &= 
\begin{bmatrix} 
\Sigma_{pp} & \Sigma_{pv} \\ 
\Sigma_{vp} & \Sigma_{vv} \\ 
\end{bmatrix} 
\end{aligned} 
\end{equation}
$$

<br>

- 물론 여기에서는 간단하게 위치와 속도만 사용하고 있습니다. 하지만 어떤 상태를 모델링 하느냐에 따라서 어떤 변수가 추가될 지는 상황에 따라 다릅니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/4.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 현재 상태가 (k - 1) 번째 상태라고 가정하고 그 다음 상태인 k번째 상태를 예측해보려고 합니다.
- 기억할 것은 우리는 어떤 상태가 `진짜`인지는 모르지만 예측 함수를 통하여 새로운 상태(k-1 → k)를 예측한다는 것입니다.

<br>
<center><img src="../assets/img/ad/kalmanfilter/linear_kf_basic/5.PNG" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이렇게 상태 예측을 하는 것을 보면 공간 상에서 상태를 변환하는 것 처럼 볼 수 있습니다.
- 공간 상에서의 변환은 행렬을 통하여 나타낼 수 있습니다. 그러면 k-1 번째 상태에서 k 번째 상태로 변환하는 행렬을 $$ F_{k} $$ 라고 두겠습니다.

- 다음 상태의 위치와 속도를 예측하기 위하여 행렬을 어떻게 사용하면 될까요?
- 이렇게 행렬을 만드는 모델링 작업을 할 때 필요한 것이 운동 방정식과 같은 기존의 알려진 물리 방정식들 입니다.

<br>

$$ 

\begin{split} 
\color{deeppink}{p_k} &= \color{royalblue}{p_{k-1}} + \Delta t &\color{royalblue}{v_{k-1}} \\ 
\color{deeppink}{v_k} &= &\color{royalblue}{v_{k-1}} 
\end{split}

$$

<br>

- 다르게 표현하면 다음과 같습니다.

<br>

$$

\begin{align} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \begin{bmatrix} 
1 & \Delta t \\ 
0 & 1 
\end{bmatrix} \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
&= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \label{statevars} 
\end{align}

$$

<br>

- 여기서 적용된 식은 "거리 = 움직인 시간 x 속도" 입니다. 따라서 거리의 변화량을 표현할 때 이 식이 적용되었습니다.
- 현재 이 식에서 속도는 변화가 없는 등속 운동이라는 상황입니다.

<br>

- 여기 까지 살펴보면 운동 방정식을 통하여 `prediction matrix`인 $$ F_{k} $$를 만들었습니다.
- 하지만 앞에서 살펴본 변수사이의 상관관계를 나타내는 공분산 행렬에 대해서는 아직 다루지 않았습니다.

<br>

- 만약 어떤 분포에 속하는 모든 점들을 행렬 $$ A $$와 곱하게 하면 공분한 행렬 $$ \Sigma $$는 어떻게 될까요?
- 이것을 살펴 보기에 앞서 다음 식을 한번 살펴보도록 하겠습니다.

<br>

$$
\begin{equation} 
\begin{split} 
Cov(x) &= \Sigma\\ 
Cov(\color{firebrick}{\mathbf{A}}x) &= \color{firebrick}{\mathbf{A}} \Sigma \color{firebrick}{\mathbf{A}}^T 
\end{split} \label{covident} 
\end{equation}
$$

<br>

- 이 식에 대한 자세한 전개 내용은 다음과 같습니다. 단순히 계산적인 측면이니 스킵하셔도 무방합니다. 단 결과는 일단 숙지하고 적용해 보도록 하겠습니다.
- 앞에서 설명한 다음 두 식을 조합해 보겠습니다.

<br>

$$
\begin{equation} 
\begin{split} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
\color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \mathbf{F}_k^T 
\end{split} 
\end{equation}
$$

<br>

### **influence**

<br>

- 여기서 더 추가해야 할 것이 있습니다. 그것은 현재 시스템에 영향을 끼칠 수 있는 요소들 입니다.
- 예를 들어 자동차나 로봇 같은 경우에 이동 중에 가속도가 붙고 있기 때문에 단순히 (현재 속도 x 이동 시간)만큼만 더 이동하지 않고 가속도 만큼 더 반영 되어 이동하게 됩니다.
- 물론 속도 또한 가속도가 반영되어 변경되게 됩니다.
- 물리 시간에 많이 사용하였듯이 가속도를 $$ a $$라고 한번 표현해 보겠습니다. 그러면 다음과 같이 표현할 수 있습니다.

<br>

$$

\begin{split} 
\color{deeppink}{p_k} &= \color{royalblue}{p_{k-1}} + {\Delta t} &\color{royalblue}{v_{k-1}} + &\frac{1}{2} \color{darkorange}{a} {\Delta t}^2 \\ 
\color{deeppink}{v_k} &= &\color{royalblue}{v_{k-1}} + & \color{darkorange}{a} {\Delta t} 
\end{split}

$$

<br>

- 이것을 행렬 형태로 한번 나타내 보겠습니다.

$$

\begin{equation} 
\begin{split} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} + \begin{bmatrix} 
\frac{\Delta t^2}{2} \\ 
\Delta t 
\end{bmatrix} \color{darkorange}{a} \\ 
&= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} + \mathbf{B}_k \color{darkorange}{\vec{\mathbf{u}_k}} 
\end{split} 
\end{equation}

$$

- 여기서 $$ B_{k} $$는 `control matrix`라고 하고 $$ \color{darkorange}{\vec{\mathbf{u}_k}} $$ 은 `control vector`라고 합니다.
- 만약 다루고 있는 시스템이 간단하여 외부 영향이 없다면 이 부분을 생략해도 됩니다. 즉, 모델링에 따라 적용할 수 있고 적용 안할 수도 있습니다.
