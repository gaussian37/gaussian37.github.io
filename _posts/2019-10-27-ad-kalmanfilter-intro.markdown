---
layout: post
title: 칼만필터 인트로덕션
date: 2019-10-27 00:00:00
img: vision/kalmanfilter/kalman.PNG
categories: [ad-kalmanfilter] 
tags: [컴퓨터 비전, 칼만 필터, kalman filter] # add tag
---

- 출처: Understanding Kalman Filters (Mathworks), Bayesian Inference : Kalman filter에서Optimization까지

<br>
 
- 이번 글에서는 칼만 필터와 관련하여 전체적으로 한번 살펴보려고 합니다. 전체 내용을 한번 쓱 살펴보겠습니다.
- 칼만 필터는 루돌프 칼만에 의해서 개발된 `Optimal estimation algorithm`입니다.
- 현재 내비게이션, 컴퓨터 비전, 신호 처리 등등 다양한 분야에서 사용중에 있고 최초로 사용된 것은 아폴로 프로젝트 였다고 전해집니다.
- 그러면 칼만 필터는 **언제 사용**될까요?
- 주로 사용 되는 것은 알고 싶은 변수를 직접적으로 확인할 수는 없고 `변수를 간접적인 방법으로 유도` 해서 알아내야 할 때 사용할 수 있습니다.
    - 예를 들면 직접적으로 확인하고 싶은 곳이 너무 온도가 높거나 아니면 외부 환경이라서 센서를 설치할 수 없는 경우가 있습니다.
- 또는 다양한 센서들을 통하여 값을 측정할 수는 있지만 노이즈가 발생할 때 `노이즈 문제를 개선`하기 위한 경우에 사용할 수 있습니다.

<br>

## **State Observer**

<br>

- 그러면 칼만 필터의 내용에 대하여 한번 접근해 보겠습니다. 첫번째로 다룰 개념은 `State Observer`입니다.
- `state` 즉, 상태를 관찰한다는 것은 무엇을 의미할까요? 이것은 **우리가 직접적으로 확인하거나 관측하기 어려운 것을 예측**하는 것이라고 생각하면 됩니다.
- 예를 들어 다른 사람의 기분이 어떤지 알고 싶은데, 그것을 우리가 직접적으로 그 사람의 마음을 확인할 수 있는 방법은 없습니다.
- 대신에 말을 걸거나 어떤 상호 작용을 해보고 그 사람의 표정이나 태도 같은 것으로 통하여 **간적접으로** 그 사람의 기분을 예상해볼 수 있습니다.
- 이렇게 직접적으로 확인하기는 어려워서 간접적인 방법으로 확인한 상태를 `Estimated state` 라고 하고 기호로 $$ \hat{x} $$ 라고 적겠습니다.   

<br>
<center><img src="../assets/img/ad/kalmanfilter/introduction/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>    

- 만약 우주 여행을 한다고 한다고 한다면(?), 제트 엔진 내부의 온도를 지속적으로 모니터링 해주어서 고장이 나지 않도록 해야 합니다.
- 제트 엔진 내부의 온도를 직접적으로 추정하는 게 좋겠지만 센서가 열 때문에 녹아버리기 때문에 **간접적으로 엔진 외부 센서를 통하여 추정**을 해야 합니다.
- 여기서 제트 엔진 내부 온도를 직접적으로 확인 하기 어려워 추정한다는 것은 마치 다른 사람의 마음을 추정하는 앞의 예제와 똑같은 상황입니다.
- 위 그림과 같이 제트 엔진 외부에 장착한 센서는 엔진 외부의 온도를 **직접적**으로 관측할 수 있습니다. 이 온도를 $$ T_{ext} $$ 라고 하겠습니다. 그리고 제트 엔진 내부의 온도는 $$ T_{in} $$이라고 하겠습니다.   

<br>
<center><img src="../assets/img/ad/kalmanfilter/introduction/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 연료 $$ w_{fuel} $$을 투입하면 제트 엔진에서 소비를 할 것이고 이 때, $$ T_{in} $$의 변화가 있을 것입니다. 그리고 제트 엔진 외부에서 직접 $$ T_{ext} $$를 관측할 수 있습니다.
- 여기서 문제는 우리가 알고 싶은 $$ T_{in} $$을 알 수 없다는 것이고 이것은 **모델**을 이용하여 간접적으로 구해야 합니다.
- 앞에서 설명하였듯이 hat은 예측값을 말합니다. 따라서 $$ \hat{T}_{in} $$은 어떤 수학적인 모델을 이용하여 예측한 제트 엔진 내부 온도이고 $$ \hat{T}_{ext} $$ 또한 수학적 모델을 통하여 예측한 제트 엔진 외부의 온도입니다.
- 그러면 실제로 비교할 수 있는 것은 센서를 이용하여 직접 관측한 외부 온두 $$ T_{ext} $$와 수학적 모델로 예측한 $$ \hat{T}_{ext} $$인데, 이 두 값의 오차를 최소화 한다면 수학적 모델이 좀 더 강건해질 것입니다.
- **모델이 강건해 지면** $$ \hat{T}_{in} $$ 또한 실제 값과 가까워 질 것으로 기대할 수 있습니다. 즉, 우리가 직접적으로 $$ T_{in} $$을 구할 수는 없지만 모델을 강건하게 만들어 $$ \hat{T}_{in} $$을 $$ T_{in} $$에 수렴하도록 만들도록 하는 것이 우리의 목표가 되겠습니다.
    - 이 목표를 달성하기 위해 사용 하는 것이 $$ T_{ext}, \hat{T}_{ext} $$ 두 값이 되겠습니다.      
    
<br>
<center><img src="../assets/img/ad/kalmanfilter/introduction/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/ad/kalmanfilter/introduction/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>
<center><img src="../assets/img/ad/kalmanfilter/introduction/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>