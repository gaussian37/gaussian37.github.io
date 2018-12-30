---
layout: post
title: Anomaly Detection (이상치 감지) 
date: 2018-12-23 00:00:00
img: ml/concept/anomaly-detection/anomaly.jpg
categories: [ml-concept] 
tags: [python, machine learning, ml, anomaly detection, 이상치 감지] # add tag
---

이번 포스트에서는 이상 탐지 (Anomaly Detection)라는 문제에 대해 
알려 드리고자합니다. 이 내용은 앤드류 응 교수님 강의 내용을 기반으로 만들어 졌습니다.

## Anomaly Detection은 무엇일까?

+ Anomaly Detection은 Unsupervised Learning이지만 Supervised Learning의 성질을 가지고 있습니다.
    + Anomaly Detection vs Supervised Learning 비교는 뒤에서 다루어 보겠습니다.
+ Anomaly Detection의 예
+ 비행기 엔진 특성
    + x1 = heat generated
    + x2 = vibration intensity
    + Dataset : $$ {x^{(1)}, x^{(2)}, ... , x^{(m)}} $$    
        + m 개의 생산된 비행기 엔지이 있다고 가정해 봅시다.
        + 아래 그림에서 파란색 점은 **unlabeled**로 학습된 데이터 입니다.
        + 빨간색 점은 테스트 데이터 입니다. 테스트 데이터는 새로 생산된 엔진으로 생각할 수 있습니다. 
        + <img src="../assets/img/ml/concept/anomaly-detection/anomaly_detection_ex.png" alt="Drawing" style="width: 300px;"/>
        + 위의 그림에서 직관적으로 이해해 보면 기존 데이터 무리에 있는 테스트 데이터는 normal 합니다.
        + 반면 기존 데이터 군집과 떨어져 있는 데이터의 경우 abnormal 하다고 생각해 볼 수 있습니다.
        + 확률적으로 나타내기 위해 학습한 모델을 P(x) 라고 가정해 보겠습니다.
            + 상수 $$ \epsilon $$ 을 anomaly를 결정짓는 임계값이라고 하겠습니다.
            + 모델 $$P(x_{test}) \lt \epsilon $$ 이면 abnormal
            + 모델 $$P(x_{test}) \ge \epsilon $$ 이면 normal 이라고 할 수 있습니다.
        + <img src="../assets/img/ml/concept/anomaly-detection/anomalyDetectionEx2.png" alt="Drawing" style="width: 300px;"/>
        + 데이터의 군집화 정도를 보았을 때, 중앙에 위치할 수록 normal일 확률은 높아지고 가장자리에 가까워질수록 abnormal에 가까워 집니다.
        
+ Fraud Detection (사기 감지)
    + 데이터 $$ x^{(i)} $$ = i 번째 사용자의 행동 특징 이라고 가정해 봅시다.
    + 비행기 엔진 예제와 비슷하게 Model p(x)를 기존의 data를 통하여 구합니다.
    + 새로운 사용자 즉, 새로운 데이터가 추가되었을 때, $$ p(x) \lt \epsilon $$을 만족하는지 확인합니다.
+ Manufacturing (비행기 엔진과 유사)
+ 데이터 센터에서의 컴퓨터 모니터링
    + 데이터 $$ x^{(i)} $$ = 컴퓨터 i의 특징이라고 하면
    + x1 = 메모리 사용량, x2 = 디스크 접근수, x3 = CPU 부하, x4 = 네트워크 트래픽 등과같은 특성을 가질 수 있습니다.
    
+ 만약 $$ \epsilon $$이 너무 크다면 너무 많은 데이터가 $$ p(x) \lt \epsilon$$을 만족해서 모델의 성능이 안좋아 질 수도 있습니다.
    + 이럴 때에는, $$ \epsilon $$의 크기를 줄여야 합니다.
+ Anomaly Detection 모델링을 할 때 주로 사용하는 것이 `Gaussian Distribution` 입니다. 그러면 `Gaussian Distribution`에 대하여 간략하게 알아보겠습니다.

## Gaussian Distribution

+ Anomaly Detection 모델링을 하기 전에 근본이 되는 Gaussian Distribution에 대하여 알아보도록 하겠습니다.
+ 데이터셋 x가 Gaussian Distribution을 따른다면 파라미터로 mean = $$ \mu $$와 variance = $$ \sigma^{2} $$을 가집니다.
    + 기호로 표시하면 $$ x \sim N(\mu, \sigma^{2}) $$이 됩니다.
+ <img src="../assets/img/ml/concept/anomaly-detection/gaussian_curve.PNG" alt="Drawing" style="width: 300px;"/>
    + Gaussian Curve는 기본적으로 `종 모양`의 곡선을 가집니다.
    + 종 모양의 중간이 `mean`에 해당합니다.
    + 곡선의 아랫 부분에 해당하는 면적이 확률이기 때문에 `mean`에서 멀어질수록 확률은 낮아지게 됩니다.
    + 일반적으로 $$ p(x; \mu, \sigma^{2}) $$로 표현합니다.
    + 수식은 $$ p(x; \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi}\sigma} exp( -\frac{ (x-\mu)^{2} }{2\sigma^{2}} ) $$ 입니다.
    + 여기서 $$ \sigma $$ 는 standard deviation으로 
    
    
          


