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
            + 모델 $$P(x_{test}) \lt \epsilon $$ 이면 abnomal
            + 모델 $$P(x_{test}) \ge \epsilon $$ 이면 normal 이라고 할 수 있습니다.
    
             
          


