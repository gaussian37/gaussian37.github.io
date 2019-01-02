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
        + 아랫 부분의 면적은 확률이므로 면적의 넓이는 1이 됩니다.
    + 일반적으로 $$ p(x; \mu, \sigma^{2}) $$로 표현합니다.
    + 수식은 $$ p(x; \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi}\sigma} exp( -\frac{ (x-\mu)^{2} }{2\sigma^{2}} ) $$ 입니다.
    + 여기서 $$ \sigma $$ 는 std(standard deviation)으로 종모양의 중심에서 얼마나 벌어져 있는지 너비에 대한 값입니다.
        + std가 클수록 너비가 커지므로 옆으로 퍼지게 되고 std가 작을수록 너비가 작아져 높이가 커지게 됩니다.
        + <img src="../assets/img/ml/concept/anomaly-detection/gaussianDistributionExample.PNG" alt="Drawing" style="width: 300px;"/>
+ Gaussian Distribution 을 구할 때 기존의 데이터를 이용하여 `Parameter extimation`을 해야 합니다.
    + 앞에서 설명한 바와 같이 $$ \mu, \sigma $$ 두 개가 있습니다.
    + DataSet = \{x^{(1)}, x^{(2)}, ... , x^{(m)} \} 가 있을 때,
    + mean = $$ \mu = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}$$
    + variance = $$ \sigma^{2} = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} -\mu)^{2} $$     
        + 이 때 표본 분산을 나누는 term은 정확하게는 m-1 입니다. 하지만 머신러닝과 같이 표본 데이터가 많은 경우에는 m으로 두고 계산해도 대동소이 합니다.
        + 왜 m-1로 나눠야 하는지는 다음 링크를 참조해 주세요.
            + https://gaussian37.github.io/interview-datascience-Q6-the-reason-of-n-1-tem/

## Anomaly Detection Algorithm

그러면 Anomaly Detection 알고리즘에 대하여 알아보도록 하겠습니다.

+ Dataset = $$ \{x^{(1)}, x^{(2)}, ..., x^{(m)} \}$$ 이 있다고 가정해 봅시다.
+ 각각의 $$ x^{(i)} $$는 벡터 입니다. 따라서 벡터 내의 원소는 각각의 특징을 가지고 있습니다.
+ 따라서 모델 $$ p(x) = p(x_{1})p(x_{2})...p(x_{m}) $$ 으로 나타낼 수 있습니다. 
    + 이 때, $$ x_{1} \sim N(\mu, \sigma^{2}), x_{2} \sim N(\mu, \sigma^{2}), ... $$ 와 같이 나타낼 수 있습니다.
    + 따라서 $$ p(x) = p(x_{1}; \mu_{1}, \sigma_{1}^{2})p(x_{2}; \mu_{2}, \sigma_{2}^{2})...p(x_{m}; \mu_{m}, \sigma_{m}^{2})$$
    + 위와 같은 형태로 확률의 곱이 유효하려면 $$ x_{i} $$ 각각의 특성이 서로 독립적이어야 합니다.
    + 실제 데이터의 경우 완전히 독립적인 feature를 가지기는 어려워 feature간 dependency가 있으나 위의 식은 효과가 있습니다.
+ p(x) 식을 좀 더 심플하게 정리해 보면 다음과 같습니다.
    + 모델  $$ p(x) = p(x_{1}; \mu_{1}, \sigma_{1}^{2})p(x_{2}; \mu_{2}, \sigma_{2}^{2})...p(x_{m}; \mu_{m}, \sigma_{m}^{2}) = \Pi_{i=1}^{m} p(x_{i}, \mu_{i}, \sigma_{i}^{2})$$

### Anomaly Detection 알고리즘 순서

+ anomalous 한 특징을 찾을 수 있는 feature $$ x_{i} $$를 선정합니다.
+ n개의 feature가 있을 때, $$ \mu_{1}, \mu_{2}, ..., \mu_{n}$$ 과 $$ \sigma_{1}, \sigma_{2}, ..., \sigma_{n} $$ 을 구합니다.
    + 즉, $$ \mu $$ 와 $$ \sigma $$ 는 n개의 원소를 가지는 벡터 입니다.
    + mean : $$ \mu_{j} = \frac{1}{m}\sum_{i=1}^{m}x_{j}^{(i)}$$
    + variance : $$ \sigma_{j}^{2} = \frac{1}{m}\sum_{i=1}^{m}(x_{j}^{(i)} - \mu_{j})^{2} $$
+ 새로운 데이터 x가 주어지면, 확률 p(x)를 계산합니다.
    + 모델 $$ p(x) = \Pi_{j=1}^{n}p(x_{j}; \mu_{j}, \sigma_{j}) = \Pi_{j=1}^{n} \frac{1}{\sqrt{2\pi}\sigma_{j}}exp( -\frac{ (x_{j}-\mu_{j})^{2} }{2\sigma_{j}^{2}} )$$ 입니다.
    + 이 때 $$ \epsilon $$ 값을 정하고 이 값보다 p(x)가 작으면 ($$ p(x) \lt \epsilon $$) `Anomaly`로 판단합니다.
+ <img src="../assets/img/ml/concept/anomaly-detection/anomalyDetectionExampleSlide.PNG" alt="Drawing" style="width: 500px;"/>
    + 2d 그래프와 3d 그래프를 비교해서 보면, 2d에서 빨간색 점들이 모여있는 위치와 3d에서 높이가 높은 부분이 대응됩니다.
    + 2d, 3d 그래프의 외곽지역 보란색 부분끼리 대응되고, 낮은 확률을 가집니다.
    + 테스트 케이스 $$ x_{test}^{(1)} $$ 의 확률은 0.0426이라고 했을 때, $$ \epsilon = 0.02 $$ 보다 크므로 normal 데이터 입니다.
        + 2d 그래프에서 $$ x_{1} $$은 군집된 데이터 `내부`에 있습니다.
    + 테스트 케이스 $$ x_{test}^{(2)} $$ 의 확률은 0.0021이라고 하면 anomalous 한 데이터 입니다.
        + 2d 그래프에서 $$ x_{2} $$은 군집된 데이터 `외곽`에 있습니다.     

## Multivariate Gaussian Distribution

## Anomaly detection using the Multivariate Gaussian Distribution 
        
## Anomaly Detection Algorithm Evaludation

Anomaly Detection 알고리즘을 만들었다면 중요한 요소 중 하나인 evaluation 하는 방법에 대하여 알아보겠습니다.
다른 알고리즘들과 마찬가지로 평가하는 방법이 있어야 `parameter tuning` 이나 `feature를 선택` 하는 것이 쉬워집니다.

+ labeled 데이터를 가지고 있다고 가정해 보겠습니다.
    + y = 0, normal
    + y = 1, anomalous
    
+ `Training set`은 `normal 데이터`로 가정합니다.    
    + Training set : $$ x^{(1)}, x^{(2)}, ... , x^{(m)} $$
    
+ `Cross validation & Test set` 에서는 `anomalous 데이터`를 추가합니다.    
    + Cross validation set : $$ (x_{cv}^{(1)}, y_{cv}^{(1)}), (x_{cv}^{(2)}, y_{cv}^{(2)}), ..., (x_{cv}^{(m_{cv})}, y_{cv}^{(m_{cv})})
    + Test set : $$ (x_{test}^{(1)}, y_{test}^{(1)}), (x_{test}^{(2)}, y_{test}^{(2)}), ..., (x_{test}^{(m_{test})}, y_{test}^{(m_{test})})

+ 예를 들어 다음과 같은 데이터 셋이 있다고 가정해 보겠습니다.
    + 10,000개의 normal 데이터 (y = 0)
    + 20개의 anomalous 데이터 (y = 1)
+ 이 때 다음과 같이 training/validation/test 셋을 구성할 수 있습니다.
    + Training set : 6,000개의 normal 데이터 (y = 0)
    + CV : 2,000개의 normal 데이터 (y = 0) + 10개의 anomalous 데이터 (y = 1)
    + Test : 2,000개의 normal 데이터 (y = 0) + 10개의 anomalous 데이터 (y = 1)
+ Algorithm evaluation 하는 방법은 다음 순서와 같습니다.
    + 모델 p(x)를 training set $$ \{ x^{(1)}, x^{(2)}, ... , x^{(m)} $$ 에 학습 시킵니다.
    + cross validation/test example x 에서 아래와 같이 판단합니다.
        + 만약 $$ p(x) \lt \epsilon $$ 이면 anomalous
        + 만약 $$ p(x) \gt \epsilon $$ 이면 normal
    + 판단한 결과를 evaluation 하는 방법에는 다음과 같은 방법이 있습니다.
        + True positive/False positive/False negative/True negative
        + Precision/Recall
        + F1-score
    + 단순히 normal, anomalous 한 데이터를 classification 하는 방식으로 accuracy를 생각하면 anomalous한 데이터가 너무 작기 때문에 적합하지 않습니다.
        + 무조건 normal 이라고 해도 accuracy가 상당히 높게 나오기 때문입니다. 
    + evaluation 한 내용을 기반으로 파라미터 $$ \epsilon $$을 최종적으로 학습할 수 있습니다.

## Anomaly Detection Vs. Supervised Learning

## Anomaly Detection에서는 어떤 feature를 사용하는 것이 좋을까?

## 
        


