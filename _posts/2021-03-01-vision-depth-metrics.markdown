---
layout: post
title: Depth Estimation의 Metric
date: 2021-03-01 00:00:00
img: vision/depth/metrics/0.png
categories: [vision-depth] 
tags: [depth estimation, metrics, rel, rmse] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 참조 : http://ylatif.github.io/papers/IROS2016_ccadena.pdf
- 참조 : https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e
- 참조 : https://towardsdatascience.com/ways-to-evaluate-regression-models-77a3ff45ba70
- 참조 : https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e
- 참조 : https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/
- 참조 : 

<br>

## **목차**

<br>

- ### Depth Estimation Metrics의 종류와 수식
- ### Depth Estimation Metrics의 의미
- ### Pytorch Code

<br>

## **Depth Estimation Metrics의 종류와 수식**

<br>

- 이번 글에서는 Depth Estimation 논문에서 성능 지표로 사용하는 metric에 대하여 간략하게 알아보도록 하겠습니다.
- 주로 사용하는 metric은 `Absolute Relative Error`, `Square Relative Error`, `Root Mean Square Error`, `log scale RMSE`, `Accuracy under a threshold` 입니다.
- 위 metric은 Regression 모델을 평가하는 지표이며 각 지표의 의미는 아래와 같습니다.

<br>

- 아래 식에서 각 기호의 의미는 다음과 같습니다.
- ① $$ p $$ : 특정 픽셀을 의미합니다.
- ② $$ \hat{d}_{p} $$ : 특정 픽셀의 Depth Estimation한 예측값을 의미합니다.
- ③ $$ d_{p} $$ : 특정 픽셀의 Depth GT 값을 의미합니다.
- ④ $$ T $$ : 유효한 Depth GT와 Depth Estimation 출력이 모두 존재하는 픽셀의 총 갯수에 해당하며 이 지점에 한하여 metric을 계산합니다.

<br>

#### **Absolute Relative Error**

<br>

- $$ \frac{1}{T} \sum_{p} \frac{\vert d_{p} - \hat{d}_{p} \vert}{d_{p}} \tag{1} $$

<br>

#### **Squre Relative Error**

<br>

- $$ \frac{1}{T} \sum_{p} \frac{(d_{p} - \hat{d}_{p})^{2}}{d_{p}} \tag{2} $$

<br>

#### **Root Mean Squre Error**

<br>

- $$ \sqrt{\frac{1}{T}\sum_{p}(d_{p} - \hat{d}_{p})^{2}} \tag{3} $$

<br>

#### **log scale RMSE**

<br>

- $$ \sqrt{\frac{1}{T}\sum_{p}(\log{(d_{p})} - \log{( \hat{d}_{p} )^{2}})} \tag{4} $$

<br>

#### **Accuracy under threshold**

<br>

- $$ \text{max} \left(\frac{\hat{d}_{p}}{d_{p}}, \frac{d_{p}}{\hat{d}_{p}} \right) = \delta < \text{threshold} \tag{5} $$

<br>

## **Depth Estimation Metrics의 의미**

<br>

- `Absolute Relative Error`, `Square Relative Error`, `Root Mean Square Error`, `log scale RMSE`은 전통적으로 Regression 모델의 성능을 측정하기 위해 사용된 Metric이며 `Accuracy under threshold`는 Depth Estimation 모델의 Accuracy를 측정하기 위해 도입되었습니다.
- 먼저 `Absolute Relative Error`와 `Squre Relative Error` 부터 살펴보겠습니다. 두 metric은 계산 시, `L1`을 사용할 지 `L2`를 사용하는 지에 대한 차이가 있으며 나머지 형태는 같습니다.
- 

<br>

## **Pytorch Code**

<br>

```python
def compute_depth_errors(gt, pred):   

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())


    delta = torch.max((gt / pred), (pred / gt))
    a1 = (delta < 1.25     ).float().mean()
    a2 = (delta < 1.25 ** 2).float().mean()
    a3 = (delta < 1.25 ** 3).float().mean()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
```

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>