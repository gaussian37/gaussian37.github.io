---
layout: post
title: 머신러닝에서 사용되는 평가지표
date: 2019-03-07 00:00:00
img: ml/concept/machineLearning.jpg
categories: [ml-concept] 
tags: [ml, machine learning, 평가 지표, evaluation] # add tag
---

이번 글에서는 머신러닝에서 사용되는 대표적인 평가 지표에 대하여 알아보도록 하겠습니다.

### Confusion Matrix (혼합 행렬)

+ 실제 라벨과 예측 라벨의 일치 갯수를 matrix로 표현하는 기법

<img src="../assets/img/ml/concept/ml-evaluation/1.PNG" alt="Drawing" style="width: 400px;"/>

+ 맞음여부 & 예측한값의 조합으로 표현
+ TP(True Positive) : Positive로 예측해서 True임
    + True : 예측이 맞음
    + Positive : 예측값이 Positive(1)
+ FP(False Positive) : Positive로 예측해서 False임
    + False : 예측이 틀림
    + Positive : 예측값이 Positive(1)인 경우 (즉, 실제값은 거짓(0)입니다.)
+ FN(False Negative) : Negative로 예측해서 False임
    + False : 예측이 틀림
    + Nagative : 예측값이 Negative(0)인 경우 (즉, 실제값은 참(1)입니다.)
+ TF(True Negative) : Negative로 예측해서 True임
    + True : 예측이 맞음
    + Negative : 예측값이 Negative(0)인 경우
    
```python
from sklearn.metrics import confusion_matrix
y_true = [1, 0, 1, 1, 0, 1]
y_pred = [0, 0, 1, 1, 0, 1]
confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
>> (tn, fp, fn, tp)

(2, 0, 1, 3)

``` 

<br><br>

### Accuracy

<img src="../assets/img/ml/concept/ml-evaluation/1.PNG" alt="Drawing" style="width: 200px;"/>

+ 전체 대비 정확하게 예측한 개수의 비율
+ ·$$ ACC = \frac{TP + TN}{TP + TN + FP + FN} $$
+ ·$$ ACC = 1 - ERR $$

```python
from sklearn.metrics import accuracy_score

y_true = np.array([0, 1, 0, 0])
y_pred = np.array([0, 1, 1, 0])

>> accuracy_score(y_true, y_pred)
0.75

```

<br>

### Error Rate

+ 전체 데이터 데비 부정확하게 예측한 갯수의 비율
+ ·$$ ERR = \frac{FP + FN}{TP + TN + FP + FN} $$
+ ·$$ ERR = 1 - ACC $$


### 불균형한 데이터셋의 문제점

+ 불균형한 데이터셋에서 Accuracy나 Error rate를 사용한다면 상당히 부정확해집니다.
    + 시험 합격률이 2%일 때, 모든 수험자가 불합격 한다고 하면 98%의 정확도를 가지게 되어 정확도는 높지만 쓸모없는 모델이 만들어 집니다.
    
### Precision

<img src="../assets/img/ml/concept/ml-evaluation/precision.PNG" alt="Drawing" style="width: 400px;"/>

+ 긍정이라고 예측한 비율 중 진짜 긍정의 비율
+ 긍정이라고 얼마나 잘 예측하였는가?
+ `Positive라고 예측한 것 중에서 얼마나 잘 맞았는지 비율`

+ ·$$ Precision = \frac{TP}{TP + FP} $$

 ```python
from sklearn.metrics import accuracy_score

y_true = np.array([0, 1, 0, 0])
y_pred = np.array([0, 1, 1, 0])

>> precision_score(y_true, y_pred)
0.50

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
>> confusion_matrix(y_true, y_pred)

array([ [2, 0, 0],
        [1, 0, 1],
        [0, 2, 0]])
        
>> precision_score(y_true, y_pred, average='macro')
0.22222222...


>> precision_score(y_true, y_pred, average='macro')
0.33333333...

```

+ average를 macro로 두면 각 열에 대한 precision 값을 모두 더한 다음 열의 갯수로 나눈 것입니다.
+ average를 micro로 두면 모든 열에서 맞은것 즉, 대각선 성분으 총 합을 총 갯수로 나눈 것입니다.

### Sensitive, Recall, True Positive Rate

<img src="../assets/img/ml/concept/ml-evaluation/recall.PNG" alt="Drawing" style="width: 400px;"/>

+ 실제 긍정 데이터 중 긍정이라고 예측한 비율, 반환율, 재현율
+ Precision과 다르게 `실제 Positive한 것 중에서 얼마나 잘 예측하였는지 비율`
+ ·$$ Recall(True\ Positive\ Rate) = \frac{TP}{TP + FN} $$

