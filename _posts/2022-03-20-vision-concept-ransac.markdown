---
layout: post
title: RANSAC (RANdom SAmple Consensus) 개념 및 실습
date: 2022-03-20 00:00:00
img: vision/concept/ransac/0.png
categories: [vision-concept] 
tags: [RANSAC, random sample consensus, 란삭, Lo-RANSAC] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>

- 참조 : https://gnaseel.tistory.com/33
- 참조 : https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html
- 참조 : https://www.youtube.com/watch?v=Q7FqV_bglHo
- 참조 : https://github.com/anubhavparas/ransac-implementation

<br>

## **목차**

<br>

- ### [RANSAC 이란](#ransac의-이란-1)
- ### [RANSAC의 필요성](#ransac의-필요성-1)
- ### [RANSAC 개념](#ransac-개념-1)
- ### [RANSAC의 파라미터 셋팅법](#ransac의-파라미터-셋팅법-1)
- ### [RANSAC의 장단점](#ransac의-장단점-1)
- ### [Early Stop 사용법](#early-stop-사용법-1)
- ### [RANSAC Python Code](#ransac-python-code-1)
- ### [Lo-RANSAC 개념](#lo-ransac-개념-1)
- ### [Lo-RANSAC Python Code](#lo-ransac-python-code-1)

<br>

## **RANSAC 이란**

<br>

- `RANSAC`은 `Random Sampling Consensus`의 줄임말로 **데이터를 랜덤하게 샘플링하여 사용하고자 하는 모델을 fitting한 다음 fitting 결과가 원하는 목표치 (합의점, Consensus)에 도달하였는 지 확인하는 과정**을 통해 모델을 데이터에 맞게 최적화하는 과정을 의미합니다.
- 따라서 `RANSAC`은 특정 모델식이나 알고리즘을 의미하는 것은 아니며 `① 데이터 샘플링`, `② 모델 fitting`, `③ 목표치 도달 확인`이라는 3가지 과정을 반복적으로 수행하는 과정을 의미합니다.
- 선형 함수 모델, 다항 함수 모델, 다양한 비선형 함수 모델 등 어떤 모델과 상관없이 `모델 fitting` 과정을 거치면 되므로 `RANSAC`은 모델을 fitting 하는 일종의 `Framework`라고 말하기도 합니다.

<br>

## **RANSAC의 필요성**

<br>

- 앞의 설명을 참조하면 `RANSAC`이란 `Framework`를 사용하지 않더라도 모델 fitting이라는 과정이 있기 때문에 굳이 `RANSAC`이란 절차를 사용하지 않아도 됩니다. 그럼에도 불구하고 `RANSAC`이 널리 사용되는 이유는 무엇일까요?

<br>

- `RANSAC`이 널리 사용되는 이유는 다음 3가지를 동시에 만족하는 유연한 방법론이기 때문입니다.
- ① `outliers`에 강건한 모델을 만들 수 있습니다.
- ② `outliers`에 강건한 모델을 만드는 방법 중 매우 단순한 방법이므로 구현이 쉽고 응용하기도 쉽습니다.
- ③ 어떤 모델을 사용하여도 `RANSAC`을 이용할 수 있으므로 임의의 모델을 `outliers`에 강건하게 만들 수 있습니다.

<br>

- `RANSAC`은 간단한 절차에도 `outliers`에 강건한 모델 fitting을 할 수 있다는 이유로 널리 사용되고 있습니다.
- 특히 컴퓨터 비전을 이용한 인식 관련 문제 해결 시, 다양한 이유로 발생하는 노이즈나 오인식이 발생하게 되고 이러한 문제에 강건한 모델 설계를 위하여 `RANSAC`은 현재까지 많이 사용되고 있습니다.

<br>

## **RANSAC 개념**

<br>

- 

<br>

## **RANSAC의 파라미터 셋팅법**

<br>

- 앞에서 살펴본 `RANSAC`의 개념 중 설정해 주어야 하는 파라미터가 3개 있습니다. `threshold`와 `sample size` 그리고 `sampling number`입니다.
- `threshold`는 fitting한 모델을 이용하여 데이터 셋에서의 `inlier`가 몇 개인지 파악하는 데 사용하는 기준값이었습니다. 이 값을 어떤 값으로 사용하는 지에 따라서 모델의 성능이 달라지게 됩니다. 매우 중요한 값이므로 `threshold`를 정하는 다양한 접근 방법이 연구가 되었습니다. 사용중인 데이터셋의 통계적인 접근 방법이나 데이터셋에 대한 전문적인 지식으로 접근하는 방법등도 있지만 본 글에서는 **가장 간단하고 확실한 `Grid Search`와 같은 방법으로 여러개의 `threshold`를 시도해보고 적합한 `threshold`를 찾는 방법**을 사용해 볼 예정입니다. 
- 여러 `threshold`에 따른 모델 fitting 양상을 보는 것이 좋은 모델을 선정하는 데 중요한 역할을 합니다.

<br>

- 단, `threshold`에 따른 변화를 살펴보기 위해서는 다른 파라미터인 `sample size`와 `sampling number`는 가능한 고정한 후 실험을 하는 것이 편리합니다. 일반적으로 `sample size`와 `sampling number`를 설정하기 위해서 다음 식을 이용합니다.

<br>

- $$ p = 1 - (1 - (1 - e)^m)^{N} $$

- $$ p \text{ : Probability of obtaining a sample consisting only of inliers – sampling success} $$

- $$ e \text{ : ratio of outliers in dataset} $$

- $$ m \text{ : Number of data sampled per time} $$

- $$ N \text{ : Number of algorithm iterations} $$

<br>

- 위 식에서 $$ p $$ 는 `inlier`로 이루어진 샘플을 얻을 확률이고 기대하는 값이기도 합니다. 따라서 $$ p = 0.99 $$ 와 같이 매우 큰 값으로 설정합니다. 따라서 $$ p $$ 는 상수와 비슷하게 사용할 수 있습니다.
- 그 다음 $$ e $$ 는 실제 데이터 확인을 통하여 `outlier`의 비율을 확인해서 정할 수 있습니다. 만약 `outlier`의 비율을 알 수 없으면 보수적으로 0.5로 적용할 수도 있습니다. 실제로 `outlier`의 비율이 0.5 정도가 된다면 노이즈가 굉장히 많은 데이터이기 때문입니다.
- 실제 정해주어야 하는 값은 $$ m, N $$ 입니다. $$ m $$ 은 크기가 작을수록 $$ p $$ 의 값이 1에 가까워지고 $$ N $$ 은 크기가 클수록 $$ p $$ 의 값이 1에 가까워 집니다.
- 예를 들어 $$ m $$ 이 클수록 샘플링 해야 하는 데이터의 수가 많아지기 때문에 `outlier`가 선택될 가능성이 더 커지게 됩니다. 즉 `inlier`로 이루어진 샘플을 얻을 확률이 낮아지므로 $$ p $$ 가 작아지게 됩니다. 간단하게 $$ e $$ 가 비율이므로 $$ 1 - e $$ 는 1보다 작은 값이고 $$ m $$ 만큼 거듭제곱이 되므로 $$ m $$ 이 커질수록 $$ (1 - e)^m $$ 은 작아지게 됩니다. 따라서 $$ p $$ 또한 작아지도록 반영됩니다. 
- 이러한 이유로 모델링에 필요한 최소 갯수를 샘플링 하는 방법을 많이 사용합니다. 예를 들면 선형 모델을 모델링할 때에는 2개의 샘플만 있으면 되기 때문에 $$ m = 2 $$ 가 될 수 있습니다. 2차 모델의 경우 3개의 샘플이 필요하므로 $$ m = 3 $$ 이 됩니다. 따라서 `RANSAC`에 사용되는 모델에 따라서 $$ m $$ 은 자동으로 결정될 수 있습니다.

<br>

- 따라서 $$ p = 1 - (1 - (1-e)^{m})^{N} $$ 에서 $$ N $$ 을 제외하고 모두 정할 수 있습니다. 직접적으로 $$ N $$ 을 구하기 위하여 다음과 같이 식을 정리할 수 있습니다.

<br>

- $$ \begin{align} N &= \frac{\log{(1 - p)}}{\log{(1 - (1-e)^{m})}} \\ &= \frac{\log{(1 - 0.99)}}{\log{(1 - (1 - 0.5)^{m})}} \end{align} $$

<br>

- 위 식을 조건으로 살펴보았을 때, $$ m = 1 $$ 일 때, $$ N \approx 7 $$, $$ m = 2 $$ 일 때, $$ N \approx 16 $$ 등으로 $$ m = 3 $$ 일 때, $$ N \approx 34 $$, $$ m = 4 $$ 일 때, $$ N \approx 71 $$, ... 과 같이 급격하게 증가하는 것을 볼 수 있습니다. 
- 즉, 모델의 복잡도가 커질수록 모델 fitting을 하기 위한 최소 필요한 샘플링 수가 많아지고 그만큼 반복 수행을 많이 해야 원하는 $$ p $$ 의 확률로 `inlier` 데이터를 뽑아낼 수 있습니다. 이러한 이유로 $$ N $$ 이 커지게 됩니다.


<br>

## **RANSAC의 장단점**

<br>

<br>

## **Early Stop 사용법**

<br>

<br>

## **RANSAC Python Code**

<br>


<br>

## **Lo-RANSAC 개념**

<br>

<br>

## **Lo-RANSAC Python Code**

<br>


<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-concept-table/)

<br>
