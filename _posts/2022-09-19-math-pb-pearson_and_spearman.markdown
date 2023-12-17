---
layout: post
title: Pearson Correlation Coefficient와 Spearman Correlation Coefficient
date: 2022-09-19 00:00:00
img: math/pb/pearson_and_spearman/0.png
categories: [math-pb] 
tags: [pearson correltation coefficient, spearman correlation coefficient] # add tag
---

<br>

[통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

- 참조 : [분산과 공분산 (variance and covariance)](https://gaussian37.github.io/math-pb-variance_covariance/)
- 참조 : [피어슨 상관 계수, 위키피디아](https://ko.wikipedia.org/wiki/%ED%94%BC%EC%96%B4%EC%8A%A8_%EC%83%81%EA%B4%80_%EA%B3%84%EC%88%98)
- 참조 : [스피어먼 상관 계수, 위키피디아](https://ko.wikipedia.org/wiki/%EC%8A%A4%ED%94%BC%EC%96%B4%EB%A8%BC_%EC%83%81%EA%B4%80_%EA%B3%84%EC%88%98)

<br>

- 이번 글에서는 **두 변수의 상관관계**를 측정하는 대표적인 방식인 `Pearson Correlation Coefficient`와 `Spearman Correlation Coefficient`에 대하여 살펴보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [Pearson Correlation과 Spearman Correlation의 비교](#pearson-correlation과-spearman-correlation의-비교-1)
- ### [Pearson Correlation의 정의와 예제](#pearson-correlation의-정의와-예제-1)
- ### [Spearman Correlation의 정의와 예제](#spearman-correlation의-정의와-예제-1)
- ### [Pearson Correlation과 Spearman Correlation의 수치 비교](#pearson-correlation과-spearman-correlation의-수치-비교-1)

<br>

## **Pearson Correlation과 Spearman Correlation의 비교**

<br>

<br>

## **Pearson Correlation의 정의와 예제**

<br>

- 먼저 `Pearson Correlation`에 대하여 살펴보도록 하겠습니다. 살펴볼 순서는 `Pearson Correlation`의 정의와 식의 의미이며 간단한 예시도 살펴보도록 하겠습니다.

<br>

#### **Pearson Correlation의 정의**

<br>


<br>

#### **Pearson Correlation의 범위가 -1과 1사이인 이유**

<br>


<br>

#### **Pearson Correlation이 선형 관계만 설명 가능한 이유**

<br>

- 아래 식에서 표본의 갯수는 $$ n $$ 개이며 $$ \overline{X}, \overline{Y} $$ 는 각각 $$ X, Y $$ 변량에 대한 표본의 평균을 의미합니다. $$ s_{x}, s_{y} $$ 는 `표본 표준편차`를 의미합니다.
- 아래 식에서 사용된 $$ 1/(n-1) $$ 은 `표분분산의 평균이 모분산과 같아져야 하기 때문`에 적용한 것이며 상세 내용은 아래 링크에서 확인가능합니다.
    - 링크 : [표본분산에서 n-1로 나누어 주는 이유](https://gaussian37.github.io/math-pb-sample_covariance_n-1/)

<br>

- $$ \begin{align} \rho_{(X, Y)} = \frac{\text{COV}(X, Y)}{\sqrt{\text{VAR}(X)}\sqrt{\text{VAR}(Y)}} &= \frac{\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})(Y_{i}-\overline{Y})}{\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}}} \\ &= \frac{\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})(Y_{i}-\overline{Y})}{s_{x}s_{y}} \\ &= \frac{1}{n-1}\sum_{i=1}^{n} \frac{(X_{i}-\overline{X})}{s_{x}}\frac{(Y_{i}-\overline{Y})}{s_{y}} \\ &= \frac{1}{n-1}\sum_{i=1}^{n} Z_{X_{i}}Z_{Y_{i}} \end{align} $$

<br>

- 위 식에서 $$ Z_{X_{i}}, Z_{Y_{i}} $$ 각각은 `Standardization`이 적용된 상태의 변량이 됩니다. 각 변량은 표준 정규 분포를 따르도록 `Standardization`가 적용된 것이기 때문에 각 변량의 스케일에 상관없이 모두 평균이 0, 표준편차는 $$ s_{x}, s_{y} $$ 를 따르도록 형성됩니다.
- 이 때, $$ Z_{X_{i}}, Z_{Y_{i}} $$ 가 같이 양의 값 방향으로 증가하거나 음의값 방향으로 감소해야 두 값을 곱하였을 때, 큰 양수 값이 되며 평균을 내었을 때, 1에 가까운 값이 됩니다.
- 반면 $$ Z_{X_{i}}, Z_{Y_{i}} $$ 값 중 한 값이 증가할 때, 나머지 값은 감소하게 되면 두 값을 곱하였을 때, 큰 음수 값이 되며 평균을 내었을 때, -1에 가까운 값이 됩니다.
- 만약 한 값이 증가하더라도 나머지 한 값이 뚜렷한 증감 없이 평균에 가까운 0에 머물게 되는 경우가 많이 생기게 된다면 평균을 내었을 때, 0에 가까운 값이 됩니다. 이러한 이유로 `Pearson Correlation`에서는 두 변수의 증감에 대한 상관관계가 없으면 0에 가까운 값을 얻게 됩니다.
- 앞에서 설명한 이유로 두 변량의 양의 상관관계가 클수록 1에 가까운 값을 얻게 되고 음의 상관관계가 클수록 -1에 가까운 값을 얻게 됩니다.

<br>

- `Pearson Correlation`은 두 변량 간의 선형관계를 나타내는 특성을 가지고 있습니다. 앞에서 `Standardization`을 하는 과정에서 각 변량에 평균을 빼고 표본 표준편차를 나누어 주는 과정을 통해 두 변량의 스케일을 맞춰준 후 곱하게 됩니다. 이 `Standardization`이 선형 변환을 적용하기 때문에 `Pearson Correlation`은 **두 변량의 선형 관계만 설명할 수 있습니다. 비선형적인 관계라면 관계성이 있다고 하더라도 낮은 correlation 수치를 얻을 수도 있습니다.** 이러한 점이 `Pearson Correlation`의 특징이면서 단점으로 작용하기도 합니다.
- 뿐만 아니라 몇 개의 데이터 쌍이 이상치가 되어 노이즈 처럼 작용된다면 비록 대부분의 데이터가 선형 관계를 가지더라도 `Pearson Correlation`을 계산할 때, 악영향을 끼칠 수 있습니다. 즉, **`Pearson Correlation`은 노이즈에 취약하다는 단점**이 있습니다. 관련 내용은 아래 예시를 살펴보면서 확인해 보도록 하겠습니다.
- 상세 내용 및 추가 해석등은 다음 위키피디아에서 참조할 수 있습니다.
    - 링크 : https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Interpretation

<br>

#### **Pearson Correlation의 예시**

<br>

<br>

## **Spearman Correlation의 정의와 예제**

<br>

- 앞에서 살펴본 `Pearson Correlation`의 경우 두 변량의 `선형 관계 정도`를 -1에서 1 사이의 범위로 나타냄을 확인할 수 있었습니다. 그리고 `Pearson Correlation`의 정의에 따라서 오직 선형 관계만 설명할 수 있다는 단점과 소수의 노이즈 데이터 쌍에도 상관계수가 영향을 받을 수 있음을 확인하였습니다.
- 이러한 문제점을 개선하기 위하여 데이터의 분산을 이용하는 방법이 아닌 `rank`를 이용하는 방법에 대하여 살펴보도록 하겠습니다.

<br>

#### **Spearman Correlation의 정의**

<br>

- 먼저 `Spearman Correlation`의 정의는 다음과 같습니다. 

<br>

- $$ r_{(X, Y)} = 1 - \frac{6 \sum_{i=1}^{n}(d_{i}^{2})}{n(n^{2}-1)} $$

- $$ d_{i} = R(X_{i}) - R(Y_{i}) $$

- $$ R(X_{i}) : \text{Ranking order of X_i among all X data} $$

- $$ R(Y_{i}) : \text{Ranking order of Y_i among all Y data} $$

<br>

- 위 식에서  $$ r_{(X, Y)} $$ 의 값의 범위는 `Pearson Correlation`과 동일하게 -1 에서 1의 범위를 가지게 되고 그 의미 또한 1에 가까울 수록 양의 상관관계가 높고 -1에 가까울수록 음의 상관관계가 높습니다.
- `Pearson Correlation`과의 차이점에 대하여 살펴보면 $$ d_{i} = R(X_{i}) - R(Y_{i}) $$ 에 있습니다. `Pearson Correlation`은 `Standardization` 과정으로 전처리한 값을 사용하는 반면에 `Spearman Correlation`은 `ranking`을 이용합니다. 즉, 가지고 있는 데이터 셋에서 크기 순서의 인덱스 번호를 값으로 대체합니다. 예를 들어 $$ X_{1} = 100, X_{2} = -10, X_{3} = 3 $$ 이면 $$ R(X_{1}) = 1, R(X_{2}) = 3, R(X_{3}) = 2 $$ 가 됩니다.
- 이와 같은 방법으로 `ranking`을 이용하면 `Pearson Correlation`에서 발생한 노이즈에 대한 영향을 줄일 수 있습니다. 뿐만 아니라 `Pearson Correlation`에서는 `선형 관계`만 설명 가능하였지만 `Spearman Correlation`는 `ranking`으로 표현 가능한 단조 증가 관계 (`monotonic relationship`)는 모두 설명 가능합니다.

<br>

#### **Spearman Correlation 식 유도 과정**

<br>


<br>

## **Pearson Correlation과 Spearman Correlation의 수치 비교**

<br>

<br>

<br>

[통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>