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

- 아래 식에서 표본의 갯수는 $$ n $$ 개이며 $$ \mu_{x}, \mu_{y} $$ 는 각각 $$ X, Y $$ 변량에 대한 표본의 평균을 의미합니다. $$ s_{x}, s_{y} $$ 는 `표본 표준편차`를 의미합니다.
- 아래 식에서 사용된 $$ 1/(n-1) $$ 은 `표분분산의 평균이 모분산과 같아져야 하기 때문`에 적용한 것이며 상세 내용은 아래 링크에서 확인가능합니다.
    - 링크 : [표본분산에서 n-1로 나누어 주는 이유](https://gaussian37.github.io/math-pb-sample_covariance_n-1/)

<br>

- $$ \begin{align} \frac{\text{COV}(X, Y)}{\sqrt{\text{VAR}(X)}\sqrt{\text{VAR}(Y)}} &= \frac{\frac{1}{n-1}\sum_{i=1}^{n}(x_{i}-\mu_{x})(y_{i}-\mu_{y})}{\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_{i}-\mu_{x})^{2}}\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_{i}-\mu_{x})^{2}}} \\ &= \frac{\frac{1}{n-1}\sum_{i=1}^{n}(x_{i}-\mu_{x})(y_{i}-\mu_{y})}{\s_{x}\s_{y}} \\ &= \frac{1}{n-1}\sum_{i=1}^{n} \frac{(x_{i}-\mu_{x})}{\s_{x}}\frac{(y_{i}-\mu_{y})}{\s_{y}} \\ &= \frac{1}{n-1}\sum_{i=1}^{n} z_{x_{i}}z_{y_{i}} \end{align} $$

<br>

- 위 식에서 $$ z_{x_{i}}, z_{y_{i}} $$ 각각은 `Standardization`이 적용된 상태의 변량이 됩니다. 각 변량은 표준 정규 분포를 따르도록 `Standardization`가 적용된 것이기 때문에 각 변량의 스케일에 상관없이 모두 평균이 0, 표준편차는 $$ \s_{x}, \s_{y} $$ 를 따르도록 형성됩니다.
- 이 때, $$ z_{x_{i}}, z_{y_{i}} $$ 가 같이 양의 값 방향으로 증가하거나 음의값 방향으로 감소해야 두 값을 곱하였을 때, 큰 양수 값이 되며 평균을 내었을 때, 1에 가까운 값이 됩니다.
- 반면 $$ z_{x_{i}}, z_{y_{i}} $$ 값 중 한 값이 증가할 때, 나머지 값은 감소하게 되면 두 값을 곱하였을 때, 큰 음수 값이 되며 평균을 내었을 때, -1에 가까운 값이 됩니다.
- 만약 한 값이 증가하더라도 나머지 한 값이 뚜렷한 증감 없이 평균에 가까운 0에 머물게 되는 경우가 많이 생기게 된다면 평균을 내었을 때, 0에 가까운 값이 됩니다. 이러한 이유로 `Pearson Correlation`에서는 두 변수의 증감에 대한 상관관계가 없으면 0에 가까운 값을 얻게 됩니다.
- 앞에서 설명한 이유로 두 변량의 양의 상관관계가 클수록 1에 가까운 값을 얻게 되고 음의 상관관계가 클수록 -1에 가까운 값을 얻게 됩니다.

<br>

- `Pearson Correlation`은 두 변량 간의 선형관계를 나타내는 특성을 가지고 있습니다. 앞에서 `Standardization`을 하는 과정에서 각 변량에 평균을 빼고 표본 표준편차를 나누어 주는 과정을 통해 두 변량의 스케일을 맞춰준 후 곱하게 됩니다. 이 `Standardization`이 선형 변환을 적용하기 때문에 `Pearson Correlation`은 **두 변량의 선형 관계만 설명할 수 있습니다. 비선형적인 관계라면 관계성이 있다고 하더라도 낮은 correlation 수치를 얻을 수도 있습니다.** 이러한 점이 `Pearson Correlation`의 특징이면서 단점으로 작용하기도 합니다.
- 상세 내용 및 추가 해석등은 다음 위키피디아에서 참조할 수 있습니다.
    - 링크 : https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Interpretation

<br>

## **Spearman Correlation의 정의와 예제**

<br>

<br>

## **Pearson Correlation과 Spearman Correlation의 수치 비교**

<br>

<br>

<br>

[통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>