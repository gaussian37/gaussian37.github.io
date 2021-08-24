---
layout: post
title: 기초 베이지안 이론 (Basic Bayesian Theory)
date: 2020-08-09 00:00:00
img: ml/concept/basic_bayesian_theory/0.png
categories: [ml-concept] 
tags: [machine learning, probability model, 베이지안, bayesian, bernoulli, binomial, multinomial, conjugate, bayes update] # add tag
---

<br>

[머신 러닝 관련 글 목록](https://gaussian37.github.io/ml-concept-table/)

<br>

- 사전 지식 1 : [https://gaussian37.github.io/ml-concept-mle-and-map/](https://gaussian37.github.io/ml-concept-mle-and-map/)
- 사전 지식 2 : [https://gaussian37.github.io/ml-concept-mlemap](https://gaussian37.github.io/ml-concept-mlemap)
- 사전 지식 3 : [https://gaussian37.github.io/ml-concept-probability_model/](https://gaussian37.github.io/ml-concept-probability_model/)
- 사전 지식 4 : [https://gaussian37.github.io/ml-concept-basic_information_theory/](https://gaussian37.github.io/ml-concept-basic_information_theory/)
- 사전 지식 5 : [https://gaussian37.github.io/ml-concept-infomation_theory/](https://gaussian37.github.io/ml-concept-infomation_theory/)

<br>

## **목차**

<br>

- ### 베르누이 분포
- ### 베르누이 분포와 MLE
- ### MLE의 가정과 문제점
- ### 베이지안 이론
- ### Conjugate Prior

<br>

- 이번 글에서는 동전 던지기 실행을 통해 베이지안 이론을 접근을 하려고 합니다. 동전 던지기 실험을 알아보기 위하여 먼저 `베르누이 분포`에 대하여 알아보도록 하겠습니다.

<br>

## **베르누이 분포**

<br>

- `베르누이 분포`란 확률 이론 및 통계학에서 자주 사용되는 분포로서 동전 전지기의 앞면 또는 뒷면, 입시의 합격과 불합격, 사업의 성공과 실패, 수술 후 환자의 치유 여부 등, **어던 실험이 두 가지 가능한 결과만을 가질 경우** 이를 표현하는 확률 모형입니다.

<br>

- $$ \text{Bernoulli RV} X \in \{0, 1 \} \tag{1} $$

- $$ P(X = 1) = p \tag{2} $$

- $$ f(x) = p^{x}(1-p)^{1-x} \tag{3} $$

<br>
<center><img src="../assets/img/ml/concept/basic_bayesian_theory/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>

## **베르누이 분포와 MLE**

<br>
<center><img src="../assets/img/ml/concept/basic_bayesian_theory/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 동전 던지기 실험을 하였을 때, 위 그림처럼 앞/뒤/뒤/뒤/뒤 와 같은 결과를 얻었다고 가정하겠습니다.
- 관측된 데이터로부터, 같은 동전을 던질 때, 앞면이 나타날 확률은 `Likelihood`를 이용하여 계산합니다.

<br>

- $$ P(x \vert \theta) = \prod_{i=1}^{5} P(x_{i} \vert \theta) = \theta \cdot (1 - \theta) \cdot (1 - \theta) \cdot (1 - \theta) \cdot (1 - \theta) \tag{4} $$

- $$ P(x \vert \theta) : \text{Likelihood} $$

- $$ \theta : \text{Parameter} $$

<br>

- 식 (4)를 `likelihood` 즉, $$ \theta $$ 에 대한 함수로 보며 **관측된 데이터를 가장 잘 설명하는 파라미터** $$ \theta $$ 를 찾기 위한 `MLE(Maximum Likelihood Estimation)`을 수행하면 다음과 같습니다. MLE`는 **데이터는 고정**이고 **파라미터만 변경**하여 **최적의 파라미터를 찾는 문제**입니다.
- 아래 식은 `MLE`를 통해 `log-likelihood`를 최대화화는 $$ \theta $$ 를 찾는 방법입니다.

<br>

- $$ \theta_{\text{ML}} = \operatorname*{argmax}_\theta l(\theta) \tag{5} $$

- $$ \frac{\partial l(\theta)}{\partial \theta} = \frac{\partial}{\partial \theta}\sum_{i=1}^{5} \log{P(x_{i} ]vert \theta)} \tag{6} $$

- $$ = \frac{\partial}{\partial \theta} \log{\theta \cdot (1 - \theta) \cdot (1 - \theta) \cdot (1 - \theta) \cdot (1 - \theta)} \tag{7} $$

- $$ = \frac{\partial}{\partial \theta} (\log{\theta} + 4\log{(1 - \theta)}) \tag{8} $$

- $$ = \frac{1}{\theta} - \frac{4}{1 - \theta} \tag{9} $$

<br>

- 미분한 결과를 정리한 식 (9)가 0이되는 변곡점에서 식의 최댓값을 가지는 $$ \theta $$를 찾을 수 있습니다.

<br>

- $$ \frac{1}{\theta} - \frac{4}{1 - \theta} = 0 \tag{10} $$

<br>

- 식 (10)을 풀면 다음과 같습니다.

<br>
<center><img src="../assets/img/ml/concept/basic_bayesian_theory/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 즉, 5번 동전을 던졌을 때, 1번 앞면이 나왔으니 앞면이 나올 확률은 1/5일 것이다와 같은 직관적인 생각과 같은 결과가 도출되었습니다.
- 그런데 일반적으로 동전 던지기는 1/2의 확률을 가진다고 알고 있습니다. **5번의 실험을 통해서 관측한 결과와 흔히 알고 있는 동전 던지기를 한 결과의 확률 차이가 크게 나는데 왜 그럴까요?**

<br>

## **MLE의 가정과 문제점**

<br>

- `MLE`는 **asymptotically unbiased** 즉, 점진적으로 bias가 없어지는 성질을 가집니다. 여기서 점진적이라는 말의 뜻은 데이터가 점점 더 관측이 된다는 뜻을 말합니다.
- 만약 무한대의 관측 데이터가 주어질 경우 `MLE`로 예측한 파라미터는 실제 파라미터로 수렴하게 됩니다.
- 하지만 현실적으로 무한대의 관측 데이터를 찾기 어렵고, 제한된 수량의 관측 데이터만 주어진다면 `bias`한 결과를 얻게 됩니다.

<br>

- 앞에서 다룬 동전 던지기 예제에서 과연 5번의 시도만으로 앞면의 확률이 1/5이라고 단정지을 수 있을까요?
- 이 예제에 따르면 `MLE`는 **초기 관측에 쉽게 overfitting하게 됩니다.**
- 극단적으로 동전 던지기를 1번 해서 앞면이 나오면 앞면의 확률이 1이라고 단정지을 수 있을까요?
- 이러한 문제를 개선하기 위하여 `베이지안` 방식으로 접근할 수 있습니다.

<br>

## **베이지안 이론**

<br>



<br>

## **Conjugate Prior**

<br>



<br>




<br>

[머신 러닝 관련 글 목록](https://gaussian37.github.io/ml-concept-table/)

<br>