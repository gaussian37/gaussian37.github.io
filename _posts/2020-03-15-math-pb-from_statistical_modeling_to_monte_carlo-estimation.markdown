---
layout: post
title: Statistical modeling, Bayesian modeling, Monte carlo estimation, Markov chain.
date: 2020-03-15 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [Statistical modeling, Bayesian modeling, Monte carlo estimation, Markov chain.] # add tag
---

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

- 이번 글에서는 `Statistical modeling`, `Bayesian modeling`, `Monte carlo estimation` 그리고 `Markov chain` 까지 연계해서 다루어보도록 하겠습니다.

<br>

## **목차**

<br>

- ### Statistical model이란?
- ### Modeling process
- ### Bayesian modeling

<br>

### **Statistical model이란?**

<br>

- `statistical model`이 무엇일까요?  statistical model은 **데이터 생성 과정을 모델링** 한 것입니다. statistical model은 다양한 변수들의 관계에 대하여 설명하는데 이 변수들은 데이터 속에서 매우 다양하게 나타나고 불확실하게 나타나는 특성을 가집니다. 
- 예를 들어 변수들이 너무 다양하거나 변수들의 관계가 너무 복잡하면 변수들 간의 관계를 알아차리기가 어려워 random behavior 형태로 데이터가 나타날 수 있습니다. 
- 이러한 random behavior들의 불확실성과 다양성을 나타내기 위해서는 `확률 이론`을 도입해서 나타낼 수 있습니다. 이 때 사용하는 것이 바로 `statistical model`이 됩니다.
- `statistical model`의 첫번째 목적은 **불확실성의 정량화(quantify uncerntainty)**입니다.
    - 예를 들어 투표를 하였을 때, 투표율이 57% 라고 한다면 이 데이터를 정확히 믿을 수 있을까요? 어떤 집단에서 어떻게 뽑았느냐에 따라서 데이터의 정합성이 달라질 수 있습니다. 만약 99%의 신뢰도로 (51%, 63%) 범위의 투표율을 가진다고 표현한다면 좀 더 적합해 보입니다. 즉, 불확실성에 대한 수치를 정량화 함으로써 막연한 데이터를 좀 더 수치적으로 표현할 수 있고 좀 더 많은 정보를 얻을 수 있습니다.
- `statistical model`의 두번째 목적은 **어떤 가정에 대한 근거를 마련**하기 위함입니다.
    - 예를 들어 어떤 후보에 대한 지지율이 여성들은 55%, 남성들은 59%이므로 남성들이 더 선호한다 와 같이 근거를 마련하는 데 사용됩니다. 
- `statistical model`의 세번째 목적은 **예측(prediction)**입니다. 사실 이 목적으로 저희는 statistical model을 많이 사용합니다. 
    - 다양한 변수들을 조합하였을 때, 어떤 후보의 지지율이 가장 높을 지 예측 하거나 어떤 변수들을 조합하였을 때, 이런 패턴의 투표자들은 어떤 후보를 뽑을 지 등등 다양한 예측을 하는데 statistical model이 사용됩니다. 특히 이러한 방법들을 machine learning 으로 사용되는데 machine learning의 주 목적인 prediction을 하는데 model이 사용됩니다.

<br>

### **Modeling process**

<br>

- 그러면 `statistical model`을 어떻게 모델링 하는 지 스텝 별로 차례대로 알아보도록 하겠습니다.
- **① Understand the problem**
    - 모델을 정의하기 전에 분석해야 할 문제를 정확히 이해해야 합니다. 예를 들어 여러 데이터를 모았는데 단위가 다르다면 모델링이 시작부터 잘못될 수도 있습니다.
- **② Plan & Collect data**
    - 어떻게 데이터를 모으고 얼마나 많이 모을 지 계획해야 합니다. 어떤 계획을 세워서 데이터를 모으느냐에 따라서 데이터의 질이 달라집니다. 특히 데이터가 **다양한 케이스를 모두 포함시킬 수 있도록 하려면** 계획을 잘 세우지 않으면 특정 케이스만 모이게 될 가능성이 높습니다.
- **③ Explore your data**
    - 수집한 데이터를 시각화하여 데이터를 이해하는 단계입니다. 이 단계에서는 데이터에서 나타나는 변수들 간의 관계에 대하여 살펴보고 어떤 statistical model을 사용할 지 결정하기 위한 정확한 데이터 이해를 해야합니다.
- **④ Postulate model**
    - 앞에서 데이터가 어떤 지 이해하였다면, 데이터를 잘 표현할 수 있는 모델을 선정해야합니다. 모델을 선정할 때에는 model complexity와 model generalizability의 tradeoff(`bias - variance tradeoff`)를 고려하여 선정해야 하는데 정확한 데이터에 대한 이해가 있으면 선정하는 데 도움이 됩니다. 즉, 복잡한 모델은 overfitting 문제를 만들 수 있는데 어느 정도 복잡한 모델을 사용할 지에 대한 기준은 데이터를 통해 정해야 합니다.
- **⑤ Fit model**
    - 이 단계에서는 모델에서 사용되는 파라미터들을 추정해야 합니다. 이 글에서는 bayesian 방법으로 파라미터 추정을 할 예정입니다.
    - 정확히 이 단계에서 **Bayesian과 Frequentist의 패러다임이 나뉘게** 됩니다.
- **⑥ Check model**
    - 선정한 모델과 파라미터 추정이 잘 되었는 지 확인하는 단계입니다. 적절한 metric을 이용하여 얼마나 데이터에 대한 일반화 성능이 좋은지 확인합니다.
- **⑦ Iterate ④, ⑤, ⑥**
    - 더 좋은 성능을 내기 위하여 모델을 다시 선택하거나 (④) 파라미터 추정(⑤)을 다시해본 뒤 성능 측정(⑥)을 하는 과정을 반복합니다.
- **⑧ Use model**
    - 수집한 데이터에 가장 적합한 모델과 파라미터를 찾았다면 이 모델을 사용합니다. 

<br>

- 지금까지 `statistical modeling`이 무엇이고 어떤 과정을 거치는 것인 지 알아보았습니다.
- 앞에서 설명하던 중 `Fit model`을 할 때에 Bayesian 패러다임을 사용할 것이라고 말하였는데 지금부터 Bayesian modeling에 대하여 간략히 알아보도록 하겠습니다.

<br>

## **Components of Bayesian models**

<br>

- 이번에는 주제를 바꿔서 `bayesian modeling`에 대하여 다루어 보도록 하겠습니다. 쉽게 예를 들어서 설명해 보겠습니다.
- 15명의 사람이 있고 이 사람들의 키가 정규 분포를 따른다고 가정해 보겠습니다. 그러면 다음 식처럼 분포를 정의할 수 있습니다.

<br>

$$ n = 15, \ \ \ \  y_{i} = \mu + \epsilon_{i}, \ \ \ \  \epsilon_{i} \sim N(0, \sigma^{2}) \dots ( 	ext{i.i.d}) $$

$$ y_{i} \sim N(\mu, \sigma^{2}) \dots ( 	ext{i.i.d}) $$

<br>

- 위 식을 이용할 때, 여기 까지는 frequentist와 bayesian의 접근 방법이 같습니다.
- 여기 까지 `frequentist`의 접근 방법은 $$ \mu $$와 $$ \sigma $$를 알기 위해서 가지고 있는 표본을 이용하여 이 값들을 구하고 만약 샘플링을 다시 한다면 다시 계산하여 이 값을 구하는 방식을 이용합니다. 일반적인 사람들이 많이 사용하는 방법입니다.
- 반면 `bayesian`의 접근 방법은 $$ \mu $$와 $$ \sigma $$를 각각 고유한 확률 분포를 가지는 random variable로 다루는 방식입니다. 이 변수들을 각각 다룰 때, 이것을 `prior` 라고 합니다.
- 베이지안에는 크게 3가지 요소인 `likelihood`, `prior`, `posterior`가 있습니다.
- `likelihood`는 **데이터에 대한 확률적 모델의 값**입니다. `데이터에 대한` 이란 의미에서는 데이터를 나타내는 파라미터 $$ \theta $$로 표현할 수 있고 `확률적 모델`에서 확률 모델을 $$ y $$ 라고 한다면 (확률 모델은 정규분포가 될 수도 있고 이항 분포가 될수도 있고 아니면 다양한 다른 다양한 분포가 될 수 있습니다.) `likelihood`는 다음과 같습니다.

<br>

$$ \text{likelihood} = p(y \vert \theta) $$

<br>

- `prior`는 **데이터의 확률**입니다. 앞에서 frequentist의 접근 방법에서 $$ \mu $$와 $$ \sigma $$ 값을 얻을 때, 단순히 주어진 데이터를 통해서 구한 것과 의미가 같습니다. frequentist의 방법에서는 $$ \mu $$와 $$ \sigma $$의 값이 목적이지만 `prior`는 그런 데이터가 나올 확률이 목적입니다.

<br>

$$ \text{prior} = p(\theta) $$

<br>

- **데이터가 나올 확률**과 **그 데이터에 대한 확률적 모델**을 연관시켜서 보면 joint distribution 형태로 나타낼 수 있습니다.

<br>

$$ p(y, \theta) = p(\theta)p(y \vert \theta) $$

<br>

- `posterior`는 likelihood와 반대로 **확률적 모델의 값에 대한 데이터의 확률**입니다. 이 값은 앞에서 정의한 joint distribution을 이용하여 다음과 같이 표현할 수 있습니다.

<br>

$$ p(\theta \vert y) = \frac{p(\theta, y)}{p(y)} = \frac{p(\theta, y)}{\int p(\theta, y)d\theta} = \frac{p(y \vert \theta)p(\theta)}{\int p(y \vert \theta)p(\theta)d\theta} $$

<br>

- 위 식을 보면 조건부 확률로 분모가 $$ p(y) $$가 되어 즉, 확률적 모델의 값이 $$ p(y) $$가 되었을 때의 경우로 한정이 되고 그 때, 데이터 $$ \theta $$가 나올 확률을 나타냅니다.
- 또한 위 식의 $$ \int $$는 일반화 시키기 위해 사용한 식의 기호이며 이산 확률 변수의 경우 $$ \sum $$으로 대체해도 됩니다. 

<br>

## **Model specification**

<br>

- 모델을 fitting 하기 전에 먼저 모든 구성 요소를 지정해야 합니다. 이를 위한 한 가지 편리한 방법은 모델을 계층적 형식(hierarchical form)으로 표현해 보는 것입니다. 즉, 계층별로 모델이 레이어(단계)에 지정되어 있음을 의미합니다.
- 앞에서 다룬 키에 관한 예제를 이용해서 다시 살펴보도록 하겠습니다.
- 먼저 random variable인 $$  y_{i} \vert \mu, \sigma^{2} $$에 대하여 알아보겠습니다.

<br>

$$ y_{i} \vert \mu, \sigma^{2} \ \sim \ N(\mu, \sigma^{2}) \dots (\text{i.i.d}), \ \ \ i = 1, 2, ..., n $$

<br>

- 그 다음 `prior`에 대하여 알아보겠습니다. 이 때, $$ \mu $$와 $$ \sigma $$는 독립이라고 가정하겠습니다. 평균과 분산을 독립이라고 가정하였기 때문에 분리해서 표현할 수 있습니다.

<br>

$$ p(\mu, \sigma^{2}) = p(\mu)p(\sigma^{2}) $$

<br>

- 그리고 $$ \sigma^{2} $$의 값을 알면 $$ \mu $$ 이전의 conjugate가 `정규 분포`이고, $$ \mu $$가 알려져있을 때 $$ \sigma^{2} $$ 이전의 conjugate가 inverse gamma distribution이라는 것이 알려져 있습니다.

<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

