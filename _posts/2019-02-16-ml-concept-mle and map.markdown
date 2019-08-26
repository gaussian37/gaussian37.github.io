---
layout: post
title: ML(Maximum Likelihood)와 MAP(maximum a posterior)
date: 2019-02-16 00:00:00
img: ml/concept/mle-and-mlp/mle.jpg
categories: [ml-concept] 
tags: [ml, machine learning, 머신 러닝, mle, map, 우도, 사전확률, 사후확률] # add tag
---

+ 출처 : 패턴인식(오일석)
+ 이번 글에서는 MLE(Maximum Likelihood Estimation)와 MAP(maximum a posteriori)에 대하여 알아보도록 하겠습니다.

<br>

## **ML(Maximum Likelihood)에 대한 개념적 설명**

<br>

+ 먼저 ML은 개념적으로 어떠한 형태의 분포에도 적용 가능합니다.
    + 현실적으로는 정규 분포와 같이 매개 변수로 표현되는 경우에만 적용 가능한 데 매개 변수로 표시 된 경우만 계산이 가능하기 때문입니다.
+ 이 매개 변수 집합을 $$ \theta $$ 라고 하겠습니다.
+ 이 때, 문제를 다음과 같이 정의 할 수 있습니다.
    + **주어진 X를 발생시켰을 가능성이 가장 높은 $$\theta$$를 찾아라**
    + **X에 대하여 가장 큰 likelihood를 갖는 $$\theta$$를 찾아라**
    
<br>
<center><img src="../assets/img/ml/concept/mle-and-mlp/mleGraph.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>    
      
+ 위 그래프에서 X는 6개의 샘플을 갖습니다.
+ 이 X를 발생시킬 가능성은 $$ \theta_{1} $$이 $$ \theta_{2} $$ 보다 높습니다.
+ likelihood를 이용하여 다시 표현하면 $$ p(X \vert \theta_{1}) > p(X \vert \theta_{2}) $$ 라고 할 수 있습니다.  
+ 이 때 우리가 풀어야 할 문제는 **어떤 $$ \theta $$가 maximum likelihood**를 가질까 입니다.

<br>

+ **Maximum likelihood**를 이해하기 위해 다음과 같은 예제를 살펴보겠습니다.
+ 3가지의 동전이 있습니다. 동전의 앞이 나올 확률이 $$ p $$, 뒤가 나올 확률이 $$1-p$$라고 하고 각각의 $$p$$는 1/4, 1/3, 1/2 입니다.
+ 임의의 동전을 하나 집어서 50번 던졌을 때 관찰 결과 22번이 나왔습니다. 이 때, 과연 어떤 동전을 던졌을지 맞추는 것이 문제입니다.
+ 이 문제의 해결법은 **각 동전의 likelihood를 구한 다음에 그 값이 최대가 되는 것**을 구하는 것입니다.
    + 즉, **maximum likelihood**를 취하는 것입니다.
+ 　$$ \hat{\theta} = argmax_{\theta} \ p(X \vert \theta) $$ 를 이용하겠습니다. $$\theta = p$$ 입니다.
    + 　$$ P(head = 22 \vert p = \frac{1}{4}) = \begin{pmatrix} 50 \\ 22 \\ \end{pmatrix} (\frac{1}{4})^{22}(\frac{3}{4})^{28} = 0.0016 $$
    + 　$$ P(head = 22 \vert p = \frac{1}{3}) = \begin{pmatrix} 50 \\ 22 \\ \end{pmatrix} (\frac{1}{3})^{22}(\frac{2}{3})^{28} = 0.0332 $$
    + 　$$ P(head = 22 \vert p = \frac{1}{2}) = \begin{pmatrix} 50 \\ 22 \\ \end{pmatrix} (\frac{1}{2})^{22}(\frac{1}{2})^{28} = 0.0788 $$
+ 따라서 p = 1/2 일 때, **likelihood**가 가장 크므로 p = 1/2 일 때 **maximum likelihood**라고 말할 수 있습니다.
       
<br>    
    
+ 문제를 좀 더 형식적으로 쓰면 다음과 같이 쓸 수 있습니다.
    + 　$$ \hat{\theta} = argmax_{\theta} \ p(X \vert \theta) $$
+ 확률 분포 추정 문제를 위와 같이 **maximum likelihood**를 갖는 매개 변수를 찾는 것으로 규정하고 해를 구하는 방법을 **Maximum Likelihood method** 라고 합니다.
+ 모든 샘플은 **독립적으로 추출되었다고 가정**할 수 있으므로, 즉 `i.i.d`조건으로 가정하면 likelihood는 다음과 같이 쓸 수 있습니다.
    + X는 훈련집합으로 $$ X = \{x_{1}, x_{2}, ... , x_{N} \} $$ 일 때, $$ p(X \vert \theta) = p(x_{1} \vert \theta)p(x_{2} \vert \theta)...p(x_{N} \vert \theta) = \prod_{i=1}^{N}p(x_{i} \vert \theta) $$
+ 　$$ \hat{\theta} = argmax_{\theta} \ p(X \vert \theta) $$를 좀더 쉽게 표현해 보겠습니다.
    + 예를 들어 $$ f() $$가 단조 증가 함수라면 $$ argmax_{\theta} \ p(X \vert \theta) $$ 에서 $$ P(X \vert \theta) $$를 최대화 하는 것과 $$ f(p(X \vert \theta)) $$를 최대화 하는 것은 같습니다.
    + **likelihood**에 단조 증가 함수인 $$ ln $$을 취한 것을 `log likelihood` 라고 합니다.
    + 　$$ \hat{\theta} = argmax_{\theta}\ \sum_{i=1}^{N} \ ln \ p(x_{i} \vert \theta) $$        
+ 위 식은 최적화 문제에 해당하고 최적화 문제를 풀기 위해서는 미분을 한 결과가 0이 되는 것을 이용하겠습니다.
    + 　$$ \frac{\partial \ L(\theta)}{\partial\theta} = \frac{\partial\sum_{i=1}^{N}ln \ p(x_{i} \vert \theta)}{\partial\theta} = 0 $$
+ 만약 여기서 추정하고자 하는 확률 분포가 `정규 분포를 따른다고 가정`하면 풀이는 쉬워 집니다.
+ 이 가정에 따르면 $$ \theta = {\mu, \Sigma} $$ 입니다.(평균과, 공분산을 뜻합니다.)
    + 즉, $$ p(x) = N(\mu, \Sigma) $$
    + ML 방법에서 어떤 데이터 X가 나오도록 하는 가장 가능성 높은 선택지는 `평균` 입니다. 정규 분포의 확률 분포 곡선을 보면 평균에서 가장 높은 확률값을 가지기 때문입니다.

<br>

+ 아래 내용은 정규 분포를 위한 ML을 추정하는 과정입니다.
+ 여기서 $$X$$가 정규 분포에서 추정되었다고 가정하겠습니다. 수식 유도를 쉽게 하기 위하여 공분산 행렬 $$ \Sigma $$는 이미 알고 있다고 가정하겠습니다.
+ 즉, 추정해야 하는 것은 평균 벡터 $$ \mu $$ 뿐입니다.
+ 　$$ \frac{\partial\sum_{i=1}^{N}ln\ p(x_{i} \vert \theta)}{\partial\theta} $$에 정규 분포 식을 대입하고 정리해 보겠습니다.
+ 아래 식에서 $$ d $$는 특징 벡터 $$ x_{i} $$의 차원 입니다.
+ 　$$ p(x_{i} \vert \theta) = p(x_{i} \vert \mu) = \frac{1}{ (2\pi)^{\frac{d}{2}} \vert \Sigma \vert^{\frac{1}{2}} } exp(-\frac{1}{2}(x_{i} - \mu)^{T}\Sigma^{-1}(x_{i} - \mu)) $$
+ 　$$ ln\ p(x_{i} \vert \mu) = -\frac{1}{2}(x_{i} - \mu)^{T}\Sigma^{-1}(x_{i} - \mu)-\frac{d}{2}ln2\pi -\frac{1}{2}ln\vert \Sigma \vert $$
+ 　$$ L(\mu) = -\frac{1}{2}\sum_{i=1}^{N}(x_{i} - \mu)^{T}\Sigma^{-1}(x_{i} - \mu)-N(\frac{d}{2}ln2\pi -\frac{1}{2}ln\vert \Sigma \vert) $$
+ 　$$ \frac{\partial L(\mu)}{\partial \mu} = \sum_{i=1}^{N} \Sigma^{-1}(x_{i} - \mu) $$
+ 이제 $$ \frac{\partial L(\mu)}{\partial \mu} = 0 $$을 두고 식을 정리해 보겠습니다.
    + 　$$ \sum_{i=1}^{N} \Sigma^{-1}(x_{i} - \mu) = 0 $$
    + 　$$ \sum_{i=1}^{N}x_{i} - N\mu = 0 $$
    + 　$$ \hat{\mu} = \frac{1}{N}\sum_{i=1}^{N}x_{i} $$
        + 이 식으로 구한 평균 벡터는 최적 매개 변수 값이기 때문에 hat 씌워 표시합니다.

+ 위 식은 두가지 정보가 제공된 상황에서 구해졌습니다.
    + 훈련 집합 X 라는 정보
    + 확률 분포가 정규 분포를 따른다는 정보
+ 이 상황에서 샘플의 특징 벡터를 모두 더하고 그것을 N으로 나누어준 값, 즉 샘플의 평균 벡터가 바로 찾고자 하는 최적의 매개 변수가 된다는 직관과 동일 합니다.

<br>

## MAP(Maximum a posterior)에 대한 개념적 설명

<br>

+ 앞에서 ML에 대하여 설명할 때에는 $$ p(\theta) $$가 균일하다는 가정으로 식을 전개하였습니다.
+ 만약 $$ p(\theta) $$에 대한 정보가 주어져서 사용 가능하다면 어떻게 사용할 수 있을까요?
    + 이 경우에 $$ p(\theta) $$ 는 동일하지 않을 수 있습니다. 따라서 이 경우에는 $$ p(\theta) $$ 를 고려하여 최적화 문제를 풀어야 합니다.
+ 식에서 $$ p(x_{i} \vert \theta) $$를 `likelihood` 라고 하고 $$ p(\theta) $$ 를 `사전 확률` 이라고 합니다.
+ 이 때, $$ p(x_{i} \vert \theta)p(\theta) $$ 를 `사후 확률`이라고 합니다.
+ 따라서 이 수식을 풀어 최적의 매개변수를 찾는 과정을 `MAP(Maximum a posterior)` 라고 합니다.

$$ argmax_{\theta} \  \sum ln(p(x_{i} | \theta)) + ln(p(\theta)) $$

<br>
<center><img src="../assets/img/ml/concept/mle-and-mlp/ml-vs-map.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>  

+ 위 그림은 `ML`과 `MAP`를 비교합니다.
+ **ML에서는 사전확률이 균일**하다고 가정합니다. 따라서 **likelihood가 최고인 점**을 찾으면 그것이 바로 최적해 $$ \theta $$가 됩니다.
+ 하지만 MAP 에서는 **사전 확률이 균일하지 않습니다.** 따라서 **사전확률이 최적 해에 영향**을 미치게 됩니다.