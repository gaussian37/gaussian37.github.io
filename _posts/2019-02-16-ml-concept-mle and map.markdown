---
layout: post
title: ML(Maximum Likelihood)와 MAP(maximum a posteriori)
date: 2019-02-16 00:00:00
img: ml/concept/mle-and-mlp/mle.jpg
categories: [ml-concept] 
tags: [ml, machine learning, 머신 러닝, mle, map, 우도, 사전확률, 사후확률] # add tag
---

+ 출처 : 패턴인식(오일석)
+ 이번 글에서는 MLE(Maximum Likelihood Estimation)와 MAP(maximum a posteriori)에 대하여 알아보도록 하겠습니다.


## ML과 MAP의 개념적 설명

+ 먼저 ML은 개념적으로 어떠한 형태의 분포에도 적용 가능합니다.
    + 현실적으로는 정규 분포와 같이 매개 변수로 표현되는 경우에만 적용 가능한 데 매개 변수로 표시 된 경우만 계산이 가능하기 때문입니다.
+ 이 매개 변수 집합을 $$ \theta $$ 라고 하겠습니다.
+ 이 때, 문제를 다음과 같이 정의 할 수 있습니다.
    + **주어진 X를 발생시켰을 가능성이 가장 높은 $$\theta$$를 찾아라**
    + **X에 대하여 가장 큰 likelihood를 갖는 $$\theta$$를 찾아라**  
+ <img src="../assets/img/ml/concept/mle-and-mlp/mleGraph.PNG" alt="Drawing" style="width: 300px;"/>
    + 위 그래프에서 X는 6개의 샘플을 갖습니다.
    + 이 X를 발생시킬 가능성은 $$ \theta_{1} $$이 $$ \theta_{2} $$ 보다 높습니다.
    + likelihood를 이용하여 다시 표현하면 $$ p(X \vert \theta_{1}) > p(X \vert \theta_{2}) $$ 라고 할 수 있습니다.  
    + 이 때 우리가 풀어야 할 문제는 **어떤 $$ \theta $$가 maximum likelihood**를 가질까 입니다.

<br>

+ maximum likelihood를 이해하기 위해 다음과 같은 예제를 살펴보겠습니다.
+ 3가지의 동전이 있습니다. 동전의 앞이 나올 확률이 p 뒤가 나올 확률이 1-p라고 하고 각각의 p는 1/4, 1/3, 1/2 입니다.
+ 임의의 동전을 하나 집어서 50번 던졌을 때 관찰 결과 22번이 나왔습니다. 이 때, 과연 어떤 동전을 던졌을지 맞추는 것이 문제입니다.
+ 이 문제의 해결법은 각 동전의 likelihood를 구한 다음에 그 값이 최대가 되는 것을 구하는 것입니다.
    + 즉, maximum likelihood를 취하는 것입니다.
+ ·$$ \hat{\theta} = argmax_{\theta} p(X \vert \theta) $$ 를 이용하겠습니다. $$\theta = p$$ 입니다.
    + ·$$ P(head = 22 \vert p = \frac{1}{4}) = \begin{pmatrix} 50 \\ 22 \\ \end{pmatrix} \frac{1}{4}^{22}\frac{3}{4}^{28} = 0.0016 $$
    + ·$$ P(head = 22 \vert p = \frac{1}{3}) = \begin{pmatrix} 50 \\ 22 \\ \end{pmatrix} \frac{1}{3}^{22}\frac{2}{3}^{28} = 0.0332 $$
    + ·$$ P(head = 22 \vert p = \frac{1}{2}) = \begin{pmatrix} 50 \\ 22 \\ \end{pmatrix} \frac{1}{2}^{22}\frac{1}{2}^{28} = 0.0788 $$
+ 따라서 p = 1/2 일 때, likelihood가 가장 크므로 p = 1/2 일 때라고 말할 수 있습니다.
       
<br>    
    
+ 문제를 좀 더 형식적으로 쓰면 다음과 같이 쓸 수 있습니다.
    + ·$$ \hat{\theta} = argmax_{\theta} p(X \vert \theta) $$
+ 확률 분포 추정 문제를 위와 같이 maximum likelihood를 갖는 매개 변수를 찾는 것으로 규정하고 해를 구하는 방법을 **Maximum Likelihood method** 라고 합니다.
+ 모든 샘플은 **독립적으로 추출되었다고 가정**할 수 있으므로 likelihood는 다음과 같이 쓸 수 있습니다.
    + X는 훈련집합으로 $$ X = \{x_{1}, x_{2}, ... , x_{N} \} $$
    + ·$$ p(X \vert \theta) = p(x_{1} \vert \theta)p(x_{2} \vert \theta)...p(x_{N} \vert \theta) = \prod_{i=1}^{N}p(x_{i} \vert \theta) $$
+ ·$$ \hat{\theta} = argmax_{\theta} p(X \vert \theta) $$를 좀더 쉽게 표현해 보겠습니다.
    + f()가 단조 증가 함수라면 $$ argmax_{\theta} p(X \vert \theta) $$ 에서 $$ P(X \vert \theta) $$를 최대화 하는 것과 $$ f(p(X \vert \theta)) $$를 최대화 하는 것은 같습니다.
    + likelihood에 단조 증가 함수인 ln을 취한 것을 `log likelihood` 라고 합니다.
    + ·$$ \hat{\theta} = argmax_{\theta}\sum_{i=1}^{N} p(x_{i} \vert \theta) $$
        + 위 식은 최적화 문제에 해당합니다.
+ 최적화 문제를 풀기 위해서는 미분을 한 결과가 0이 되는 것을 이용하겠습니다.
    + ·$$ \partial\frac{L(\theta)}{\theta} = \partial\frac{\sum_{i=1}^{N}ln p(x_{i} \vert \theta)}{\theta} $$
+ 추정하고자 하는 확률 분포가 정규 분포를 따른다고 가정하면 풀이는 쉬워 집니다.
+ 이 가정에 따르면 $$ \theta = {\mu, \Sigma} $$ 입니다.(평균과, 공분산을 뜻합니다.)
    + 즉, $$ p(x) = N(\mu, \Sigma) $$


