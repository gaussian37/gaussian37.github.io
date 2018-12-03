---
layout: post
title: Bayesian Decision Theory  
date: 2018-11-29 08:42:00
img: ml/concept/bayesian-dicition-theory/Bayes_Theorem_web.png
categories: [ml-concept] 
tags: [Bayesian, Bayesian Decision] # add tag
---

이번 글에서는 Bayesian Decision Theory에 대하여 알아보도록 하겠습니다.
자세히 알아보기 전에 간단한 확률 통계 이론 부터 시작해보도록 하겠습니다.
이 글은 오일석 교수님의 패턴인식을 참조하였습니다. 책이 너무 좋습니다 ㅠㅠ

자 그러면 시작해 보겠습니다.

## 확률 기초

![1](../assets/img/ml/concept/bayesian-dicition-theory/1.png)

<br>

### Discrete & Continuous Probability

+ 주사위를 던졌을 때 3이 나올 확률(probability)는 $$ \frac{1}{6} $$ 입니다.
    + 이것을 기호로 표시하면 $$ P(X = 3) = \frac{1}{6} $$
    + 여기서 X는 랜덤 변수(`random variable`) 라고 합니다.
    + X는 `discrete` 값을 가집니다.

+ 사람 키를 생각해보면 주사위와는 조금 다릅니다. `continuous` 한 값을 가집니다.
    + 이 때 확률 분포는 `PDF`로 나타냅니다. 
        + PDF = probability density function 입니다.
        
### Basic Bayes Rule

![2](../assets/img/ml/concept/bayesian-dicition-theory/2.png)

간단하게 Bayes Rule을 게임을 통하여 알아보도록 하겠습니다.
주머니에는 A 카드 7장과 B카드 3장이 있습니다. A, B 상자에는 각각 하얀공과 파란 공이 있습니다.
예상 되다 싶이, 먼저 주머니에서 카드 한장을 뽑고, 뽑은 카드에 맞는 상자에서 공을 뽑습니다.
공의 색을 확인한 뒤 다시 집어 넣습니다. 복원 하는 것이죠!

자 그러면 문제를 풀 때 먼저 `랜덤 변수`를 먼저 정해야 합니다.

+ 랜덤 변수 X : 주머니에서 뽑은 용지
    + 이 때, $$ X \in \{A, B\} $$
+ 랜덤 변수 Y : 상자에서 뽑은 공
    + 이 때, $$ Y \in \{하얀, 파란\} $$
    
+ 이 때 상자 A가 선택될 확률은 어떻게 표현할 수 있을까요?
    + 확률은 $$ P(X = A) = P(A) = \frac{7}{10} $$ 로 표현할 수 있습니다.

+ 상자 A에서 하얀 공이 뽑힐 확률은 어떻게 될까요? 주머니에서 A라 쓰인 용지를 뽑았다는 조건 하에 확률을 따져보겠습니다.
    + 조건하에 따진다고 하여 조건부 확률(`conditional probability`) 라고 합니다.
    + 확률은 $$ P(Y= 하얀 \| X = A) = P(하얀 \| A) = \frac{2}{10} $$
    
+ 상자는 A 이고 공은 하얀공이 뽑힐 확률 P(A, 하얀)은 얼마일까요? 두 가지 서로 다른 사건이 동시에 일어날 확률입니다.
    + 이 경우 `joint probability` 라고 합니다. `joint probability`는 두 확률의 곱으로 구합니다.
    + 다음 계산 법을 `product rule` 이라고 합니다. $$ P(A, 하얀) = P(하얀 \| A) P(A) = (\frac{2}{10})(\frac{7}{10}) = \frac{7}{50} $$ 입니다.
    
+ 하얀 공이 나올 확률은 어떻게 될까요? 이 때 가능한 경우의 수는 다음과 같습니다.
    + A가 선택되고 하얀 공이 나오는 경우
    + B가 선택되고 하얀 공이 나오는 경우
    + 두 경우의 확률을 더하면 답이 되며 이런 계산 방법을 `sum rule`이라고 합니다. 
        + 이런 방식으로 구한 확률 $$ P(하얀) $$ 을 `marginal probability` 라고 합니다.
        + 이 때, $$ P(하얀) = P(하얀 \| A)P(A) + P(하얀 \| B)P(B) = (\frac{2}{10})(\frac{7}{10}) + (\frac{9}{15})(\frac{3}{10}) = \frac{8}{25} $$

+ 두 랜덤 변수가 서로 영향을 미치지 못하였다면 `independent` 한다고 합니다. 
    + 독립인 두 랜덤 변수는 $$ P(X,Y) = P(X)P(Y) $$ 를 만족해야 합니다. 
    + 위의 예제에서는 $$ P(X,Y) \neq P(X)P(Y) $$ 이므로 독립이 아닙니다.
        + **주머니가 공의 색깔에 영향**을 미치고 있다는 뜻입니다.
        
+ P(A)와 P(B)를 `prior event` 라고 합니다. 
    +확률 실험에서 상자 선택과 공 선택이라는 event가 연이어 일어나는데, 공 선택 전에 상자 선택이 일어나기 때문입니다.

+ 만약 하얀 공이 뽑혔을 때, 하얀 공이 어느 상자에서 나왔는지를 확률적으로 맞추어 보겠습니다.
    + 상자 A의 하얀 공 확률 $$ P(하얀\|A) = \frac{2}{10} $$
    + 상자 B의 하얀 공 확률 $$ P(하얀\|B) =  \frac{9}{15} $$
    + 이렇게 비교할 경우 조건부 확률 $$ P(Y\|X) $$를 사용한 셈이고, Y는 이미 관찰되어 하얀으로 고정되어 있어 X에 따라 확률을 계산합니다.
    + 이 때 조건부 확률을 `likelihood` 라고 합니다. 
        + 하지만 $$ P(하얀 \| A) + P(하얀 \| B) \neq $$ 1 이므로 probability의 개념은 아닙니다.

+ 하지만 likelihood만 고려한다면 주머니에서의 카드 A와 B의 분포가 무시됩니다.
    + 만약 A가 1억장 , B가 1장이라면 likelihood만 고려한 B가 더 가능성이 높다고 할 수 있을까요?
    + `prior event` P(X) 와 `likelihood` $$ P(Y\|X) $$ 모두를 고려해야만 합리적인 선택을 할 수 있습니다.
    
+ 하얀 공이 관찰된 조건 하에서 어떤 상자에서 나왔는지를 알아내야 되는 문제 입니다.
    + P(A | 하얀) vs P(B | 하얀) 을 비교하는 것이 더 맞아 보입니다.
    + 즉, P(X | Y)를 사용하려고 하고 이 때 Y가 고정이 되고 X에 따라 확률을 계산합니다.
    + 이 조건부 확률을 `posterior probability` 라고 합니다.
        + 사건 Y가 일어날 이후에 따지는 확률이므로 `posterior` 라고 불리게 됩니다.
        
+ `posterior probability` $$ P(X\|Y) = \frac{P(Y\|X)P(X)}{P(Y)} = \frac{likelihood \times prior event}{P(Y)} $$

+ P(Y)의 경우 $$ X \in \{A,B\} $$ 이므로 $$ P(Y) = P(Y\|A)P(A) + P(Y\|B)P(B) $$ 가 됩니다.
    + 따라서, $$ P(Y) = \sum_{X} P(Y\|X)P(X) $$ 이 됩니다.
    
+ 이제 `Bayes 정리`를 이용하여 `posterior probability`를 다음과 같이 정리할 수 있습니다.
    + `prior event` : P(X)
        + 주머니에서 카드 A, B를 뽑는 사건에 대한 확률
    + `posterior event` : P(Y)
        + 상자에서 하얀 공을 거낼 확률 : `marginal probability`
        + `marginal probability` = $$ \sum likelihood \times prior\ event $$
        + P(하얀) = P(하얀|A)P(A) + P(하얀|B)P(B) = $$ \frac{2}{10}\frac{7}{10} + \frac{9}{15}\frac{3}{10} = \frac{8}{25} $$
    + `likelihood` : P(Y|X)
        + $$ P(하얀\|A) = \frac{2}{10} $$ : A상자에서 하얀공을 꺼낼 확률
        + $$ P(하얀\|B) = \frac{9}{15} $$ : B상자에서 하얀공을 꺼낼 확률
    + `posterior probability` $$ P(X \| Y) = \frac{likelihood \times prior\ event}{P(Y)} $$
        + 하얀 공이 나왔을 때, A 상자에서 꺼냈을 확률 : $$ P(A\|하얀) = \frac{P(하얀 \| A) \times P(A)}{P(하얀)} = 0.4375 $$
        + 하얀 공이 나왔을 때, B 상자에서 꺼냈을 확률 : $$ P(B\|하얀) = \frac{P(하얀 \| B) \times P(B)}{P(하얀)} = 0.5625 $$
    + `posterior probability`에 따라서 신뢰도 0.5625로 B에서 나왔다고 할 수 있습니다.
        + 이와 같이 `confidence`를 제공할 수 있습니다.