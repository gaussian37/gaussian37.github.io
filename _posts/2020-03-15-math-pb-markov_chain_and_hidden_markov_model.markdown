---
layout: post
title: Markov chain 과 Hidden markov model
date: 2020-03-15 00:00:00
img: math/pb/probability.jpg
categories: [math-pb] 
tags: [마코프 체인, Markov chain] # add tag
---

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

- 참조1 : https://youtu.be/Ws63I3F7Moc
- 참조2 : https://youtu.be/kqSzLo9fenk

<br>

- 이번 글에서는 Markov chain에 대하여 알아보고 Hidden Markov Model에 대하서도 다루어 보도록 하겠습니다.

<br>

## **Markov Chain**

<br>

- Markov Chain이 나오게 된 배경은 마코프가 파벨 네크로소프의 주장을 반박하려는 과정 속에서 나오게 되었다고 알려져 있습니다. 정확히는 **큰 수의 법칙을 해석**하는 관점의 차이에서 마코프가 Markov Chain을 도입하였습니다.
- 네크로소프의 주장은 신학생이었는데 운명이 결정되어 있다는 것이 아닌 종교적 자유의지를 중요하게 생각하였었습니다. 즉 사건의 `독립성`에 대하여 상당히 중요하게 생각하였다고 합니다.
- 큰 수의 법칙을 해석하는 네크로소프의 주장은 `큰 수의 법칙`에서 **독립성은 꼭 필요한 조건**이라고 주장하였습니다. 예를 들어 주사위 던지기 같은 사건의 결과가 현재와 미래의 주사위를 던지는 사건의 확률을 변화시키지 않는데 이러한 예시들이 독립성을 통하여 설명할 수 있기 때문입니다.
- 이 관점을 좀 더 확대시켜 보면 세상의 수많은 분포가 가우시안 분포로 되어있고 그 이론적 배경에는 중심 극한 이론이 있습니다.
- 수많은 시행을 거치게 되면(즉, `큰 수의 법칙`을 거치게 되면) 대부분의 분포들이 가우시안 분포를 따르게 되는데 이 시행들은 독립이어야 한다 라고 확장하고 싶었던 것입니다. (그것이 자유의지에 해당하는 종교적 신념 이었기 때문입니다.)

<br>

- 하지만 세상의 많은 일들은 독립적이라기 보다 `종속적`입니다. 대부분의 사건, 날씨 등을 보면 어느 하나 독립적인 것이 잘 없습니다.
- 즉, 현실 세계의 많은 경우와 같이 어떤 사건이 다른 사건에 영향을 주게 되는 사건을 `종속 사건`이라고 합니다.
- 그래서 마코프는 `Markov Chain`을 이용하여 **종속 사건 또한 큰수의 법칙을 따른다**는 것을 보여주었습니다.

<br>
<center><img src="../assets/img/math/pb/markov_chain/0.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림을 살펴보겠습니다. 두 가지의 상태 0, 1이 있고 각 상태에 머무르는 확률과 다른 상태로 천이하는 확률이 나뉘어져 있습니다.
- 즉, 현재 상태가 0일 확률이 이전 상태에 영향을 받게 되는 구조입니다. 단순히 주사위 던지기 문제와는 다른 `종속 사건`이 됩니다.
- 위 경우의 예에서는 모든 사건이 0.5로 설정하였기 때문에 큰 수의 법칙을 적용하였을 때에 동일하게 각 상태에 머무를 확률이 0.5가 됩니다.

<br>
<center><img src="../assets/img/math/pb/markov_chain/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이번에는 확률의 비율을 다르게 가져갔으나 각 상태 0, 1이 천이하거나 머무르는 확률을 동일하게 설정하였습니다. (천이 = 0.3, 머무르는 경우 = 0.7)
- 이 경우에도 큰 수의 법칙을 적용하였을 때, 각 상태에 머무를 확률은 0.5와 0.5가 되었습니다.

<br>
<center><img src="../assets/img/math/pb/markov_chain/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 반대로 위와 같은 경우에는 0번 상태에 머무르는 확률이 0.72로 수렴하였습니다. 계속 돌려봤을 때에도 0.72로 수렴한 것으로 보아서 큰 수의 법칙이 적용된 것으로 볼 수 있습니다.

<br>
<center><img src="../assets/img/math/pb/markov_chain/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이번에는 양쪽의 확률을 완전 다르게 잡아보았습니다. 이 때에는 0번 상태에 머무르는 확률이 0.77로 수렴하였습니다. 즉 큰 수의 법칙이 적용되었습니다.
- 여기서 살펴보면 n번째에 0번 상태일 확률과 1번 상태일 확률은 이전 상태에 영향을 받기 때문에 독립적이지 않습니다.
- 하지만 **상태의 모든 확률을 아는 상태**에서 (위 처럼 천이할 때와 머무를 때의 확률을 아는 경우) 계속 반복하게 되면 어떤 특정한 확률로 수렴하게 됩니다.
- 즉, 이 방법은 독립성이 있는 사건들만 예측 가능한 분포에 수렴하게 된다는 **네크로소프의 주장을 반박**하게 됩니다. 즉 종속 사건들에서도 `상태의 모든 확률을 아는 경우에` 예측 가능한 분포로 수렴하게 된다는 것입니다.
- 이 때, 위 상태 천이를 모델링 한 것을 `Markov Chain` 이라고 하며 모델링 행렬을 다음과 같이 나타낼 수 있습니다.

<br>

$$ \text{trasition matrix} = \begin{pmatrix} 0.8 & 0.2 \\ 0.72 & 0.28 \end{pmatrix} $$

<br>

- 위와 같은 형태로 다양한 Markov Chain을 모델링 할 수 있고 다음과 같이 조금 더 복잡하게 만들 수도 있습니다.

<br>
<center><img src="../assets/img/math/pb/markov_chain/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 다음 링크를 클릭하면 마코프 체인에 대한 시뮬레이션을 해볼 수 있습니다.

<br>

<h3><a href="https://ko.khanacademy.org/computing/computer-science/informationtheory/moderninfotheory/pi/markov-chain-exploration">마코프 체인</a></h3><script src="https://ko.khanacademy.org/computer-science/markov-chain-exploration/Interactive:x9ac63c8f9bbfd6cf/embed.js?editor=yes&buttons=yes&author=yes&embed=yes"></script>

<br>

## **Hidden markov model**

<br>



<br>

- [통계학 관련 글 목록](https://gaussian37.github.io/math-pb-table/)

<br>

  
  
        
        
    
          
  
