---
layout: post
title: 기초 정보 이론 (Entropy, Cross Entropy, KL divergence 등)
date: 2020-08-09 00:00:00
img: ml/concept/basic_information_theory/0.png
categories: [ml-concept] 
tags: [machine learning, probability model, entropy, kl-divergence, mutual information, cross entropy, Jensen's inequality] # add tag
---

<br>

[머신 러닝 관련 글 목록](https://gaussian37.github.io/ml-concept-table/)

<br>

- 참조 : [https://gaussian37.github.io/ml-concept-infomation_theory/](https://gaussian37.github.io/ml-concept-infomation_theory/)

<br>

- 이번 글에서는 머신 러닝을 공부할 때 중요한 개념 중 하나인 `정보 이론`에 대하여 다루어 보도록 하겠습니다. 이번 글에서 다루게 될 키워드는 `Entropy`, `Cross Entropy`, `KL divergence`, `Jensen's inqeuality`, `Mutual Information`이 있습니다.
- ① `정보 이론 (Information Theory)` : 확률 분포 $$ P(X) $$ 의 불확실 정도를 평가하는 방법
- ② `정보량 (information)` : 불확실함을 해소하기 위해 필요한 질문(정보)의 수, 또는 어떤 Event가 발생하기 까지 필요한 시행의 수
- ③ `엔트로피 (Entropy)` : 확률분포 $$ P(X) $$ 에 대한 정보량의 기댓값 (평균)
- ④ `크로스 엔트로피 (Cross Entropy)` : 
- ⑤ `KL divergence` : 

<br>

## **목차**

<br>

- ### 정보이론의 의미
- ### 정보량의 의미
- ### Entropy
- ### Cross Entropy
- ### KL divergence
- ### Mutual Information

<br>

## **정보이론의 의미**

<br>

- 먼저 정보이론 (`Information Theory`)의 의미에 대하여 살펴보도록 하겠습니다. 머신 러닝을 다루다 보면 나오는 개념인 `Probability Theory`와 `Decision Theory`와 비교하여 설명하면 다음과 같습니다.

<br>

- ① `Probability Theory` : 불확실성 (Evnet, 변수 등)에 대한 일어날 가능성을 모델링 하는 것을 의미합니다. 아래는 확률 모형 중 하나인 베이즈 모델의 예시 입니다.

<br>

- $$ P(Y \vert X) = \frac{P(X \vert Y)P(Y)}{P(X)} \tag{1} $$

<br>

- ② `Decision Theory` : 불확실한 상황에서 추론에 근거해 결정을 내리는 방식입니다.

<br>

- $$ Y = 1 \text{if } \frac{P(x \vert y = 1)p(y = 1)}{P(x \vert y = 0)p(y = 0)} > 1 \tag{2} $$

<br>

- ③ `Information Theory` : 확률 분포 $$ P(X) $$ 의 불확실 정도를 평가하는 방법입니다.

<br>

- $$ H(X) = -\sum_{x}P(X) \log{P(X)} \tag{3} $$

<br>

- 이번 글에서 집중적으로 다룰 내용은 ③ `Information Theory` 이며 **식 (3)의 의미를 이해하는 것이 글의 목표**가 되겠습니다.


<br>

## **정보량의 의미**

<br>

- 정보량의 의미를 이해하기 위하여 1 ~ 16 까지의 숫자 중 상대방이 선택한 하나의 숫자를 알아 맞추는 게임을 진행한다고 해보겠습니다. 
- 상대방은 나의 질문에 Yes / No 2가지 답변만 가능합니다. 이와 같은 문제는 `binary search` 방식을 이용하여 어떤 수 $$ X $$ 보다 큰 지 질문하고 그 질문에 Yes / No 답변을 이용하여 찾으면 $$ \log_{2}{16} = 4 $$ 만큼의 질문을 통해 찾을 수 있음이 알려져 있습니다. 
- 이 때, `4`를 정보량이라고 합니다. 여기서 `log`의 밑인 `2`는 `Yes/No` 2가지 질문의 경우의 수를 의미합니다.

<br>

- 위 예제에서 주목할 점은 처음에는 상대방이 생각한 숫자를 전혀 알 지 못해서 불확실성이 굉장히 큰 상황에서 질의를 통해 점점 불확실성을 줄일 수 있었고 최종적으로 불확실성이 완전히 줄어들어서 정답을 찾게 될 때까지 4번이라는 질의가 필요하다는 것입니다.
- 이와 같이 **불확실함을 해소하기 위해 필요한 질문(정보)의 수, 또는 어떤 Event가 발생하기 까지 필요한 시행의 수**를 `정보량 (information)`이라고 합니다.
- 특히, 위 예제와 같이 2가지의 질문 (Yes / No)를 사용한 경우 `bit`가 **정보량의 단위**가 됩니다.

<br>

- 이 문제에서 암묵적으로 가정한 것은 어떤 숫자를 선택할 지를 **모두 동등한 확률**인 $ p = 1/16 $$ 로 가정한 점이 있습니다.
- 선택한 숫자를 맞추귀 위한 정보량을 확률 $$ p = 1/16 $$ 를 이용하여 표현하면 다음과 같습니다.

<br>

- $$ n = -\log{2}{(p)} \tag{4} $$

- $$ n = \log{2}{(1/p)} \tag{5} $$

- $$ 4 = -\log{2}{(16)} \tag{6} $$

<br>

- 이와 같이 일반적으로 어떤 event의 확률이 $$ p $$ 라고 하였을 떄, 그 event에 대한 정보량 $$ I $$ 는 다음과 같이 계산합니다.

<br>

- $$ I = \log_{2}{(\frac{1}{p})} = -log_{2}{(p)} \tag{7} $$

<br>

- 식 (7)과 같이 정보량을 계산할 때, 다음 3가지 예시에 대하여 정보량을 각각 구해보도록 하겠습니다.
- ① 주사위를 던져서 짝수의 눈이 나타날 event $$ E_{1} $$ 의 정보량

<br>

- $$ P(E_{1}) = \frac{1}{2} $$

- $$ I = -\log_{2}{\frac{1}{2}} = 1 \text{(bit)} \tag{8} $$

<br>

- ② 주사위를 던져서 2의 눈이 나타날 event $$ E_{2} $$ 의 정보량

<br>

- $$ P(E_{2}) = \frac{1}{6} $$

- $$ I = -\log_{2}{\frac{1}{6}} = 2.58492... \text{(bit)} \tag{9} $$

<br>

- ③ 주사위를 던져서 1 ~ 6의 눈이 나타날 event $$ E_{1} $$ 의 정보량

<br>

- $$ P(E_{3}) = 1 $$

- $$ I = -\log_{2}{1} = 0 \text{(bit)} \tag{10} $$

<br>

## **Entropy**

<br>

- 앞의 예제를 통해 `정보량`의 의미에 대하여 살펴보았습니다. 결론적으로 `정보량의 기댓값`을 `엔트로피 (Entropy)`라고 합니다.
- 앞에서 16개의 수 각각에 대하여 정보량을 구하고 그 정보량의 평균을 구한다면 다음과 같이 계산할 수 있습니다.

<br>

- $$ H(X) = -\sum_{i=1}^{16} P(X = i)\log_{2}P(X = i) \tag{11} $$

- $$ = \sum_{i=1}^{16} \frac{1}{16} \log_{2}{(\frac{1}{16})} = 4 (\text{(bit)} \tag{12} $$

<br>

- 이는 불확실성을 해소하기 위해서 **평균적으로 4번의 질의가 필요하다는 의미**입니다.
- 즉, `Entropy`란 확률 분포 $$ P(X) $$ 에 대한 `정보량의 기댓값`을 의미 합니다.

<br>

- $$ H(X) = -\sum_{X} \color{blue}{P(X)} \color{red}{\log_{2}{({P(X)})}} \tag{13} $$

<br>

- 식 (13)에서 파란색 항은 `확률`을 의미하고 빨간색 항은 `정보량`을 의미하므로 식 (13)은 `기댓값`으로 정의됩니다.

<br>

- 위에서 살펴본 예시에서는 각 event가 발생할 확률이 모두 같은 `uniform distribution`이었습니다. 
- 이번에는 동전 던지기를 이용하여 `uniform distribution`과 `non-uniform distribution`에 대하여 엔트로피를 계산해 보도록 하겠습니다.

<br>

- 먼저 동전 던지기를 하여 앞면, 뒷면의 발생 확률이 각각 1/2일 때, 앞면 뒷면을 맞추기 위한 엔트로피 즉, 정보량의 평균을 계산해 보겠습니다.

<br>

- $$ H(P) = P(X = H) \times -\log{P(X = H)} + P(X = T) \times -\log{P(X = T)} = \frac{1}{2} \times -\log_{2}{(\frac{1}{2})} + \frac{1}{2} \times -\log_{2}{(\frac{1}{2})} = 1 \text{(bit)} \tag{14} $$

<br>

- 만약 동전이 굽어져서 앞면이 나올 확률이 1/4, 뒷면이 나올 확률일 3/4라고 가정해 보겠습니다. 이 때, 앞면 또는 뒷면을 맞추기 위한 엔트로피를 계산해 보면 다음과 같습니다.

<br>

- $$ H(P) = P(X = H) \times -\log{P(X = H)} + P(X = T) \times -\log{P(X = T)} = \frac{1}{4} \times -\log_{2}{(\frac{1}{4})} + \frac{3}{4} \times -\log_{2}{(\frac{3}{4})} = 0.81127... \text{(bit)} \tag{15} $$

<br>

- 여기서 주목할만한 점은 확률 분포 $$ P(X) $$ 의 변화에 따라 엔트로피 값이 달라진다는 것입니다. 식 (15)의 경우에 엔트로피가 줄어든 것을 알 수 있습니다.

<br>

- $$ P(X = H) = p $$

- $$ P(X = T) = 1 - p $$

- $$ H(P) = -p\log_{2}{p} -(1-p)\log_{2}{(1-p)} \tag{16} $$

<br>

- 식 (16)을 이용하여 $$ p $$ 값에 따른 그래프를 그려보면 다음과 같습니다.

<br>
<center><img src="../assets/img/ml/concept/basic_information_theory/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 즉, `uniform distribution`일 때, 가장 엔트로피가 높고 한쪽으로 확률이 기울수록 엔트로피는 낮아집니다. 엔트로피의 의미를 곰곰히 생각해 볼 때, 위 그래프와 같은 결과가 나오는 것을 알 수 있습니다.

<br>
<center><img src="../assets/img/ml/concept/basic_information_theory/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 정리하면, 왼쪽 그래프와 같이 불균형한 분포 $$ P(X) $$의 불확실성이 더 적으므로 엔트로피 $$ H(X) $$ 가 더 낮고 오른쪽 그래프와 같이 균등한 분포 $$ P(X) $$ 의 불확실성이 높으므로 엔트로피 $$ H(X) $$ 가 높습니다.
- 엔트로피가 낮을 경우의 확률 분포에서는 표본을 샘플링하였을 때, 그 샘플링된 결과를 맞추기 쉽다고 생각하면 됩니다. 텍스트에 비유를 하자면 어떤 글에서 단어의 출현 빈도를 확률 분포라고 나타내었을 때, 분포가 균등한 것은 특정 주제가 없는 글이라서 단어의 출현 빈도가 균등하다고 보면되고 분포가 편향되어 있는 것은 특정 주제가 있는 글이라서 특정 단어가 자주 나온다고 생각할 수 있습니다.

<br>

- 지금까지 살펴본 `엔트로피`를 정리하면 다음과 같습니다.

<br>

- `엔트로피` : 확률 분포 $$ P(X) $$ 에서 일어날 수 있는 모든 사건들의 정보량의 기대값으로 $$ P(X) $$의 불확실 정도를 평가합니다.

<br>

- $$ H_{P} = -\sum_{x}P(X)\log_{2}{P(X)} \tag{17} $$

- $$ \text{discrete : } H_{P} = -\sum_{x}P(X)\log_{2}{P(X)} \tag{18} $$

- $$ \text{continuous : } H_{P} = -\int P(X)\log_{2}{P(X)}dX \tag{19} $$

<br>

- `엔트로피`에 대하여 추가적으로 몇가지 살펴보면 다음과 같습니다.
- ① 머신 러닝 문제에서 `Entropy`를 계산할 때, `log`의 밑을 `e`로 사용하는 자연 로그를 많이 사용합니다. `e`를 사용하여도 앞에서 살펴 보았던 예제에서 엔트로피 크기를 비교할 때는 전혀 문제가 없으며 미분 등의 연산을 할 때에도 계산에 유리하기 때문입니다.
- ② 엔트로피 $$ H(X) $$ 는 확률 분포 $$ P(X) $$의 불확실 정도를 축정할 때 사용할 수 있습니다.
- ③ 엔트로피는 확률 분포 $$ P(X) $$ 가 `constatnt` 또는 `uniform distribution`일 때 최대화가 됩니다.
- ④ 엔트로피는 확률 분포 $$ P(X) $$ 가 `delta function`일 때 최소화가 됩니다.
- ⑤ 엔트로피는 항상 양수입니다.

<br>

- `엔트로피`라는 개념을 이용하여 인코딩을 할 때에도 많이 사용됩니다. 이 방법을 `엔트로피 인코딩`이라고 합니다. `엔트로피 인코딩`은 심볼이 나올 확률에 따라 심볼을 나타내는 코드의 길이를 달리하는 방법으로 좋은 인코딩 방식은 **실제 데이터 분포 $$ P(X) $$ 를 알고 있을 때** (ex, 어떤 문자가 출현할 갯수 및 확률) 이에 반비례하게 코딩 길이를 정하는 것입니다. 즉, 높은 확률로 등장하는 심볼 및 데이터는 짧은 길이로 정한다고 보시면 됩니다.
- 이와 같이 `엔트로피 인코딩`을 이용한 유명한 사례가 `모스 부호`입니다.

<br>
<center><img src="../assets/img/ml/concept/basic_information_theory/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 E, I 같은 알파벳은 빈도수가 많기 때문에 그만큼 정보량이 작으므로 짧은 심볼에 매칭하였습니다.
- 반면 X, Y, Z 와 같은 알파벳은 빈도수가 작기 때문에 그만큼 정보량이 크므로 긴 심볼에 매칭하였습니다.
- 과거 모스 부호를 작성할 때에는 실제 데이터 즉, 알파벳이 발생하는 확률 분포 $$ P(X) $$ 를 알 수 없기 떄문에 도서관에 있는 책, 문서들을 보고 대략적으로 추정한 확률 분포 $$ Q(X) $$ 를 사용하였다고 알려져 있습니다.

<br>

## **Cross Entropy**

<br>


<br>

## **KL divergence**

<br>


<br>

## **Mutual Information**

<br>


<br>





<br>

[머신 러닝 관련 글 목록](https://gaussian37.github.io/ml-concept-table/)

<br>