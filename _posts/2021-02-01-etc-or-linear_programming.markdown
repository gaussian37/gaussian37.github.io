---
layout: post
title: 선형 및 정수 계획법과 엑셀의 활용
date: 2021-02-12 00:00:00
img: etc/or/linear_programming/0.png
categories: [etc-or] 
tags: [선형 계획법, 정수 계획법, 엑셀] # add tag
---

<br>

- 이번 글에서는 경영 과학에서 다루는 선형 계획법, 정수 계획법 등과 같은 문제를 엑셀을 이용하여 어떻게 다루는 지 살펴보도록 하겠습니다.
- 기업들이 갖는 가장 일반적인 목표는 `수익 최대화` 또는 `비용 최소화` 문제이며 **주어진 제한 조건하에서 어떤 목표를 달성**하려는 문제를 풀 때 `선형 계획법`을 종종 사용합니다. 예제를 통하여 선형 계획법의 문제를 살펴보도록 하겠습니다.

<br>

- 먼저 선형 계획법 및 정수 계획법에 필요한 기본 용어들을 기업 활동에 빗대어 표현하면 다음과 같습니다.
- ① `의사 결정 변수` : 기업에 의한 활동의 수준을 나타내는 수학적 기호 ($$ X_{1},X_{2}, \cdots $$)
- ② `목적 함수` : 의사 결정 변수를 사용하여 기업의 목표를 선형적인 수학 관계로 표현하며 이 함수식을 **최대화 혹은 최소화** 합니다. 일반적으로 $$ Z $$로 나타냅니다.
- ③ `제약식` : 경영 환경에 의해 기업에게 주어지는 제한 사항을 의미(의사결정 변수의 선형적인 관계로 표현)합니다. 제약식의 경우 `subject to (S.T)`라는 항목으로 하나씩 나열해서 나타냅니다.

<br>

## **선형 계획법**

<br>

- 이 글에서는 크게 `선형 계획법`과 `정수 계획법`에 대하여 다룹니다. 먼저 `선형 계획법`에 대하여 다루어 보도록 하겠습니다.

<br>

#### **타이어 문제**

<br>

- 선형 계획법 타이어 문제 엑셀 시트 : [https://drive.google.com/file/d/10zppt6eSdJW9nGYDcg3wYuHc6MncH-RG/view?usp=sharing](https://drive.google.com/file/d/10zppt6eSdJW9nGYDcg3wYuHc6MncH-RG/view?usp=sharing)

<br>

- 그러면 구체적인 예시를 통하여 `의사 결정 변수`, `목적 함수`, `제약식`,  모델링하고 엑셀을 통하여 문제를 푸는 방법에 대하여 다루어 보도록 하겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 데이터를 이용하여 문제를 정리해 보겠습니다.
- ① `의사 결정 변수` :
    - 　$$ X_{1} $$ : 1일 대형 타이어 생산량
    - 　$$ X_{2} $$ : 1일 소형 타이어 생산량
- ② `목적 함수` :     
    - 최대화 : $$ Z = 10X_{1} + 6X_{2} $$, $$ Z $$ : 1일 총 이익
- ③ `제약 조건` : 
    - 　$$ 5X_{1} + 2X_{2} \le 20 $$
    - 　$$ 4X_{1} + 4X_{2} \le 24 $$
    - 　$$ X_{1}, X_{2} \ge 0 $$

<br>

- 위 정보를 정리하여 엑셀에 표시해 보겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 엑셀 시트를 통하여 `의사 결정 변수`, `목적 함수`, `제약 조건`을 표현할 수 있습니다. 위 시트의 수식을 살펴보면 다음과 같습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 위 엑셀에서 사용된 `SUMPRODUCT` 함수는 마치 벡터의 내적처럼 동작하게 됩니다. 대응되는 값들을 곱한 다음에 모두 더하는 연산입니다.
- 예를 들어 `SUMPRODUCT($B$7:$C$7,B10:C10)`와 같은 연산은 `B7*B10 + C7*C10` 연산이라고 생각하시면 됩니다.
- 의사 결정 변수 값은 비어져 있습니다. 목적 함수를 최적화 하였을 때, 자동으로 결정되는 변수이므로 엑셀을 이용하여 목적 함수를 최적화 하였을 때 자동적으로 결정됩니다.
- 그러면 엑셀의 `데이터` 탭의 `해 찾기` 기능을 이용하여 어떻게 선형 계획법 문제를 푸는 지 살펴보도록 하겠습니다. (해 찾기 기능은 옵션에서 추가를 해야 사용할 수 있습니다.)

<br>
<center><img src="../assets/img/etc/or/linear_programming/4.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- `해 찾기` 기능을 통하여 앞에서 정의한 `의사 결정 변수 조건`, `제약 조건`을 만족 하면서 `목적 함수`를 최적화 시키는  `의사 결정 변수`를 찾을 수 있습니다.
- ① 목표 설정에는 `목적 함수`가 대응되어야 합니다. 목적 함수는 풀어야 할 최종 목적이 되기 때문입니다.
- ② `목적 함수`가 풀어야 할 문제를 지정합니다. 보통 최적화 문제는 **최대값**을 찾거나 **최솟값**을 찾습니다. 따라서 최댓값 또는 최솟값을 찾을 수 있도록 설정합니다.
- ③ `의사 결정 변수`를 입력합니다. 각 셀 별로 따로 입력을 해도 되고 위 예시 처럼 범위로 입력하여도 됩니다.
- ④ `제약 조건`을 차례 차례 입력 합니다.
- ⑤ `의사 결졍 변수 조건` 중 의사 결정 변수가 음수아 되지 않는 조건을 만족하도록 체크 박스를 체크합니다.
- ⑥ 현재 풀려는 문제가 **선형**식 (1차 함수 식)으로 되어 있으므로 `단순 LP (Simplex Linear Programming)`을 선택 합니다.
- 위 입력 사항들을 모두 입력한 뒤, 해 찾기 버튼을 누르면 다음과 같이 해를 찾습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 대형 타이어 약 2.67개, 소형 타이어 약 3.33개를 생산할 때, 수익은 약 46.67으로 최대화 할 수 있습니다.

<br>

#### **자동차 조립 문제 1**

<br>

- 선형 계획법 자동차 조립 문제 엑셀 시트 : [https://drive.google.com/file/d/1wGrsPBD1Ix_LBdoG7S1kMpe2gBUyVsnb/view?usp=sharing](https://drive.google.com/file/d/1wGrsPBD1Ix_LBdoG7S1kMpe2gBUyVsnb/view?usp=sharing)

<br>

- 자동차 조립을 할 때, 모델 A는 한 대당 3,600 달러의 수익을 얻을 수 있고 모델 B는 한 대당 5,400 달러의 수익을 얻을 수 있습니다.
- 모델 A의 조립 시간은 6시간이고 모델 B의 조립 시간은 10.5 시간입니다.
- 공장의 총 가용 시간은 48,000 시간이고 생산된 모든 차는 팔린다고 가정합니다.
- 모델 A는 문이 4개 짜리 승용차이고 모델 B는 문이 2개짜리 승용차 이며 모델 A, B의 문은 같습니다. 문은 최대 20,000개 까지 업체로 부터 공급 받을 수 있습니다.
- 모델 B의 수요가 3,500대 이하일 것으로 예상하므로 모델 B의 공급은 최대 3,500대로 제한합니다.

<br>

- ① `의사 결정 변수` :
    - 　$$ X_{1} $$ : 모델 A의 생산량
    - 　$$ X_{2} $$ : 모델 B의 생산량
- ② `목적 함수` : $$ Z = 3600X_{1} + 5400X_{2} $$ ($$ Z $$를 최대화)
- ③ `제약식` : 
    - 　$$ 6X_{1} + 10.5X_{2} \le 48000 $$
    - 　$$ 4X_{1} + 2X_{2} \le 20000 $$
    - 　$$ X_{2} \le 3500 $$
    - 　$$ X_{1}, X_{2} \ge 0 $$

<br>

- 위 식과 같이 모델링을 할 수 있으며, 엑셀에서 정리하면 다음과 같은 방식으로 문제를 해결할 수 있습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 결과를 살펴 보면 모델 B의 수요가 3,500대로 예쌍하여 생산 대수 제한을 3,500대로 두었음에도 불구하고 수익을 최대화 하기 위해서는 그것 보다 작은 2,400대를 생산해야 하는 것을 알 수 있습니다.
- 따라서 현재 제약 조건이 고정되어 있는 상태에서는 모델 B의 공급이 부족하기 떄문에 모델 B에 대한 광고 캠페인은 옳지 못하다고 판단할 수 있습니다.

<br>

#### **자동차 조립 문제 2**

<br>

- 자동차 조립 문제 1번 조건에서 공장의 추가 노동 시간을 25% 증가시킬 수 있다고 가정하겠습니다.
- 이 때, 각 모델의 생산량과 총 수입을 구하고 추가 노동 시간에 대한 비용은 최대 얼마 까지 지불할 수 있는지 구해보겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 엑셀 시트를 보면 노동 시간에 대한 제약 조건이 48,000 → 60,000으로 25% 증가된 것을 확인할 수 있습니다.
- 이 때, 동일한 방법으로 총 수익의 최대값을 찾았을 때, 위 식과 같이 구할 수 있습니다.
- 추가 노동으로 인한 수익은 3,960,000 달러 만큼 늘어났으므로 추가 노동 비용은 최대 차익 만큼 지불할 수 있습니다.

<br>

#### **자동차 조립 문제 3**

<br>

- 자동차 조립 문제 2 조건에서 광고 캠페인을 통하여 모델 B의 수요를 20% 끌어올린다고 가정하겠습니다.
- 자동차 조립 문제 2 조건의 추가 노동 비용이 160만 달러 이고 광고 캠페인 비용이 50만 달러일 때, 이러한 전략이 옳은 지 확인해 보겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/10.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 엑셀 식의 결과와 같이 추가 노동 비용 160만 달러, 광고 캠페인 비용 50만 달러 총 210만 달러를 사용하더라도 366만 달러의 추가 수익을 얻을 수 있으므로 추가 노동과 광고 캠페인을 진행하는 것이 합리적이라고 말할 수 있습니다.

<br>

#### **자동차 조립 문제 4**

<br>

- 문제 1의 조건에서 모델 B의 시장에서의 더 많은 판매 우위를 가지기 위하여 최대 수요량 만큼 생산해 보겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/11.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 엑셀 시트에서 C12를 보면 부등호에서 등호로 바뀌었습니다. 이 조건을 해 찾기를 할 때, 그대로 적용하면 됩니다. 이 조건을 통하여 최대 수요량 만큼 생산하였을 때, 총 수익을 확인할 수 있습니다.

<br>

## **정수 계획법**

<br>

- 정수 계획법은 간단히 앞에서 다룬 선형 계획법에서 `의사 결정 변수`를 정수로 제한하는 방법이라고 말할 수 있습니다.
- 정수 계획법을 사용하는 이유는 선형 계획법 특성 중 해가 실수로 나오는 경우 사용 불가능한 해가 될 수 있기 때문입니다. 예를 들어 생산해야 할 차의 댓수를 구하는데 0.5대와 같은 최적해는 실현 불가하기 때문입니다.
- 정수 계획법에서 다루는 모델은 다음과 같습니다.
    - ① `순수 정수 모형` : 모든 의사 결정 변수들이 정수해를 가지는 경우
    - ② `혼합 정수 모형` : 일부 의사 결정 변수들이 정수해를 가지는 경우
    - ③ `0-1 정수 모형` : 모든 의사 결졍 변수들이 0 또는 1의 정수해를 가지는 경우
- 정수 계획법은 실수 계획법에 비하여 수의 범위가 작음에도 불구하고 현실 문제에는 정수가 많기 때문에 오히려 정수 계획법이 더 많이 사용됩니다.
- 특히 `0-1 정수 모형`은 선택의 문제에 많이 사용되며 정수 계획법에서도 많은 비중을 차지하고 있습니다.

<br>

#### **선풍기 생산 문제 (순수 정수 모형)**

<br>

- 정수 계획법 선풍기 생산 문제 엑셀 링크 : [https://drive.google.com/file/d/1qk-AzyPo8_G8Sk1NqTzKLWD6qhTfZIc_/view?usp=sharing](https://drive.google.com/file/d/1qk-AzyPo8_G8Sk1NqTzKLWD6qhTfZIc_/view?usp=sharing)

<br>

- 두 종류의 선풍기 A와 B를 생산하는 문제 입니다.  선풍기 A의 단위당 이익은 7달러, 선풍기 B의 단위당 이익은 6달러 입니다. 
- 두 제품 모두 전선 공정과 조립 공정으로 생산해야 하며 선풍기 A의 전선 공정은 2시간, 선풍기 B의 전선 공정은 3시간이며 총 사용 가능한 전선 공정은 12시간 입니다. 선풍기 A의 조립 공정은 6시간, 선풍기 B의 조립 공정은 5시간이며 총 사용 가능한 조립 공정은 30시간입니다.
- 이 때, 최대 이익을 구해보겠습니다.

<br>

- 순수 정수 모형 문제를 풀 때에 앞에서 사용한 선형 계획법을 그대로 이용하면 되고 제약 조건에 의사결정 변수가 정수임을 추가하면 됩니다.
- ① `의사 결정 변수` : 
    - 　$$ X_{1} $$ : 선풍기 A 생산량
    - 　$$ X_{2} $$ : 선풍기 B 생산량
- ② `목적 함수` : $$ Z = 7X_{1} + 6X_{2} $$ ($$ Z $$를 최대화)
- ③ `제약식` : 
    - 　$$ 2X_{1} + 3X_{2} \le 12 $$
    - 　$$ 6X_{1} + 5X_{2} \le 30 $$
    - 　$$ X_{1}, X_{2} \ge 0 $$
    - 　$$ X_{1}, X_{2} : \text{integer} $$

<br>
<center><img src="../assets/img/etc/or/linear_programming/12.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/13.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 해 찾기 기능을 보면 의사 결정 변수에 정수 조건이 추가되었습니다. 앞에서 말씀드린 바와 같이 정수 계획법은 의사 결정 변수에서 정수가 적용되어야 하는 변수에 정수 조건을 추가하면 끝입니다.

<br>

#### **식품 생산 문제 (혼합 정수 모형)**

<br>

- 정수 계획법 식품 생산 문제 엑셀 링크 : [https://drive.google.com/file/d/13J5GaVl5SIol2bcLl3Wsta42Y6o37_93/view?usp=sharing](https://drive.google.com/file/d/13J5GaVl5SIol2bcLl3Wsta42Y6o37_93/view?usp=sharing)

<br>

- 두 종류의 식품 A와 B를 생산하는 문제입니다. 식품 A는 1개 당 판매이익이 8500원이고 식품 B는 1개당 150원 입니다.
- 식품 A는 포장 단위(개)로 생산이 되어야 하고 식품 B는 일부분만 생산이 되어도 됩니다.
- 식품 A와 B는 원료가 3종류이며 식품 A는 1개당 원료1=30g, 원료2=18g, 원료3=2g이 필요하고 식품 B는 원료1=0.5g, 원료2=0.4g, 원료3=0.1g이 듭니다.
- 각 원료의 사용 가능양은 원료1=2000g, 원료2=800g, 원료3=200g 입니다.
- 이 때, 수익을 최대화 하는 조합을 찾아보겠습니다.

<br>

- ① `의사 결정 변수` : 
    - 　$$ X_{1} $$ : 식품 A 생산량
    - 　$$ X_{1} $$ : 식품 B 생산량
- ② `목적 함수` : $$ Z = 8500X_{1} + 150X_{2} $$ ($$ Z $$를 최대화)
- ③ `제약식` : 
    - 　$$ 30X_{1} + 0.5X_{2} \le 2000 $$
    - 　$$ 18X_{1} + 0.4X_{2} \le 800 $$
    - 　$$ 2X_{1} + 0.1X_{2} \le 200 $$
    - 　$$ X_{1}, X_{2} \ge 0 $$
    - 　$$ X_{1} : \text{integer} $$

<br>
<center><img src="../assets/img/etc/or/linear_programming/14.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/15.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 해 찾기 기능을 보면 의사 결정 변수에 정수 조건이 추가되었고 변수 $$ X_{1} $$에 대해서만 정수 조건이 추가된 것을 확인할 수 있습니다.

<br>

#### **부동산 투자 문제 (0-1 정수 모형)**

<br>

- 정수 계획법 식품 생산 문제 엑셀 링크 : [https://drive.google.com/file/d/12eHQVzu-kBKbCO9OM72_1OhxKucSfZUe/view?usp=sharing](https://drive.google.com/file/d/12eHQVzu-kBKbCO9OM72_1OhxKucSfZUe/view?usp=sharing)

<br>

- 부동산 투자를 위하여 최대 3,000만원을 사용 할 수 있습니다.
- A지역에는 2개 이상을 매입하려고 하고 B 지역에는 1개 이하를 매입하려고 하고 C 지역에는 1개를 매입하려고 합니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/16.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 문제는 각 투자 대안을 선택 또는 미선택 하였을 때, 이익을 최대화 하는 문제 입니다. 즉 0-1 정수 모형이 됩니다.

<br>

- ① `의사 결정 변수` : 
    - 　$$ X_{i} $$ : 투자 대안 $$ i $$, ($$ i = 0, 1, ..., 7 $$)
- ② `목적 함수` : $$ Z = 480X_{1} + 540X_{2} + 680X_{3} + 1000X_{4} + 700X_{5} + 510X_{6} + 900X_{7} $$ ($$ Z $$를 최대화)
- ③ `제약식` : 
    - 　$$ X_{1} + X_{4} + X_{5} \ge 2 $$
    - 　$$ X_{2} + X_{3} \le 1 $$
    - 　$$ X_{6} + X_{7} = 1 $$
    - 　$$ X_{i} : \text{0 or 1} $$

<br>
<center><img src="../assets/img/etc/or/linear_programming/17.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/18.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 노란색 조건을 보면 2진수 조건 즉, 0과 1을 사용함으로써 선택 또는 미선택을 통해 이익을 최대화 할 수 있습니다.

<br>

- 0-1 정수 모형 방식을 사용할 때, 제약식은 몇가지 방법을 예시를 통하여 나타내 보겠습니다.
- `상호 베타적 제약식` : A 지역의 토지 또는 아파트 중 오직 하나만 선택 한다면 **둘 다 선택이 불가**하므로 다음과 같이 제약 할 수 있습니다.
    - 제약식 : $$ X_{2} + X_{3} \le 1 $$
- `다지선택 제약식` : C 지역의 토지 혹은 상가 중 오직 하나만 선택한다면 **둘 중 하나는 꼭 선택**해야 합니다.
    - 제약식 : $$ X_{6} + X_{7} = 1 $$
- `조건부 제약식` : C 지역 상가에 투자를 하기 위해서는 꼭 토지에도 투자를 해야 한다면 토지 투자가 더 많이 되어야 합니다.
    - 제약식 : $$ X_{7} \le X_{6} $$
- `동시 요구 제약식` : C 지역은 두 지역에 동시 투자하지 않으면 아예 투자가 불가능 하다면
    - 제약식 : $$ X_{6} = X_{7} $$

<br>

- 이와 같이 정수 계획법의 유용한 사례들은 0-1 정수 모형으로 나타내어 집니다.
- `1` : 선택 / 구매 / 채용 등
- `0` : 기각 / 비구매 / 비채용 등

<br>

#### **자본재 문제 (0-1 정수 모형)**

<br>

- 정수 계획법 자본재 문제 엑셀 링크 : [https://drive.google.com/file/d/1cm3fpOl0kxI-8qUBBqeKI0Z-sA2DsoVl/view?usp=sharing](https://drive.google.com/file/d/1cm3fpOl0kxI-8qUBBqeKI0Z-sA2DsoVl/view?usp=sharing)

<br>

- 한 대학 서점이 몇가지 사업 확장 프로젝트를 검토하고 있습니다.
    - ① 온라인 소매와 제품 목록 구입을 가능하게 하는 매장 웹사이트 구축
    - ② 캠퍼스 외부의 창고 구입과 지속적인 확장
    - ③ 학교 로고가 새겨진 옷을 전문적으로 취급하는 의류 및 기념품 판매부서 개발
    - ④ 하드웨어와 소프트웨어를 동시에 취급하는 컴퓨터 관련 판매부서 개발
    - ⑤ 매장 외부에 3대의 ATM 설치
- 매장의 공간 부족으로 인해 컴퓨터 부서와 의류 부서를 동시에 설치 할수는 없습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/19.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이윤을 최대화 하기 위해 추진해야 할 프로젝트를 선택해 보겠습니다.

<br>

- ① `의사 결정 변수` : 
    - 　$$ X_{1} $$ : 웹사이트 구축 프로젝트 선택
    - 　$$ X_{1} $$ : 창고 구입 프로젝트 선택
    - 　$$ X_{1} $$ : 의류 판매부서 프로젝트 선택
    - 　$$ X_{1} $$ : 컴퓨터 판매부서 프로젝트 선택
    - 　$$ X_{1} $$ : ATM 프로젝트 선택    
- ② `목적 함수` : 최대화 $$ Z = 120X_{1} + 85X_{2} + 105X_{3} + 140X_{4} + 70X_{5} $$
- ③ `제약식` : 
    - 　$$ 55X_{1} + 45X_{2} + 60X_{3} + 50X_{4} + 30X_{5} \ge 150 $$
    - 　$$ 40X_{1} + 35X_{2} + 25X_{3} + 35X_{4} + 30X_{5} \ge 110 $$
    - 　$$ 25X_{1} + 20X_{2} + 0*X_{3} + 30X_{4} + 0*X_{5} \ge 60 $$
    - 　$$ X_{3} + X_{4} \le 1 $$
    - 　$$ X_{i} $$ : 1 (선택), 0 (미선택)

<br>
<center><img src="../assets/img/etc/or/linear_programming/20.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/21.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

#### **서비스 조직망 문제 (0-1 정수 모형)**

<br>

- 정수 계획법 서비스 조직망 문제 엑셀 링크 : [https://drive.google.com/file/d/1221WzIljbzJlyL3LrQPoWUR6qYR_3OMn/view?usp=sharing](https://drive.google.com/file/d/1221WzIljbzJlyL3LrQPoWUR6qYR_3OMn/view?usp=sharing)

<br>

- 12개의 도시들에 우편 서비스를 제공하기 위하여 수송 거점 도시를 선정해야 합니다.
- 각 도시들로 부터 300 마일 이내에 다음 12개의 도시들 중에서 최소 한의 갯수로 이어지는 새 수송거점을 선택하여 12개 도시 모두를 감당할 수 있도록 도시를 선택해 보도록 하겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/22.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 도표는 각 도시 별 300 마일 이내에 있는 도시 목록 입니다.

<br>

- ① `의사 결정 변수` : 
    - 　$$ X_{i} $$ : $$ i $$ 번째 도시가 선택되었는 지 유무 ($$ i = 1, 2, ..., 12 $$)
- ② `목적 함수` : 최소화 $$ Z = \sum_{i=1}^{12}X_{i} $$
- ③ `제약식` : 
    - 　$$ 1*X_{1} + 0*X_{2} + 1*X_{3} + 0*X_{4} + 0*X_{5} + 0*X_{6} + 0*X_{7} + 1*X_{8} + 0*X_{9} + 0*X_{10} + 0*X_{11} + 0*X_{12} \ge 1 $$
    - 　$$ 0*X_{1} + 1*X_{2} + 0*X_{3} + 0*X_{4} + 0*X_{5} + 0*X_{6} + 0*X_{7} + 0*X_{8} + 0*X_{9} + 1*X_{10} + 0*X_{11} + 0*X_{12} \ge 1 $$
    - (중략)
    - 　$$ 0*X_{1} + 0*X_{2} + 0*X_{3} + 0*X_{4} + 0*X_{5} + 1*X_{6} + 0*X_{7} + 1*X_{8} + 0*X_{9} + 0*X_{10} + 0*X_{11} + 1*X_{12} \ge 1 $$
    - 　$$ X_{i} $$ : 1 (선택), 0 (미선택)

<br>

- 여기서 주목해야 할 점은 `제약식`의 부등호 방향입니다. 부등호 방향을 보면 모두 1보드 크거나 같다로 되어있습니다. 즉, 최소 1개는 선택되도록 제약을 주었습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/23.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/24.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

#### **고정비용과 시설의 위치 문제 (0-1 정수 모형)**

<br>

- 정수 계획법 고정비용과 시설의 위치 문제 엑셀 링크 : [https://drive.google.com/file/d/1sqBjPMem3R_KnjNU6EejhRu3XQ8WuDIr/view?usp=sharing](https://drive.google.com/file/d/1sqBjPMem3R_KnjNU6EejhRu3XQ8WuDIr/view?usp=sharing)
<br>

- 이번 문제는 마지막 문제로 꽤 복잡한 문제입니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/28.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 농장에서 수확한 농작물을 공장으로 전달하여 제품을 만들려고 합니다.
- 이 때, 고려해야 할 점은 선택 가능한 6개의 농장의 생산량과 고정 비용이 다르다는 점이 있고 공장에서도 수용 가능한 양이 다른점 입니다.
- 이 때, 농장과 공장 간 수송 비용도 고려해야 합니다.
- 이 상황에서의 사용 가능한 생산 용량을 모두 사용하면서 총 비용을 최소화 하는 방법을 구해보겠습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/25.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 먼저 후보지인 6개의 농장들은 아래와 같은 연간 고정 비용과 연간 추정 생산량을 가지고 있습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/26.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 회사가 현재 보유하고 있는 가공 공장들은 추가 가능 용량이 위 표과 같습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/27.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 농장에서 각 공장 까지 수송 비용이 위 표와 같습니다.

<br>
<center><img src="../assets/img/etc/or/linear_programming/29.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 전체 정보를 요약하면 위 그림과 같습니다.

<br>

- ① `의사 결정 변수` : 
    - 　$$ y_{i} $$ : 농장 $$ i $$를 선택하지 않음 (0), 농장 $$ i $$를 선택 (1) ($$ i = 1, 2, 3, 4, 5, 6 $$)
    - 　$$ X_{ij} $$ : 농장 $$ i $$로부터 공장 $$ j $$ 까지 수송할 감자의 양 ($$ i = 1, 2, 3, 4, 5, 6 / j = A, B, C $$)
- ② `목적 함수` : 최소화 $$ Z = 18X_{1A} + 15X_{1B} + 12X_{1C} + ... + 520y_{5}+ 465y_{6} $$
- ③ `제약식` : 
    - 　$$ X_{1A} + X_{1B} + X_{1C} \le 11.2y_{1} $$
    - 　$$ X_{2A} + X_{2B} + X_{2C} \le 10.5y_{2} $$
    - 　$$ X_{3A} + X_{3B} + X_{3C} \le 12.8y_{3} $$
    - 　$$ X_{4A} + X_{4B} + X_{4C} \le 9.3y_{4} $$
    - 　$$ X_{5A} + X_{5B} + X_{5C} \le 10.8y_{5} $$
    - 　$$ X_{6A} + X_{6B} + X_{6C} \le 9.6y_{6} $$
    - 　$$ X_{1A} + X_{2A} + X_{3A} + X_{4A} + X_{5A} + X_{6A} = 12 $$
    - 　$$ X_{1B} + X_{2B} + X_{3B} + X_{4B} + X_{5B} + X_{6B} = 10 $$
    - 　$$ X_{1C} + X_{2C} + X_{3C} + X_{4C} + X_{5C} + X_{6C} = 14 $$
    - 　$$ X_{ij} \ge 0 $$
    - 　$$ y_{i} $$ : 1 (선택), 0 (미선택)

<br>
<center><img src="../assets/img/etc/or/linear_programming/30.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/etc/or/linear_programming/31.png" alt="Drawing" style="width: 800px;"/></center>
<br>