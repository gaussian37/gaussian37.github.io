---
layout: post
title: 배치 정규화(Batch Normalization)
date: 2019-09-01 00:00:00
img: dl/concept/batchnorm/batchnorm.png
categories: [dl-concept] 
tags: [배치 정규화, 배치 노멀라이제이션, batch normalization] # add tag
---

<br>

- 참조
- https://youtu.be/TDx8iZHwFtM?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS

- 이번 글에서는 딥러닝 개념에서 중요한 것 중 하나인 `batch normalization`에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### Batch
- ### Internal Covariant Shift
- ### Batch Normalization
- ### Internal Covariant Shift 더 알아보기]
- ### Batch Normalization의 효과
- ### Pytorch에서의 사용 방법

<br>

## **Batch**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 배치 정규화를 설명하기에 앞서서 gradient descent 방법에 대하여 한번 생각해 보도록 하겠습니다.
- 먼저 위와 같이 일반적인 gradient descent에서는 gradient를 한번 업데이트 하기 위하여 모든 학습 데이터를 사용합니다.
- 즉, 학습 데이터 전부를 넣어서 gradient를 다 구하고 그 모든 gradient를 평균해서 한번에 모델 업데이트를 합니다. 
- 이런 방식으로 하면 대용량의 데이터를 한번에 처리하지 못하기 때문에 데이터를 `batch` 단위로 나눠서 학습을 하는 방법을 사용하는 것이 일반적입니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 그래서 사용하는 것이 stochastic gradient descent 방법입니다.
- SGD에서는 gradient를 한번 업데이트 하기 위하여 `일부의 데이터`만을 사용합니다. 즉, `batch` size 만큼만 사용하는 것이지요.
- 위 gradient descent의 식을 보면 $$ \Sum $$에 $$ j = Bi $$가 있는데 $$ B $$가 `batch`의 크기가 됩니다. 
- 한 번 업데이트 하는 데 $$ B $$개의 데이터를 사용하였기 때문에 평균을 낼 때에도 $$ B $$로 나누어 주고 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- 용어를 살펴보면 학습 데이터 전체를 한번 학습하는 것을 `Epoch` 라고 하고 Gradient를 구하는 단위를 `Batch` 라고 합니다. 

<br>

## **Internal Covariant Shift**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- `Batch` 단위로 학습을 하게 되면 발생하는 문제점이 있는데 이것이 논문에서 다룬 **Internal Covariant Shift** 입니다.
- 먼저 Internal Covariant Shift 의미를 알아보면 위 그림과 같이 학습 과정에서 계층 별로 입력의 데이터 분포가 달라지는 현상을 말합니다.
- 각 계층에서 입력으로 feature를 받게 되고 그 feature는 convolution이나 위와 같이 fully connected 연산을 거친 뒤 activation function을 적용하게 됩니다.
- 그러면 연산 전/후에 데이터 간 분포가 달라질 수가 있습니다. 
- 이와 유사하게 Batch 단위로 학습을 하게 되면 Batch 단위간에 데이터 분포의 차이가 발생할 수 있습니다.
- 즉, Batch 간의 데이터가 상이하다고 말할 수 있는데 위에서 말한 Internal Covariant Shift 문제입니다.
- 이 문제를 개선하기 위한 개념이 **Batch Normalization** 개념이 적용됩니다.

<br>

## **Batch Normalization**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>   

- batch normalization은 학습 과정에서 각 배치 단위 별로 데이터가 다양한 분포를 가지더라도 **각 배치별로 평균과 분산을 이용해 정규화**하는 것을 뜻합니다. 
- 위 그림을 보면 batch 단위나 layer에 따라서 입력 값의 분포가 모두 다르지만 정규화를 통하여 분포를 zero mean gaussian 형태로 만듭니다. 
- 그러면 평균은 0, 표준 편차는 1로 데이터의 분포를 조정할 수 있습니다.

<br>

- 여기서 중요한 것은 Batch Normalization은 학습 단계와 추론 단계에서 조금 다르게 적용되어야 합니다.
- 먼저 학습 단계를 살펴보도록 하겠습니다.

<br>

$$ BN(X) = \gamma \Bigl(  \frac{X - \mu_{batch} }{\sigma_{batch} } \Bigr) + \beta $$

<br>

- 여기서 $$ \gamma $$는 스케일링 역할을 하고 $$ \beta $$는 bias입니다. 물론 두 값 모두 backprpagation을 통하여 학습을 하게 됩니다.

<br>

$$ \mu_{batch} = \frac{1}{B} \sum_{i} x_{i} $$

<br>

$$ \sigma^{2}_{batch} = \frac{1}{B} \sum_{i} (x_{i} - \mu_{batch})^{2} $$

<br>

### **학습 단계의 배치 정규화**

<br>

- 학습 단계의 BN을 구하기 위하여 사용된 평균과 분산을 구할 때에는 **배치별로 계산**되어야 의미가 있습니다. 그래야 각 배치들이 표준 정규 분포를 각각 따르게 되기 때문이지요. 따라서 평균과 분산을 구할때에도 나눠주는 값이 $$ B $$ 입니다.
- 학습 단계에서 모든 Feature에 정규화를 해주게 되면 정규화로 인하여 Feature가 동일한 Scale이 되어 learning rate 결정에 유리해집니다.
- 왜냐하면 Feature의 Scale이 다르면 gradient descent를 하였을 때, gradient가 다르게 되고 같은 learning rate에 대하여 weight마다 반응하는 정도가 달라지게 됩니다.
    - gradient의 편차가 크면 gradient가 큰 weight에서는 gradient exploding이, 작으면 vanishing 문제가 발생하곤 합니다.
- 하지만 정규화를 해주면 gradient descent에 따른 weight의 반응이 같아지기 때문에 학습에 유리해집니다. 
- 여기서 사용된 값 중 $$ \gamma, \bete $$의 역할을 확인하는 것이 필요합니다.
- 먼저 batch normalization은 activation function 앞에 적용됩니다.
- batch normalization을 적용하면 weight의 값이 평균이 0, 분산이 1인 상태로 분포가 되어지는데, 이 상태에서 ReLU가 activation으로 적용되면 전체 분포에서 음수에 해당하는 (1/2 비율) 부분이 0이 되어버립니다. 기껏 정규화를 했는데 의미가 없어져 버리게 됩니다.
- 따라서 $$ \gamma, \bete $$가 정규화 값에 곱해지고 더해져서 ReLU가 적용되더라도 기존의 음수 부분이 모두 0으로 되지 않도록 방지해 주고 있습니다. 물론 이 값은 학습을 통해서 효율적인 결과를 내기 위한 값으로 찾아갑니다. 

<br>

### **추론 단계의 배치 정규화**

<br>

- 이번에는 추론 단계에서의 배치 정규화에 대하여 알아보도록 하겠습니다. 수식으로 먼저 살펴보면

<br>

$$ BN(X) = \gamma \Bigl(  \frac{X - \mu_{BN} }{\sigma_{BN} } \Bigr) + \beta $$

<br>


$$ \mu_{BN} = \frac{1}{N} \sum_{i} \mu_{batch}^{i} $$

<br>

$$ \sigma^{2}_{BN} = \frac{1}{N} \sum_{i} \sigma_{batch}^{i} $$

<br>

- 추론 과정에서는 BN에 적용할 평균과 분산에 고정값을 사용합니다.
- 이 때 사용할 고정된 평균과 분산은 학습 과정에서 이동 평균(moving average) 또는 지수 평균(exponential average)을 통하여 계산한 값입니다. 즉, 학습 하였을 때의 최근 $$ N $$ 개에 대한 평균 값을 고정값으로 사용하는 것입니다. 
    - 이동 평균을 하면 $$ N $$개 이전의 평균과 분산은 미반영 되지만 지수 평균을 사용하면 전체 데이터가 반영됩니다.
- 그리고 이 때 사용되는 $$ \gamma, \beta $$는 학습 과정에서 학습한 파라미터 입니다. 
- 중요한 것은 학습 과정과 추론 과정의 알고리즘이 다르므로 framework에서 사용할 때, `학습`과정인지 `추론`과정인지에 따라 다르게 동작하도록 관리를 잘 해주어야 한다는 것입니다.
