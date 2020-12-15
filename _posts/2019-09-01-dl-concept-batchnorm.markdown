---
layout: post
title: 배치 정규화(Batch Normalization)
date: 2019-09-01 00:00:00
img: dl/concept/batchnorm/batchnorm.png
categories: [dl-concept] 
tags: [배치 정규화, 배치 노멀라이제이션, batch normalization] # add tag
---

<br>

### 참조 

<br>

- https://youtu.be/TDx8iZHwFtM?list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS
- https://blog.naver.com/infoefficien/221122737854 (저의 또 다른 블로그)
- 이번 글에서는 딥러닝 개념에서 중요한 것 중 하나인 `batch normalization`에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [Batch](#batch-1)
- ### [Internal Covariant Shift](#internal-covariant-shift-1)
- ### [Batch Normalization](#batch-normalization-1)
- ### [Internal Covariant Shift 더 알아보기](#internal-covariant-shift-더-알아보기-1)
- ### [Batch Normalization의 효과](#batch-normalization의-효과-1)
- ### [Pytorch에서의 사용 방법](#pytorch에서의-사용-방법)

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
- 위 gradient descent의 식을 보면 $$ \sum $$에 $$ j = Bi $$가 있는데 $$ B $$가 `batch`의 크기가 됩니다. 
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

- $$ BN(X) = \gamma \Bigl(  \frac{X - \mu_{batch} }{\sigma_{batch} } \Bigr) + \beta $$

<br>

- 여기서 $$ \gamma $$는 스케일링 역할을 하고 $$ \beta $$는 bias입니다. 물론 두 값 모두 backprpagation을 통하여 학습을 하게 됩니다.

<br>

- $$ \mu_{batch} = \frac{1}{B} \sum_{i} x_{i} $$

<br>

- $$ \sigma^{2}_{batch} = \frac{1}{B} \sum_{i} (x_{i} - \mu_{batch})^{2} $$

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

- $$ BN(X) = \gamma \Bigl(  \frac{X - \mu_{BN} }{\sigma_{BN} } \Bigr) + \beta $$

<br>


- $$ \mu_{BN} = \frac{1}{N} \sum_{i} \mu_{batch}^{i} $$

<br>

- $$ \sigma^{2}_{BN} = \frac{1}{N} \sum_{i} \sigma_{batch}^{i} $$

<br>

- 추론 과정에서는 `BN`에 적용할 평균과 분산에 고정값을 사용합니다.
- 왜냐하면 학습 단계에서는 데이터가 배치 단위로 들어오기 때문에 배치의 평균, 분산을 구하는 것이 가능하지만, **테스트 단계에서는 배치 단위로 평균/분산을 구하기가 어려워** 학습 단계에서 배치 단위의 평균/분산을 저장해 놓고 테스트 시에는 평균/분산을 사용합니다.
- 이 때 사용할 고정된 평균과 분산은 학습 과정에서 이동 평균(moving average) 또는 지수 평균(exponential average)을 통하여 계산한 값입니다. 즉, 학습 하였을 때의 최근 $$ N $$ 개에 대한 평균 값을 고정값으로 사용하는 것입니다. 
    - 이동 평균을 하면 $$ N $$개 이전의 평균과 분산은 미반영 되지만 지수 평균을 사용하면 전체 데이터가 반영됩니다.
- 그리고 이 때 사용되는 $$ \gamma, \beta $$는 학습 과정에서 학습한 파라미터 입니다. 
- 다시 한번 말하자면 `중요한 것`은 학습 과정과 추론 과정의 알고리즘이 다르므로 framework에서 사용할 때, `학습`과정인지 `추론`과정인지에 따라 다르게 동작하도록 관리를 잘 해주어야 한다는 것입니다. 즉, 추론 과정에서는 framework에서 옵션을 지정하여 평균과 분산을 **moving average/variance**를 사용하도록 해야합니다.
    - 개인적으로 이 옵션 설정을 잘못 해서 디버깅 한 적이 몇번 있어서.. 강조 합니다.

<br>

## **Internal Covariant Shift 더 알아보기**

<br>

- 앞에서 언급한 Internal Covariant Shift에 대하여 좀 더 자세하게 다루어 보도록 하겠습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>  

- Neural Netowork에서는 많은 파라미터들에 의해서 학습에 어려움이 있습니다.
- 위의 노드들을 이은 edge들이 파라미터이니 그 숫자가 꽤 많은것을 시작적으로도 알 수 있지요.
- 위 그림에서 $$ X $$는 Input 이고 $$ H $$는 Hidden layer 그리고 $$ O $$는 Output layer 입니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>  

- 딥러닝에서는 학습의 어려움의 정도가 좀 더 커졌는데 Hidden layer의 layer 수가 점점 더 증가하기 때문입니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>  

- 딥러닝에서 Layer가 많아질 때 학습이 어려워지는 이유는 `weight`의 미세한 변화들이 가중이 되어 쌓이면 Hidden Layer의 깊이가 깊어질수록 **그 변화가 누적되어 커지기 때문입니다.**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>  

- 예를 들어 학습 중에 weight들이 위와 같이 기존과는 다른 형태로 변형될 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>  

- 즉, 위와 같이 기존의 Hidden Layer와는 또 다른 Layer의 결과를 가지게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/9.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 논문의 저자는 이 문제를 **Internal Covariate Shift**라고 말합니다.
- 어떤 문제든지 `Variacne`는 문제를 일으키곤 합니다. 의도한 대로 움직이지 않으니 말이죠.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/11.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 각 layer의 **Input feature가 조금씩 변해서** Hidden layer에서의 Input feature의 변동량이 누적되게 되면 각 layer에서는 입력되는 값이 전혀 다른 유형의 데이터라고 받아들일 수도 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/12.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 예를 들어 Training 셋의 분포와 Test 셋의 분포가 다르면 학습이 안되는 것과 같이 같은 학습 과정 속에서도 각 layer에 전달되는 feature의 분포가 다르게 되면 학습하는 데 어려움이 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/13.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 이 문제는 training 할 때 마다 달라지게 되고 Hidden Layer의 깊이가 깊어질수록 변화가 누적되어 feature의 분포 변화가 더 심해지게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/14.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 이 문제를 다루기 위하여 처음에는 weight 초기화를 잘 해주는 방법 또는 작은 learning rate를 주어서 변화량을 줄이려는 방법이 사용되기도 하였습니다.
- 하지만 weight를 잘 주는 것은 어려운 방법이고 작은 learning rate를 사용하는 방법은 학습이 매우 더디게 되어 local minimum에 빠지는 위험도 존재하였습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/15.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 그래서 weight initialization 또는 learning rate 조절이 아닌 새로운 방법의 솔루션을 제시한 것이 `Batch Normalization`입니다.
- Hidden layer에서의 변화량에 너무 크지 않으면 학습도 안정하게 될 것이라는 컨셉입니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/16.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 위 그림에서 $$ x \to H_{1} $$의 부분을 살펴보면 Input $$ x $$에서 weight를 곱하고 activation을 취하여 최종적으로 Hidden layer $$ H_{1} $$을 가지게 됨을 알 수 있습니다.
- 여기서 변화하는 부분은 **weight에 따른 가중치가 되는 부분**입니다.
- Batch normalization에서는 이러한 weight에 따른 가중치의 변화를 줄이는 것을 목표로 합니다. 
- 따라서 **Activation 하기 전 값의 변화를 줄이는 것**이 Batch Normalization의 목표가 됩니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/17.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- weight와 Input의 연살 결과를 Batch Normalization함으로 scale을 줄게 됩니다. 따라서 변화가 줄어들게 되는 것이지요.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/18.png" alt="Drawing" style="width: 800px;"/></center>
<br> 

- 왼쪽은 Batch Normalization을 적용한 케이스이고 오른쪽은 적용하지 않은 케이스입니다.
- 그래프의 x축은 wx를 뜻하고 y축은 activation으로 변환된 값입니다.
- BN의 경우 Scale을 줄이기 때문에 분포의 variance가 작아지게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/19.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 새로운 layer가 추가되면 BN이 적용된 것과 안된것의 차이가 더 벌어지게 됩니다.
- BN을 사용하지 않으면 변화값들이 점점 더 가중되면서 기존에 학습하였던 분포와는 다른 분포를 가지게 됩니다. 

<br>
<center><img src="../assets/img/dl/concept/batchnorm/20.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 일반적으로 Batch Normalization에서의 Scale은 평균 0, 분산 1을 사용하고 있습니다. 
- 즉, 엔지니어링에서 많이 사용하고 있는 `표준 정규 분포`를 따르도록 하고 있는 것이지요.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/21.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Batch Normalization의 output은 Activation이 Input으로 받고 있습니다.
- 예를 들어 `ReLU`가 BN의 output을 어떻게 받아들이는 지 해석해 보면 ReLU에서는 0보다 큰 값은 linear하게 전달하기 때문에 Non linearity를 주지 못하고 0보다 작은 값은 0으로 수렴해 버리기 때문에 0을 기준으로 대칭되도록 하는 것이 의미가 있습니다. 즉, 너무 양수만 나오도록 또는 너무 음수만 나오도록 하는것을 지양합니다.
- 이 의도가 평균이 0이 되도록 하는 것과 일치 합니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/22.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 물론 이 해석은 sigmoid에서도 적용될 수 있습니다. sigmoid도 입력이 0 근처에서 의미가 있고 큰 양수나 음수가 되면 1 또는 0으로 수렴해 버리기 때문에 의미가 없어져 버립니다. Vanishing Gradient 문제도 발생하기도 하구요.
- 따라서 sigmoid 경우에도 BN의 output이 평균이 0이고 그 근처에 유의미한 값의 variance를 갖는 것이 의미가 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/23.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 분산이 1인 이유에 대해서는 경험적으로 표준 정규 분포에 따랐다는 의견도 있고 적절한 값을 선택하였다는 의견도 있습니다.
- 중요한 것은 **BN의 목적이 Variance를 줄이는 것**이므로 **크지않은 Variance를 선택하는 것**이 핵심입니다. 물론 너무 작은 Variance를 주게 되면 Activation의 Input이 0에 수렴해 버리므로 적당한 선의 Variance를 선택하는 것이 중요하겠습니다. (그것이 경험상 1을 주자고 한 것입니다.)

<br>
<center><img src="../assets/img/dl/concept/batchnorm/24.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 앞의 글에서 편의를 위하여 layer 단으로 Batch Normalization을 표현해 보았는데, BN이 적용되는 것은 Layer 기준으로 적용되는 것이 아니라 `Layer 내부의 노드 기준`으로 Normalization을 하게 됩니다.
- 위 그림과 같이 $$ x_{i} $$와 $$ z_{i} $$의 weighted sum한 결과를 BN 하게 됩니다. 
- 위 식에서 추가된 $$ \eta $$는 분산이 0에 가까워 졌을 때, divide by zero를 방지 하기 위하여 추가된 상수입니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/25.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Batch Normalization 식 전체를 보면 위와 같습니다.
- 앞에서 설명한 바와 같이 마지막에 $$ \gamma $$와 $$ \beta $$가 추가된 것을 보실 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/26.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- ReLU에 적용하는 경우 기본적인 Batch Normalization은 음수에 해당하는 범위의 반이 0이 되어버릴 수 있습니다.
- 기껏 Normalization을 해주었는데 반을 잃어버리면 낭비가 심하니 일부 비 대칭적으로 사용할 수 있게 $$ \gamma $$와 $$ \beta $$를 이용하여 더 많은 영역을 activation이 값으로 가져갈 수 있게 합니다.
- 물론 $$ \gamma $$와 $$ \beta $$는 학습을 통하여 값이 결정됩니다.

<br>

## **Batch Normalization의 효과**

<br>
<center><img src="../assets/img/dl/concept/batchnorm/27.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이렇게 Batch Normalization을 적용하면 Internal Covariate Shift를 개선하고자 고려했던 weight initialization과 learning rate를 감소하는 것에서 다소 자유로워질 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/batchnorm/28.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- Batch Normalization의 또 다른 장점으로는 Regularization Effect가 있습니다. 
- Mini-batch에 들어가는 평균과 분산이 지속적으로 변화는 과정 속에서 분포가 조금씩 바뀌게 되고 학습하는 과정에 weight에 영향을 주게 되기 때문입니다.
- 만약 평균과 분산이 고정이라면 학습 하는 동안 계속 고정된 값이 연산되기 때문에 특정 weight에 큰 값 즉, 가중치가 큰 weight가 발생할 수 있습니다. 하지만 Batch Normalization에서 평균과 분산이 지속적으로 변하고 weight 업데이트에도 계속 영향을 주어 한 weight에 가중치가 큰 방향으로만 학습되지 않기 때문에 Regularization effect를 얻을 수 있습니다.
- 이와 같은 효과를 얻기 위해 적용하는 Dropout을 Batch Normalization을 통하여 얻을 수 있기 때문에 이론적으로는 안써도 된다고는 하지만 Batch Normalization - Activation 이후에 Dropout을 써서 효과를 더 낼 수 도 있기 때문에 상황에 맞춰서 사용하시면 됩니다.

<br>

- 정리하면 Batch Normalization은 **Internal Covariate Shift 문제를 개선하기 위하여 고안**되었고 이것은 **Activation에 들어가게 되는 Input의 Range를 제한**시키는 방법으로 문제를 개선하였습니다.
- 그 결과 학습을 좀 더 안정적으로 할 수 있게 되었고 Internal Covariate Shift 문제를 해결하기 위한 과거의 개선책인 **weight initialization과 small learning rate에서 좀 더 자유로워**졌습니다.
- 또한 **Regularization effect** 까지 있어서 **overfitting 문제에 좀 더 강건**해 질 수 있습니다.

<br>

## **Pytorch에서의 사용 방법**

<br>