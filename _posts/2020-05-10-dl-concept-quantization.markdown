---
layout: post
title: Quantization과 Quantization Aware Training
date: 2020-05-30 00:00:00
img: dl/concept/quantization/0.png
categories: [dl-concept]
tags: [dilated residual network, DRN] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 참조 : https://arxiv.org/pdf/1806.08342.pdf (Quantizing deep convolutional networks for
efficient inference: A whitepaper)
- 참조 : https://youtu.be/Oh1pLlir39Q
- 참조 : https://leimao.github.io/article/Neural-Networks-Quantization/
- 참조 : https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/
- 참조 : https://spell.ml/blog/pytorch-quantization-X8e7wBAAACIAHPhT
- 참조 : https://www.youtube.com/playlist?list=PLC89UNusI0eSBZhwHlGauwNqVQWTquWqp
- 참조 : https://wannabeaprogrammer.tistory.com/42

<br>

## **목차**

<br>

- ### Quantization 이란
- ### Quantization의 방법
- ### QAT (Quantization Aware Training) 방법
- ### Post Training Quantization과 Quantization Aware Training 비교

<br>

## **Quantization 이란**

<br>

- Quantization은 실수형 변수(floating-point type)를 정수형 변수 (integer or fixed point)로 변환하는 과정을 뜻합니다.

<br>
<center><img src="../assets/img/dl/concept/quantization/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이 작업은 weight나 activation function의 값이 **어느 정도의 범위 안에 있다는 것을 가정**하여 이루어 지는 모델 경량화 방법입니다.
- 위 그림과 같이 floating point로 학습한 모델의 weight 값이 -10 ~ 30 의 범위에 있다고 가정하겠습니다. 이 때, 최소값인 -10을 uint8의 0에 대응시키고 30을 uint8의 최대값인 255에 대응시켜서 사용한다면 32bit 자료형이 8bit 자료형으로 줄어들기 때문에 전체 메모리 사용량 및 수행 속도가 감소하는 효과를 얻을 수 있습니다.

<br>

- 이와 같이 Quantization을 통하여 효과적인 모델 최적화를 할 수 있는데, float 타입을 int형으로 줄이면서 용량을 줄일 수 있고 bit 수를 줄임으로써 계산 복잡도도 줄일 수 있기 때문입니다. (일반적으로 정수형 변수의 bit 수를 N배 줄이면 곱셈 복잡도는 N*N배로 줄어듭니다.)
- 또한 정수형이 하드웨어에 좀 더 친화적인 이유도 있기 때문에 Quantization을 통한 최적화가 필요합니다.
- 정리하면 `① 모델의 사이즈 축소`, `② 모델의 연산량 감소`, `③ 효율적인 하드웨어 사용`이 Quantization의 주요 목적이라고 말할 수 있습니다.

<br>
<center><img src="../assets/img/dl/concept/quantization/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 예를 들어 위 그림의 형광펜을 칠한 위치를 살펴보겠습니다. `Precision`에 해당하는 2, 3, 4, 8은 Quantization 하였을 때, 사용한 `bit`가 됩니다. ResNet-34를 2 bit로 표현하였을 때의 Top-1 Accuracy가 ResNet-18을 4-bit로 표현하였을 때 보다 성능이 더 좋은 것을 알 수 있습니다. 이 때 모델 사이즈는 오히려 ResNet-34가 조금 더 가벼운 것도 확인할 수 있습니다.
- 즉, 여기서 얻을 수 있는 교훈은 작은 네트워크로 quantization을 대충하는 것 보다 큰 네트워크로 quantization을 더 잘하는게 성능 및 모델 사이즈 측면에서 더 좋을 수 있다는 점입니다.

<br>

- **Quantization의 의미를 수식으로** 알아보도록 하겠습니다. floating point 값인 $$ x \in [\alpha, \beta] $$ 를 quantized가 적용된 값인 $$ x_q \in [\alpha_q, \beta_q] $$ 라고 한다면 `de-quantization`의 과정은 다음과 같습니다.

<br>

- $$ x = c (x_q + d) $$

<br>

- 반대로 `quantization`의 과정은 다음과 같이 나타낼 수 있습니다.

<br> 

- $$ x_q = \text{round}\big(\frac{1}{c} x - d\big) $$

<br>

- 위 식에서 $$ c, d $$는 Quantization 시 필요한 변수입니다.

<br>

- 구해야 하는 변수가 $$ c, d $$ 2개이므로 2개의 식을 통해 연립방정식 형태로 $$ c, d $$를 구할 수 있습니다.

<br>

- $$ \begin{align} \beta &= c (\beta_q + d) \\ \alpha &= c (\alpha_q + d) \\ \end{align} $$

<br>

- 위 식과 같이 2개의 $$ \alpha, \beta $$식이 있다고 가정하면 다음과 같이 식을 전개할 수 있습니다.

<br>

- $$ \begin{align} c &= \frac{\beta - \alpha}{\beta_q - \alpha_q} \\ d &= \frac{\alpha \beta_q - \beta \alpha_q}{\beta - \alpha} \\ \end{align} $$

<br>

- 이러한 Quantization 과정 거칠 때, floating point $$ 0 $$은 quantization error가 없다고 가정합니다. 이 때, $$ x = 0 $$을 $$ x_{q} $$로 나타내면 다음과 같습니다.

<br> 

- $$ \begin{align} x_q &= \text{round}\big(\frac{1}{c} 0 - d\big) \\ &= \text{round}(- d) \\ &= - \text{round}(d) \\ &= - d \\ \end{align} $$

<br>

- 위 식은 다음과 같은 의미를 가집니다.

<br>

- $$ \begin{align} d &= \text{round}(d) \\ &= \text{round}\big(\frac{\alpha \beta_q - \beta \alpha_q}{\beta - \alpha}\big) \\ \end{align} $$

<br>

- Quantization을 나타낼 때, 관습적으로 위 식에서 $$ c $$는 `scale` $$ s $$로 나타내고 $$ -d $$는 `zero point` $$ z $$로 나타냅니다.
- 정리하면  `de-quantization`과 `quantizaton`과 $$ s, z $$는 다음과 같이 나타낼 수 있습니다.

<br>

- $$ \text{de-quantization : } x = s (x_q - z) $$

- $$ \text{quantization : } x_q = \text{round}\big(\frac{1}{s} x + z\big) $$

- $$ \begin{align} s &= \frac{\beta - \alpha}{\beta_q - \alpha_q} \\ z &= \text{round}\big(\frac{\beta \alpha_q - \alpha \beta_q}{\beta - \alpha}\big) \\ \end{align} $$

<br>

- 위 식에서 $$ z $$ 는 `integer` 이고 $$ s $$는 `positive floating point`입니다.

<br>


## **Post Training Quantization과 Quantization Aware Training 비교**

<br>

- 이번에는 `QAT(Quantization Aware Training)`에 대하여 알아보려고 합니다. 그전에 Post Training Quantization과 Quantization Aware Traing에 대한 비교를 통하여 어떤 차이점이 있는 지 차이점을 먼저 비교하여 **QAT의 필요성에 대하여 먼저 느껴보도록 하겠습니다.**

<br>

- `PTQ(Post Training Quantization)` : floating point 모델로 학습을 한 뒤 결과 weight값들에 대하여 quantization하는 방식으로 학습을 완전히 끝내 놓고 quantization error를 최소화 하도록 합니다.  
    - 장점 : 파라미터 size 큰 대형 모델에 대해서는 정확도 하락의 폭이 작음
    - 단점 : 파라미터 size가 작은 소형 모델에 대해서는 정확도 하락의 폭이 큼
- `QAT(Quantization Aware Training)` : 학습 진행 시점에 inference 시 quantization 적용에 의한 영향을 미리 시뮬레이션을 하는 방식으로 최적의 weight를 구하는 것과 동시에 quantization을 하는 방식을 뜻합니다.
    - 장점 : Quantization 이후 모델의 **정확도 감소 폭을 최소화** 할 수 있음
    - 단점 : 모델 학습 이후 추가 학습이 필요함

<br>
<center><img src="../assets/img/dl/concept/quantization/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 도표는 mobilenet v2와 inception v3에 대하여 기존 성능, QAT, PTQ의 Accuracy 성능에 대하여 나타냅니다.
- QAT의 경우 두 모델 모두 성능 하락의 폭이 거의 없다고 말할 수 있습니다.
- 반면 PTQ의 경우 mobilenet v2에서 큰 성능 하락이 발생하였습니다. 이것이 앞에서 언급한 소형 모델에 대해서는 정확도 하락의 폭이 큰 것에 해당합니다. 하지만 모델의 크기가 큰 inception v3에서는 PTQ에서도 성능 하락이 크지 않는 것도 관측할 수 있습니다.
- 이와 같이 소형 모델에 대한 성능 하락이 크게 감소하는 이유는 **소형 모델에서는 weight 및 layer의 갯수가 작으므로 weight가 가지는 정보의 양이 상대적으로 작다**는 것에 원인이 있습니다. 따라서 소형 모델의 경우 여러가지 에러 및 outlier의 상황에 민감해 지는데, **Quantization으로 인하여 학습을 통해 얻은 실제 weight와 달라지는 경우 그 영향이 더욱 커져서** PTQ에 의한 성능 감소가 더 크게 발생할 수 있습니다.
- 특히 Quantization을 해야 하는 상황은 Edge device와 같은 환경에서 사용해야하는 경우가 많은데 이 경우 소형 모델을 사용해야 하므로 `PTQ`를 이용하면 좋은 Quantization 성능을 기대하기 어렵습니다.
- 따라서 학습을 하면서 Quantization을 시뮬레이션하는 `QAT`를 이용하여 **Quantization으로 인하여 학습을 통해 얻은 실제 weight와 달라지는 상황을 최소화 하는 작업이 필요**합니다. 이러한 이유로 소형 모델에서는 특히 QAT가 필수적이라고 말할 수 있습니다.

<br>

- 다음으로 간략하게 구조적으로 `PTQ`와 `QAT`의 구조적인 차이점을 살펴보겠습니다.

<br>
<center><img src="../assets/img/dl/concept/quantization/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서 왼쪽 (a)는 단순히 int형으로 quantization하여 사용하는 방식이고 오른쪽 (b)는 `QAT`를 사용한 방식입니다.
- (b)를 보면 빨간색에 해당하는 `act quant (activation quantization)`과 `wt quant (weight quantization)`이 추가된 것을 확인할 수 있습니다.
- 앞에서 설명하였듯이 `QAT`는 forward / backward 에서 `weight`와 `activation function` 출력에 대한 quantization을 학습 중에 시뮬레이션 합니다.
- 위의 빨간색 노드를 `fake quantization node`라고 하며 forward / backward pass에서 quantization 적용 시의 영향을 시뮬레이션하게 됩니다.
- 추가적으로 batch normalization은 inference에 사용되는 모드를 적용하여 시뮬레이션을 하게 됩니다.

<br>

## **QAT (Quantization Aware Training) 방법**

<br>

- static quantization 을 통하여 inference 중에 매우 효율적인 quantized된 int형 모델을 만들 수 있습니다.
- 하지만 post-training calibration을 하더라도 모델의 정확도가 감소할 수 있고 때때로 감소의 폭이 커서 큰 성능 하락이 발생할 수도 있습니다.
- 이러한 경우 post-training calibration 만으로는 quantized int형 모델을 만드는 데 충분하지 않을 수 있으므로 `QAT(Quantization Aware Training)`를 사용해 보고자 합니다. `QAT`는 학습 중에 quantization 효과를 모델링 할 수 있습니다.

<br>

- `QAT`의 메커니즘은 간단합니다. floating point 모델에서 quantized된 정수 모델로 변환하는 동안 quantized가 발생하는 위치에 fake quantization module, 즉 quantization 및 dequantization 모듈을 배치하여 integer quantization에 의해 가져오는 [클램핑](https://en.wikipedia.org/wiki/Clamping_(graphics)) 및 반올림의 효과를 시뮬레이션합니다.
- fake quantization 모듈은 `scale`과 `weight` 및 `activation`의 zero-point를 모니터링합니다. 
- `QAT`가 일단 끝나면 floating point 모델은 fake quantization 모듈에 저장된 정보를 사용하여 quantized integer model로 변환될 수 있습니다.


<br>



<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>
