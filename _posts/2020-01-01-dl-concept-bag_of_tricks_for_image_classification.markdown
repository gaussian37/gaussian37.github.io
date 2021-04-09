---
layout: post
title: Bag of Tricks for Image Classification with Convolutional Neural Networks
date: 2020-01-01 00:00:00
img: dl/concept/bag_of_tricks/0.png
categories: [dl-concept] 
tags: [딥러닝, bag of tricks] # add tag
---

<br>


- 논문 : https://arxiv.org/abs/1812.01187
- 참조 : https://www.youtube.com/watch?v=D-baIgejA4M&list=WL&index=34&t=0s
- 참조 : https://www.youtube.com/watch?v=2yxsg_aMxz0
- 참조 : https://hoya012.github.io/blog/Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks-Review/

<br>

- 19년도 CVPR에 발표된 Bag of Tricks 논문 입니다. 개인적으로 이런 논문은 너무 좋습니다. 블로그에서 핵심 내용만 뽑아준 것 같은 느낌이 들기 때문입니다.
- 이 논문은 네트워크 구조를 고치지 않고 ResNet-50을 기준으로 **Trick 들만 적용하여 Accuracy와 학습속도를 높인 방법론**을 정리한 내용을 다룹니다.
- 한번 쯤 들어본 내용들을 정리해서 다루었기 때문에 내용이 크게 어렵지 않을 것이라 생각 됩니다.

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- Trick을 모두 적용하였을 때, 위 성능 지표와 같이 성능을 향상할 수 있었습니다.
- 먼저 논문에서 사용한 다양한 전처리 등은 생략하도록 하겠습니다. 자세한 내용은 논문 또는 위의 참조 링크를 참조하시기 바랍니다.
- 이 글에서 자세히 다루어 볼 것은 실제 `Trick` 내용만 작성하였습니다.

<br>

## **Efficient Training**

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 논문에서 사용된 기본적인 학습 절차 (base line)은 위 알고리즘을 따릅니다.

<br>

#### **Linear scaling learning rate**

<br>

- batch size가 증가하게 되면, 그 만큼 linear 하게 learning rate를 증가시켜주는 것이 학습하는 데 좋다는 논문을 반영합니다.
    - 링크 : [Accurate, large minibatch SGD: training imagenet in 1 hour](https://arxiv.org/pdf/1706.02677.pdf)
- 추천하는 방식으로는 `initial learning rate' ← initial learning rate * batch size / 256`입니다. 예를 들어 초기 learning rate가 0.1이고 batch size가 1024이면 `0.1 * 1024 / 256 = 0.4`로 초기 learning rate를 적용해야 함을 의미합니다. 이 방식을 통해 batch size가 커져서 성능이 떨어지는 문제를 개선할 수 있다고 설명합니다.

<br>

#### **Learning rate warm-up**

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 다음으로 Learning rate warm-up과 관련된 내용입니다. 보통 learning rate는 초기 설정한 값을 기준으로 계속 줄여주는 방식을 많이 사용하고 있습니다.
- 이와는 반대로 초기에 learning rate를 0으로 설정하고 이를 일정 기간 동한 linear 하게 키워주는 방식을 learning rate warmup 방식으로 소개합니다.
- 위 그래프에서는 5 epoch 동안 조금씩 learning rate를 키워서 initial learning rate 값을 5 epoch에 0.4까지 도달하게 만든 후 점점 learning rate가 줄어들도록 변경하였습니다.
- 위 방법은 이 글의 뒷부분에 제시된 `Cosine Learning Rate Decay` 부분과 혼합하여 사용할 수 있으며 그 코드는 다음 링크를 참조하시기 바랍니다.
    - 링크 : [https://gaussian37.github.io/dl-pytorch-lr_scheduler/](https://gaussian37.github.io/dl-pytorch-lr_scheduler/)

<br>

#### *Zero Gamma in BatchNorm**

<br>

- 먼저 [Batch Normalization](https://gaussian37.github.io/dl-concept-batchnorm/)에 대한 글을 읽고 오시길 추천 드립니다.
- Batch Normalization을 사용하면 대표적으로 다음과 같은 장점이 있습니다.
    - ① 학습 속도를 빠르게 할 수 있습니다.
    - ② weight initialization에 대한 민감도를 감소 시킬 수 있습니다.
    - ③ regularization 효과가 있습니다.

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- Batch Normalization을 적용할 때, 위 식과 같이 $$ \gamma, \beta $$가 사용되며 학습하면서 결정되는 파라미터입니다. 이 값은 `scale`과 `shift`를 변경해주는 값으로 입력 차원이 $$ k $$이면 두 파라미터의 차원도 $$ k $$를 가집니다. 
- 두 파라미터를 사용하는 이유 중의 하나로 Batch Normalization로 인한 activation function의 non-linearity 영향력 감소 문제를 개선하기 위함이 있습니다.

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 예를 들면 위 그림과 같은 시그모이드 함수를 activation function으로 사용하였을 때, Batch Normalization의 결과에 scale과 shift를 적용하지 않으면 위 그림의 빨간색 영역만 그대로 사용할 수 있습니다. 그러면 non-lineariry의 효과가 줄어들게 됩니다. 이 문제를 해결하는 역할로 $$ \gamma, \beta $$가 사용됩니다.
- 이 값을 일반적으로 $$ \gamma = 0, \beta = 1 $$로 초기화 하고 학습을 하는데 이 논문에서는 $$ \gamma = 0 $$로 초기화 하는 것을 추천합니다.

<br>
<center><img src="../assets/img/dl/concept/bag_of_tricks/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 만약 $$ \gamma = 0 $$으로 두면 학습 시작할 때, 위 그림과 같은 ResNet 구조에서 Residual Block 내부의 weight에 0이 곱해지므로 무시하게 됩니다. 따라서 `Identity mapping`으로만 weight가 전달됩니다.
- 이와 같은 방법을 사용하면 학습 시작할 때 네트워크의 구조를 단순화 함으로써 네트워크의 복잡도를 일시적으로 낮출 수 있고, backpropagation을 통해 의미 있는 weight update가 발생하면 그 때 부터 Residual Block 내부가 사용됨과 동시에 원래 설계한 네트워크의 복잡도를 사용할 수 있습니다.
- Pytorch의 `BatchNorm2d`는 다음 링크에서 사용 방법을 알 수 있습니다.
    - 참조 : [https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
    - 참조 : [https://discuss.pytorch.org/t/change-batch-norm-gammas-init/81278/3](https://discuss.pytorch.org/t/change-batch-norm-gammas-init/81278/3)
- `BatchNorm2d` 함수의 파라미터에 Gamma를 직접 제어하는 파라미터는 없으므로 Gamma에 직접 접근하여 값을 변경해주면 됩니다. 다음과 같습니다.

<br>

```python
def SetZeroGammaInBatchNorm2d(bn):
    bn.weight.data = torch.zeros(bn.weight.data.shape, requires_grad=True)
    return bn

bn = nn.BatchNorm2d(3, affine=True)
print(bn.weight, bn.bias)
# Parameter containing:
# tensor([1., 1., 1.], requires_grad=True)
# tensor([0., 0., 0.], requires_grad=True)

bn = SetZeroGammaInBatchNorm2d(bn)
print(bn.weight, bn.bias)
# Parameter containing:
# tensor([0., 0., 0.], requires_grad=True) ← Changed
# tensor([0., 0., 0.], requires_grad=True)
```

<br>

#### **No bias decay**

<br>

- 학습을 할 때, `L2 Regularization` 방법을 통하여 weight의 크기를 줄이는 방법을 많이 사용 합니다. 이 때, weight는 `weight`와 `bias`를 모두 포함하지만 이 논문에서는 오직 `weight`에만 decay를 주는 것을 권장합니다. (이 또한 **휴리스틱**한 방법으로 접근하였습니다.)
- 참조 논문은 [Highly scalable deep learning training system with mixed-precision: Training imagenet in four minutes.](https://arxiv.org/pdf/1807.11205.pdf)입니다.

<br>

#### **Low-precision training**

<br>

- 