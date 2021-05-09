---
layout: post
title: Pytorch Loss function 정리
date: 2020-10-15 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, loss function, softmax, negative log likelihood] # add tag
---

<br>

[Pytorch 관련 글 목차](https://gaussian37.github.io/dl-pytorch-table/)

<br>

<br>
<center><img src="../assets/img/dl/pytorch/loss/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>

## **목차**

<br>

- ### Negative Log Likelihood Loss

<br>

## **Negative Log Likelihood Loss**

<br>

- 개념 설명 : [https://gaussian37.github.io/dl-concept-nll_loss/](https://gaussian37.github.io/dl-concept-nll_loss/)
- 위 개념 설명 내용에 해당하는 `NLLLoss`는 Pytorch에 기본적으로 적용이 되어 있으므로 다음 링크의 함수를 사용하면 됩니다.
    - 링크 : [https://pytorch.org/docs/master/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss](https://pytorch.org/docs/master/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)
- `NLLLoss`에서 사용하는 파라미터 중 가장 중요한 파라미터는 `weight`입니다. `NLLLoss`의 장점 중 하나인 `weight`는 클래스 불균형 문제를 개선하기 위하여 수동으로 weight 별 학습 비중의 스케일을 조정하기 위해 사용됩니다. 
- 예를 들어 클래스의 갯수가 3개이고 클래스 별 데이터의 갯수가 (10, 50, 100) 이라면 weight는 데이터 갯수와 역의 관계로 대입을 해주어야 합니다. 예를 들면 (1, 0.2, 0.1)와 같이 weight를 설정할 수도 있고 (1/10, 1/50, 1/100)과 같이 사용할 수도 있습니다. **핵심은 적은 갯수의 데이터에 해당하는 클래스에 높은 weight를 주어 학습 양을 늘리는 데 있습니다.**
    - 참조 : [https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch](https://datascience.stackexchange.com/questions/48369/what-loss-function-to-use-for-imbalanced-classes-using-pytorch)
    - 참조 : [https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/25](https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/25)

<br>

```python
import torch.nn as nn
import nn.functional as F

loss = nn.NLLLoss(weight)
# output shape : (Batch Size, C, d1, d2, ...)
loss(F.log_softmax(output, 1), targets)
```

<br>

[Pytorch 관련 글 목차](https://gaussian37.github.io/dl-pytorch-table/)

<br>
