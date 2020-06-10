---
layout: post
title: pytorch 코드 snippets
date: 2019-03-01 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, snippets] # add tag
---

<br>

- 이 글은 pytorch 사용 시 참조할 수 있는 코드들을 모아놓았습니다.
- 완전히 기본 문법은 [이 글](https://gaussian37.github.io/dl-pytorch-pytorch-tensor-basic/)에서 참조하시기 바랍니다.

<br>

## **목차**

<br>

- ### pytorch import 모음
- ### GPU/CPU Device 세팅
- ### tensor.argmx(input, dim, keepdim)

<br>

## **pytorch import 모음**

<br>

```python
import torch
import torchvision
import torch.nn as nn # neural network 모음. (e.g. nn.Linear, nn.Conv2d, BatchNorm, Loss functions 등등)
import torch.optim as optim # Optimization algorithm 모음, (e.g. SGD, Adam, 등등)
import torch.nn.functional as F # 파라미터가 필요없는 Function 모음
from torch.utils.data import DataLoader # 데이터 세트 관리 및 미니 배치 생성을 위한 함수 모음
import torchvision.datasets as datasets # 표준 데이터 세트 모음
import torchvision.transforms as transforms # 데이터 세트에 적용 할 수있는 변환 관련 함수 모음
from torch.utils.tensorboard import SummaryWriter # tensorboard에 출력하기 위한 함수 모음

from torchsummary import summary # summary를 통한 model의 현황을 확인 하기 위함
import torch.onnx # model을 onnx 로 변환하기 위함
```

<br>

## **GPU/CPU Device 세팅 코드**

<br>

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

<br>

- 위 코드와 같이 device의 유형을 선택하면 GPU가 존재하면 `cuda:0`에 할당되고 GPU가 없으면 `cpu`에 할당 되도록 할 수 있습니다.

<br>

## **tensor.argmx(input, dim, keepdim)**

<br>

- 딥러닝 각 framework마다 존재하는 `argmax` 함수에 대하여 다루어 보겠습니다. keras, tensorflow와 사용법이 유사하니 참조하시면 됩니다. 아래 링크는 pytorch의 링크입니다.
    - pytorch 링크 : https://pytorch.org/docs/stable/torch.html
- argmax 함수가 받는 argument는 차례대로 `input`, `dim`, `keepdim` 입니다. `input`은 Tensor를 나타내고 `dim`은 몇번 째 축을 기준으로 argmax 연산을 할 지 결정합니다. 마지막으로 `keepdim`은 argmax 연산을 한 축을 생략할 지 그대로 둘 지에 대한 기준이 됩니다. argmax를 하면 각 축마다 값이 1개만 남게 되므로 필요 여부에 따라 남길 수도 있고 삭제 할 수도 있습니다.

<br>

- pytorch를 이용하여 이미지 처리를 할 때, 주로 사용하는 방법은 다음과 같습니다.

<br>

```python
torch.argmax(input, dim = 0, keepdim = True) # input이 (channel, height, widht) 인 경우
torch.argmax(input, dim = 1, keepdim = True) # input이 (batch, channel, height, width) 인 
```

<br>

```python
>> A = torch.randn(3, 4, 2)
>> print(A)

tensor([[[-0.8375,  0.7884],
         [-0.8836, -1.0729],
         [ 1.0928,  0.5377],
         [-0.0916,  0.8920]],

        [[-1.5294,  0.1188],
         [-0.8611, -1.0401],
         [-0.3119, -0.6326],
         [ 0.5503,  0.5102]],

        [[-0.0564, -0.3603],
         [ 0.6313,  1.2920],
         [ 1.1927, -1.3898],
         [-0.1280, -1.7702]]])
         
 >> torch.argmax(A, dim=0)
  tensor([[1, 0],
        [1, 1],
        [1, 0],
        [2, 2]])
        
>> torch.argmax(A, dim=0, keepdim = True)
tensor([[[1, 0],
         [1, 1],
         [1, 0],
         [2, 2]]])
         
 >> torch.argmax(A, dim=0, keepdim = True).shape
 
 


```
