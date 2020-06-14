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
- ### pytorch 셋팅 관련 코드
- ### GPU 셋팅 관련 코드
- ### torch.argmx(input, dim, keepdim)
- ### torch.from_numpy(numpy.ndarray)

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
import torch.backends.cudnn as cudnn # cudnn을 다루기 위한 값 모음

from torchsummary import summary # summary를 통한 model의 현황을 확인 하기 위함
import torch.onnx # model을 onnx 로 변환하기 위함
```

<br>

## **pytorch 셋팅 관련 코드**

<br>

```python
# pytorch 내부적으로 사용하는 seed 값 설정
torch.manual_seed(seed)
```

<br>

## **GPU/CPU Device 세팅 코드**

<br>

```python
# cuda가 사용 가능한 지 확인
torch.cuda.is_available()

# cuda가 사용 가능하면 device에 "cuda"를 저장하고 사용 가능하지 않으면 "cpu"를 저장한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 현재 PC의 사용가능한 GPU 사용 갯수 확인
torch.cuda.device_count()

# cudnn을 사용하도록 설정. GPU를 사용하고 있으면 기본값은 True 입니다.
import torch.backends.cudnn as cudnn
cudnn.enabled = True

# inbuilt cudnn auto-tuner가 사용 중인 hardware에 가장 적합한 알고리즘을 선택하도록 허용합니다.
cudnn.benchmark = True
```

<br>

- 위 코드와 같이 device의 유형을 선택하면 GPU가 존재하면 `cuda:0`에 할당되고 GPU가 없으면 `cpu`에 할당 되도록 할 수 있습니다.

<br>

## **torch.argmx(input, dim, keepdim)**

<br>

- 딥러닝 각 framework마다 존재하는 `argmax` 함수에 대하여 다루어 보겠습니다. keras, tensorflow와 사용법이 유사하니 참조하시면 됩니다. 아래 링크는 pytorch의 링크입니다.
    - pytorch 링크 : https://pytorch.org/docs/stable/torch.html
- argmax 함수가 받는 argument는 차례대로 `input`, `dim`, `keepdim` 입니다. `input`은 Tensor를 나타내고 `dim`은 몇번 째 축을 기준으로 argmax 연산을 할 지 결정합니다. 마지막으로 `keepdim`은 argmax 연산을 한 축을 생략할 지 그대로 둘 지에 대한 기준이 됩니다. argmax를 하면 각 축마다 값이 1개만 남게 되므로 필요 여부에 따라 남길 수도 있고 삭제 할 수도 있습니다.

<br>

- pytorch를 이용하여 이미지 처리를 할 때, 주로 사용하는 방법은 다음과 같습니다.

<br>

```python
# 1. input이 (channel, height, widht) 인 경우
torch.argmax(input, dim = 0, keepdim = True)

# 2. input이 (batch, channel, height, width) 인 경우
torch.argmax(input, dim = 1, keepdim = True) 
```

<br>

- 일반적으로 이미지 처리를 할 때, 출력의 `channel`의 갯수 만큼 클래스 label을 가지고 있는 경우가 많습니다. 이 때, 가장 큰 값을 argmax 함으로써 가장 큰 인덱스를 구할 수 있습니다.
- 예를 들어 segmentation을 하는 경우 위의 코드와 같은 형태가 그대로 사용될 수 있습니다. 1번 케이스의 경우 batch가 고려되지 않은 것이고 2번 케이스의 경우 batch가 고려된 것입니다. segmentation의 경우 이미지의 height, width의 크기에 channel의 갯수가 label의 갯수와 동일하게 되어 있습니다. 그 중 가장 큰 값을 가지는 channel이 그 픽셀의 label이 되게 됩니다.
- 따라서 argmax를 취하면 `channel`은 1로 되고 height와 width의 크기는 유지됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/snippets/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림을 예제로 이용하여 각 축에 대하여 argmax한 결과를 알아보도록 하겠습니다.
- 아래 예제 코드에서는 0번째 (channel), 1번째 (height), 2번째 (width) 방향으로 각각 argmarx를 한 것입니다. 또한 `keepdim`을 기본값인 False로 둘 경우와 True로 둘 경우를 구분하여 어떻게 `shape`이 변화하는 지 살펴보았습니다.

<br>

```python

# channel : 3, height : 4, width : 2로 가정합니다.
>> A = torch.randint(10, (3, 4, 2))
>> print(A)

tensor([[[6, 4],
         [8, 0],
         [8, 4],
         [2, 4]],

        [[2, 8],
         [3, 6],
         [4, 7],
         [0, 9]],

        [[8, 0],
         [3, 9],
         [0, 5],
         [7, 3]]])

# 0번째 축(channel) 기준 argmax w/o Keepdim
 >> torch.argmax(A, dim=0)
tensor([[2, 1],
        [0, 2],
        [0, 1],
        [2, 1]])

>> torch.argmax(A, dim=0).shape
torch.Size([4, 2])

# 0번째 축(channel) 기준 argmax w/ Keepdim
>> torch.argmax(A, dim=0, keepdim = True)
tensor([[[2, 1],
         [0, 2],
         [0, 1],
         [2, 1]]])

 >> torch.argmax(A, dim=0, keepdim = True).shape
 torch.Size([1, 4, 2])

# 1번째 축(height) 기준 argmax w/o Keepdim
>> torch.argmax(A, dim=1)
tensor([[2, 3],
        [2, 3],
        [0, 1]])
 
 >> torch.argmax(A, dim=1).shape
torch.Size([3, 2])


# 1번째 축(height) 기준 argmax w/ Keepdim
>> torch.argmax(A, dim=1, keepdim = True)
tensor([[[2, 3]],

        [[2, 3]],

        [[0, 1]]])

>> torch.argmax(A, dim=1, keepdim = True).shape
torch.Size([3, 1, 2])


# 2번째 축(width) 기준 argmax w/o Keepdim
>> torch.argmax(A, dim=2)
tensor([[0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0]])

>> torch.argmax(A, dim=2).shape
torch.Size([3, 4])

# 2번째 축(width) 기준 argmax w/ Keepdim
>> torch.argmax(A, dim=2, keepdim=True)
tensor([[[0],
         [0],
         [0],
         [1]],

        [[1],
         [1],
         [1],
         [1]],

        [[0],
         [1],
         [1],
         [0]]])


>> torch.argmax(A, dim=2, keepdim=True).shape
torch.Size([3, 4, 1])

```

<br>
<center><img src="../assets/img/dl/pytorch/snippets/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 보면 왼쪽 부터 0번째 축, 1번째 축, 2번째 축 방향을 기준으로 argmax하였을 때 선택된 결과를 노란색 음영으로 표시하였습니다. 위 코드 결과를 그대로 표현한 것으로 이해하는 데 참조하시기 바랍니다.

<br>

- 이번에는 간단하게 height와 width만 고려하여 다루어 보겠습니다.

<br>

```python
# 간단하게 height, width의 크기만을 이용하여 다루어 보겠습니다.
>> B = torch.rand(3, 2)

tensor([[0.8425, 0.3970],
        [0.5268, 0.7384],
        [0.5639, 0.3080]])

# 1번 예제. 아래 첫번째 그림 참조
>> torch.argmax(B, dim=0)
tensor([0, 1])

>> torch.argmax(B, dim=0, keepdim=True)
tensor([[0, 1]])

# 2번 예제. 아래 두번째 그림 참조
>> torch.argmax(B, dim=1)
tensor([0, 1, 0])

>> torch.argmax(B, dim=1, keepdim=True)
tensor([[0],
        [1],
        [0]]) 
```

<br>

- 위의 1번 예제에 해당하는 그림입니다. 매트릭스에서 0번째 축은 세로(height)축입니다. 따라서 각 열에서 세로 방향으로 최대값이 선택됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/snippets/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 다음으로 2번 예제에 해당하는 그림입니다. 매트릭스에서 1번째 축은 가로(width)축입니다. 따라서 각 행에서 가로 방향으로 최댁밧이 선택됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/snippets/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

## **torch.from_numpy(numpy.ndarray)**

<br>

- torch에서 numpy를 이용해 선언한 데이터를 Tensor 타입으로 변환하려면 아래와 같이 사용할 수 있습니다.

<br>

```python
A = np.random.rand(3, 100, 100)
torch.from_numpy(A)
```
