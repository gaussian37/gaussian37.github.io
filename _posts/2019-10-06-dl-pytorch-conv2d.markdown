---
layout: post
title: Pytorch Conv2d 함수 다루기
date: 2019-09-27 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, conv2d] # add tag
---

<br>

[Pytorch 관련 글 목록](https://gaussian37.github.io/dl-pytorch-table/)

<br>

- 참조 : https://pytorch.org/docs/stable/index.html

<br>

- Pytorch로 Computer vision을 한다면 반드시 사용하는 `conv2d`에 대하여 정리하겠습니다.

<br>

```python
torch.nn.Conv2d(
    in_channels, 
    out_channels, 
    kernel_size, 
    stride=1, 
    padding=0, 
    dilation=1, 
    groups=1, 
    bias=True, 
    padding_mode='zeros'
)
```

<br>

- `conv2d`에서 사용되는 파라미터는 위와 같습니다. 여기서 입력되어야 하는 파라미터는 `in_channels`, `out_channels`, `kernel_size` 입니다.
- 나머지 파라미터는 기본값이 입력이 되어있고 기본값들이 일반적으로 많이 사용되는 값들입니다.
- 먼저 위 함수는 input 값에 2d convolution 연산을 적용하는 함수 입니다.
- input size는 $$ (N, C_{in}, H, W) $$ 이고 output은 $$ (N, C_{out}, H_{out}, W_{out}) $$ 입니다.
- 여기서 $$ N $$은 batch size이고  $$ C $$는 채널의 수를 나타냅니다. $$ H, W $$는 각각 height와 width를 나타냅니다.
- 이 기호들을 이용하여 식을 자세하게 나타내면 다음과 같습니다. 조금 어렵게 표현되어 있긴 하지만 자세히 살펴보면 어렵진 않습니다.

<br>

$$ out(N_{i}, C_{out}) = bias(C_{out_{j}}) + \sum_{k=0}^{C_{in}-1} weight(C_{out}, k) \otimes input(N_{i}, k) $$ 

<br>   

- 여기서 $$ \otimes $$는 2D convolution 연산을 뜻합니다.
- `stride`를 설정하기 위해 숫자 또는 튜플을 받습니다.
- `padding`은 zero-padding할 사이즈를 입력 받습니다.
- `dilation`은 atrous 알고리즘으로도 불리고 필터(커널) 사이의 간격을 의미합니다.
- `group`은 입력 채널과 출력 채널 사이의 관계를 나타내고 옵션은 다음과 같습니다. 
    - `group=1`이면 모든 입력은 모든 출력과 convolution 연산이 됩니다. 일반적으로 알려진 convolution 연산과 같습니다.
    - `groups=2`이면 입력을 2그룹으로 나누어서 각각 convolution 연산을 하고 그 결과를 concatenation을 합니다.
    - `groups=in_channels`이면 각각의 인풋 채널이 각각의 아웃풋 채널에 대응되어 convolution 연산을 하게 됩니다. 그 사이즈는 `out_channels // in_channels`가 됩니다.
- `kernel size, stride, padding, dilation`은 int 나 tuple이 될 수 있고 int이면 width와 height에 동시에 같은 값이 적용됩니다.
- 특히 `group=in_channels`이고 `out_channels == K * in_channels`이면 `depthwise convolution`이라고 합니다.
    - 이 개념은 `Xeption`이나 `mobilenet`에서 대표적으로 사용되고 있습니다.

<br>

- 아래는 예제입니다. 

<br>

```python
>>> # With square kernels and equal stride
>>> m = nn.Conv2d(16, 33, 3, stride=2)
>>> # non-square kernels and unequal stride and with padding
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
>>> # non-square kernels and unequal stride and with padding and dilation
>>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
>>> input = torch.randn(20, 16, 50, 100)
>>> output = m(input)
```

<br>

- 위에 내용만 숙지하고 특히 `dilation`과 `group`의 개념만 잘 이해하면 충분히 사용하는 데 문제는 없을 것입니다.

<br>

- 이번에는 MNIST 데이터 셋을 이용하여 Conv2d를 한번 다루어 보도록 하겠습니다.
- 먼저 아래 코드를 통해 간단하게 DataLoder를 하겠습니다.

<br>

```python
import torch
from torchvision import datasets, transforms

batch_size = 1
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root = "datasets/", # 현재 경로에 datasets/MNIST/ 를 생성 후 데이터를 저장한다.
        train = True, # train 용도의 data 셋을 저장한다.
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(), # tensor 타입으로 데이터 변경
            transforms.Normalize(mean = (0.5,), std = (0.5,)) # data를 normalize 하기 위한 mean과 std 입력
        ])
    ),
    batch_size=batch_size, 
    shuffle=True
)
```

<br>

- 위 MNIST 데이터 셋을 읽어 들여서 다음과 같이 실행하여 shape을 알아보겠습니다.

<br>

```python
image, label = next(iter(train_loader))
print(image.shape, label.shape)
# torch.Size([1, 1, 28, 28]) torch.Size([1])
```

<br>

- 네트워크를 구성할 때 기본적으로 다음 패키지는 import를 해야 합니다.

<br>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

<br>

- 앞에서 설명한 것 처럼 다음 3가지 필수 속성에 유념해서 `Conv2d`를 사용해 보겠습니다.

<br>

```python
nn.Conv2d(
    in_channels = 1,
    out_channels = 20,
    kernel_size = 5,
    stride = 1
)

# Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
```

<br>

- 그러면 위 Conv2d를 이용하여 layer를 만들어 보겠습니다. 이 때, device도 지정해 보겠습니다.

<br>

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3, stride = 1).to(device)
print(layer)
# Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1))
```

<br>

- 위 layer가 현재 가지고 있는 weight를 보려면 `.weight`를 이용하여 살펴볼 수 있습니다.

<br>

```python
print(layer.weight)

# Parameter containing:
# tensor([[[[ 0.1651,  0.2933, -0.1360],
#           [-0.0189,  0.0794,  0.0879],
#           [-0.1197,  0.2944,  0.2361]]],


#         [[[-0.2551, -0.0717,  0.0259],
#           [ 0.2137,  0.3300, -0.3100],
#           [-0.1889, -0.3203, -0.1923]]],


#         [[[-0.2630,  0.1230, -0.1499],
#           [ 0.0073,  0.2863,  0.1842],
#           [-0.0092,  0.1429,  0.1623]]]], requires_grad=True)

print(layer.weight.shape)
# torch.Size([3, 1, 3, 3])
```

<br>

- 위 layer의 weight가 가지는 shape을 살펴보면 (3, 1, 3, 3)임을 확인할 수 있습니다. 이것은 차례대로 batch_size, channels, height, width의 크기를 가집니다.

<br>

- 만약 위에서 다룬 `layer.weight`를 numpy로 변형하고 싶어서 `layer.weight.numpy()`라고 호출을 하면 다음과 같은 에러가 발생합니다.
- `Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.`
- pytorch에서 weight는 **학습 가능한 상태**이기 때문에 바로 numpy()를 이용하여 뽑아낼 수 없도록 하였습니다.
- 이 때, `detach().numpy()` 를 통하여 그래프에서 빼서 gradient의 영향을 받지 않도록 할 수 있습니다.
- 그러면 다음 코드를 통하여 weight의 크기를 컬러로 시각화 해보겠습니다. 논문에서 많이 사용하는 방법입니다.

<br>

```python
weight = layer.weight.detach().numpy()

import matplotlib.pyplot as plt
plt.imshow(weight[0, 0, :, :], 'jet')
plt.colorbar()
plt.show()
```

<br>
<center><img src="../assets/img/dl/pytorch/conv2d/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>




<br>

[Pytorch 관련 글 목록](https://gaussian37.github.io/dl-pytorch-table/)

<br>