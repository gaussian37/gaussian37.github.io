---
layout: post
title: Pytorch - Conv2d
date: 2019-09-27 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, conv2d] # add tag
---

- 출처: https://pytorch.org/docs/stable/index.html

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