---
layout: post
title: Neural Network with PyTorch
date: 2019-05-19 01:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, Neural Network] # add tag
---

+ 출처 : https://github.com/GunhoChoi/PyTorch-FastCampus
+ 이번 글에서는 Pytorch의 Tensor를 사용하는 간단한 방법에 대하여 알아보겠습니다.

+ 간단한 MLP를 작성하여 PyTorch로 Neural Network를 만들어 보겠습니다.
+ 아래 코드를 보면 `nn.Sequential`은 `nn.Module` 층을 차례로 쌓아서 신경망을 구축할 때 사용합니다. 

```python
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
)
``` 

<br>