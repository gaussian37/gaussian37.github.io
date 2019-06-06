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

+ MNIST 데이터를 한번 학습시켜 보겠습니다.

```python
import torch
from torch import nn, optim
from sklearn.datasets import load_digits
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
Y = digits.target

# Numpy의 ndarray를 파이토치 텐서로 변환
X = torch.tensor(X, dtype=torch.float32).to("cuda:0")
Y = torch.tensor(Y, dtype=torch.int64).to("cuda:0")

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
)

net.to("cuda:0")

summary(net.cuda(), input_size=(64,))

# Loss function을 정의 : cross entropy
lossFunc = nn.CrossEntropyLoss()

# Optimizer 정의
optimizer = optim.Adam(net.parameters())

# loss function의 log
losses = []

for epoch in tqdm(range(100)):
    # backward 메서드로 계산된 이전 값을 삭제
    optimizer.zero_grad()
    
    # 구축한 네트워크로 y의 예측값을 계산
    yPred = net(X)
    
    # loss와 w를 사용한 미분 계산
    loss = lossFunc(yPred, Y)
    loss.backward()
    
    # 경사를 갱신
    optimizer.step()
    
    # 수렴 확인을 위해 loss를 기록합니다.
    losses.append(loss.item())

plt.plot(losses)

```