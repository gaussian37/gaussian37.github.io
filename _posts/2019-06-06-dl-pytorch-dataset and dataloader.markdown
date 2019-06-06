---
layout: post
title: Dataset과 Dataloader
date: 2019-06-06 01:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, Dataset, Dataloader] # add tag
---

+ 이번 글에서는 Dataset과 Dataloader에 대하여 알아보도록 하겠습니다.
+ 간단한 예제에서는 모든 데이터를 모아서 학습용으로 사용했지만, 데이터가 늘어나거나 신경망 계층의 증가 또는 파라미터가 늘어나면서 전체 데이터를 메모리에서 처리하기가 어려워집니다.
+ 이 문제를 해결하기 위하여 데이터의 일부 미니배치만 사용하는 방법에 대하여 알아보겠습니다.

### Dataset과 Dataloader

+ 파이토치에는 Dataset과 DataLoader라는 기능이 있어서 미니 배치 학습이나 데이터 셔플, 병렬 처리등을 간단하게 할 수 있습니다.
+ TensorDataset은 Dataset을 상속한 클래스로 학습 데이터 X와 레이블 Y를 묶어 놓은 컨테이너 입니다.
+ TensorDataset을 DataLoader에 전달하면 for 루프에서 데이터의 일부만 간단하게 추출할 수 있습니다.
+ TensorDataset에는 텐서만 전달할 수 있으며 Variable은 전달할 수 없습니다.
+ 아래 코드에서 TensorDataset을 DataLoader에 전달해서 데이터의 일부만 추출하는 예제입니다.

```python
import torch
from torch import nn, optim
from sklearn.datasets import load_digits
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
Y = digits.target

# Numpy의 ndarray를 파이토치 텐서로 변환
X = torch.tensor(X, dtype=torch.float32).to("cuda:0")
Y = torch.tensor(Y, dtype=torch.int64).to("cuda:0")

# Dataset 작성
ds = TensorDataset(X, Y)

# 데이터 순서를 섞어서 64개씩 데이터를 반환하는 DataLoader
loader = DataLoader(ds, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

losses = []
for epoch in range(10):
    runningLoss = 0.0
    for batchX, batchY in loader:
        optimizer.zero_grad()
        # batchX, batchY는 64개씩 받습니다.
        yPred = model(batchX)
        loss = lossFunc(yPred, batchY)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
    losses.append(runningLoss)
```

<br>

+ Dataset은 직접 작성할 수도 있어서 대량의 이미지 파일을 한 번에 메모리에 저장하지 않고, 필요할 때마다 읽어서 학습하는듯 다양하게 활용할 수 있습니다.



