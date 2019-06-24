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

<br>


### Neural network 예제

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn import datasets

# 생성할 데이터의 갯수
n_pts = 500
# 데이터 생성
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
x_data = torch.Tensor(X)
y_data = torch.Tensor(y)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    
    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0

torch.manual_seed(2)
model = Model(2, 4, 1)
# 학습하기 전의 weight와 bias를 출력합니다.
print(list(model.parameters()))

lossFunc = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
losses = []
for i in range(epochs):
    yPred = model.forward(x_data)
    loss = lossFunc(yPred, y_data)
    if i % 100 == 0:
        print("epoch : ", i, "loss", loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("loss")
plt.xlabel("epoch")

x1 = 0.025
y1 = 0.025
x2 = -1
y2 = 0
point1 = torch.Tensor([x1, y1])
point2 = torch.Tensor([x2,y2])
prediction1 = model.predict(point1)
prediction2 = model.predict(point2)
print(prediction1, prediction2)
```