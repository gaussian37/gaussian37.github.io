---
layout: post
title: Dataset과 Dataloader 및 Sampler
date: 2019-06-06 01:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, Dataset, Dataloader, DataSet, Sampler] # add tag
---

<br>

## **목차**

<br>

- ### [Dataset과 Dataloader](#dataset과-dataloader-1)
- ### [Custom Dataset for Image](#custom-dataset-for-image-1)

- 이번 글에서는 Dataset과 Dataloader에 대하여 알아보도록 하겠습니다.
- 간단한 예제에서는 모든 데이터를 모아서 학습용으로 사용했지만, 데이터가 늘어나거나 신경망 계층의 증가 또는 파라미터가 늘어나면서 전체 데이터를 메모리에서 처리하기가 어려워집니다.
- 이 문제를 해결하기 위하여 데이터의 일부 미니배치만 사용하는 방법에 대하여 알아보겠습니다.

<br>

## **Dataset과 Dataloader**

<br>

- 파이토치에는 Dataset과 `DataLoader`라는 기능이 있어서 미니 배치 학습이나 데이터 셔플, 병렬 처리등을 간단하게 할 수 있습니다.
- TensorDataset은 Dataset을 상속한 클래스로 학습 데이터 X와 레이블 Y를 묶어 놓은 컨테이너 입니다. TensorDataset을 DataLoader에 전달하면 for 루프에서 데이터의 일부만 간단하게 추출할 수 있습니다. TensorDataset에는 텐서만 전달할 수 있으며 Variable은 전달할 수 없습니다.
- 아래 코드에서 TensorDataset을 `DataLoader`에 전달해서 데이터의 일부만 추출하는 예제입니다.
- `DataLoader`의 정의는 다음과 같습니다.

<br>

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```

<br>

- 아래 코드에서 핵심은 `loader = DataLoader(ds, batch_size=64, shuffle=True)` 입니다.


<br>

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

- Dataset은 직접 작성할 수도 있어서 대량의 이미지 파일을 한 번에 메모리에 저장하지 않고, 필요할 때마다 읽어서 학습하는듯 다양하게 활용할 수 있습니다.
- `loader = DataLoader(ds, batch_size=64, shuffle=True)` 코드를 살펴보면 `ds`라는 데이터셋을 batch_size = 64, shuffle 방식으로 사용하도록 설정되어 있습니다.
- `loader`는 학습을 진행하는 for문에 사용되었으며 generator 사용 되어 각 iteration 마다 batch size 만큼 가져와서 사용하는 것을 알 수 있습니다. for 문의 `batchX`, `batchY`가 각각 학습 데이터와 라벨로 사용되는 데이터입니다.

<br>

## **Custom Dataset for Image**

<br>

- `DataLoader`의 필수 입력 인자는 `DataSet` 입니다. 위 예제에서는 TensorDataset라는 pytorch에서 제공하는 클래스를 사용하였습니다.
- `DataSet`은 DataLoader를 통하여 data를 받아오는 역할을 합니다.
- 만약 사용자가 원하는 방식의 `custom dataset`이 필요하다면 어떻게 만들 수 있을까요?
- 먼저 `DataSet`에 반드시 필요한 2가지 요소를 이해해야 합니다. ① `__len__`, ② `__getitem__` 입니다.

<br>

```python
import torch
from torch.utils.data import Dataset

class TestDataSet(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        return {"input":torch.tensor([index, 2*index, 3*index], dtype=torch.float32), 
                "label": torch.tensor(index, dtype=torch.float32)}

test_dataset = TestDataSet()

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
for data in dataloader:
    print(data)

# {'input': tensor([[0., 0., 0.],
#         [1., 2., 3.],
#         [2., 4., 6.],
#         [3., 6., 9.]]), 'label': tensor([0., 1., 2., 3.])}
# {'input': tensor([[ 4.,  8., 12.],
#         [ 5., 10., 15.],
#         [ 6., 12., 18.],
#         [ 7., 14., 21.]]), 'label': tensor([4., 5., 6., 7.])}
# {'input': tensor([[ 8., 16., 24.],
#         [ 9., 18., 27.]]), 'label': tensor([8., 9.])}
```

<br>

- 위 예제에서 살펴볼 수 있듯이 `__len__`은 총 반환해야 할 데이터의 갯수가 정의되며 `__getitem__`은 반환해야 할 데이터 타입이 정의됩니다.
- `__getitem__`의 핵심은 `index` 입니다. 즉, index 순서대로 DataLoader를 통해 data가 반환되는 것을 알 수 있습니다.

<br>

- `DataLoader`에서 사용되는 또 다른 중요한 인자로 `Sampler`가 있습니다.
- dataset은 사용된 index에 맞는 data를 가져오도록 되어있습니다. 따라서 데이터를 섞어서 가져올 때, index만 적절히 섞어주어도 됩니다. 이 때 사용되는 것인 Sampler입니다.
- Sampler를 생성할 때에는 `RandomSampler`를 사용하면 되며 dataloader의 sampler 인자에 아래 코드와 같이 넣어주면 됩니다.

<br>

```python
import torch
from torch.utils.data import Dataset, RandomSampler

for data in dataloader:
    print(data['input'].shape, data['label'])

class TestDataSet(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        return {"input":torch.tensor([idx, 2*idx, 3*idx], dtype=torch.float32), 
                "label": torch.tensor(idx, dtype=torch.float32)}

test_dataset = TestDataSet()
sampler = RandomSampler(test_dataset)
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, sampler=sampler)

for data in dataloader:
    print(data)

# torch.Size([4, 3]) tensor([0., 3., 1., 2.])
# torch.Size([1, 3]) tensor([4.])
# {'input': tensor([[ 2.,  4.,  6.],
#         [ 0.,  0.,  0.],
#         [ 1.,  2.,  3.],
#         [ 7., 14., 21.]]), 'label': tensor([2., 0., 1., 7.])}
# {'input': tensor([[ 6., 12., 18.],
#         [ 8., 16., 24.],
#         [ 3.,  6.,  9.],
#         [ 4.,  8., 12.]]), 'label': tensor([6., 8., 3., 4.])}
# {'input': tensor([[ 5., 10., 15.],
#         [ 9., 18., 27.]]), 'label': tensor([5., 9.])}
```