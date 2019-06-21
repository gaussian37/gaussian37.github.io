---
layout: post
title: Image Recognition with Pytorch
date: 2019-05-19 01:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, Neural Network] # add tag
---

- 이번 글에서는 Image Recognition을 위한 기본 내용 부터 필요한 내용까지 전체를 다루어 볼 예정입니다.
    - `MNIST` 데이터셋 이미지 인식을 먼저 실습해 보겠습니다.
    - 좀 더 이미지를 잘 분석하기 위하여 Convolutional Neural Network 구조를 사용해 보겠습니다.
    - 그 다음으로는 좀 더 다양한 이미지를 분석 하기 위하여 `CIFAR10` 이미지를 분석해 보겠습니다.
    - 마지막으로 `Transfer learning`을 이용하여 성능을 더 향상시켜보도록 하겠습니다. 
    
<br>

### Fashion MNIST 데이터를 Naive 하게 학습하기

- 먼저 다음과 같이 필요한 라이브러리를 import합니다.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
```

<br>

- 다음으로 데이터 셋을 받기 위한 설정을 한 뒤 데이터를 로컬에 받습니다.

```python
# 데이터 셋을 받을 때 transform할 조건을 설정합니다.
mean = 0.5
sigma = 0.5
transform = transforms.Compose([transforms.ToTensor(), # 데이터를 불러 올 때 Tensor 타입으로 불러옵니다.  
                                 transforms.Normalize([mean], [sigma]) # 앞의 괄호는 평균, 뒤의 괄호는 표준편차이며 세 차원 모두 적용
                                ])
# root경로에 train 데이터를 다운 받습니다. 이 때 조건은 transform 조건에 따라 Tensor 타입과 Normalize가 된 상태로 받습니다.
training_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

# Dataloader를 정의합니다. Dataloader에서는 batch size와 shuffle을 세팅해줍니다.
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
```

<br>

- 위 옵션으로 데이터를 받으면 학습에 편리하도록 값을 변형해서 받는 것이지 시각화 하기에는 적합하지 않습니다.
- 아래 코드는 시각화 하기 좋도록 image를 복사한 뒤 차원을 (H, W, C)순서로 변형하고 Normalize 하기 전 값으로 복원합니다.
- 시각화 할 때 사용하면 되고 학습에는 전혀 상관없으므로 무시하셔도 되는 코드입니다.

```python
# Tensor 형태의 이미지 데이터를 원래 데이터로 바꿉니다.
def imgConvert(tensor, mean, sigma):
    # tensor를 복사하고 그래프와 독립적으로 분리한 다음 numpy로 변형합니다.
    image = tensor.clone().detach().numpy()
    # Pytorch에서 제공하는 (H, W, C) 순서의 차원을 (H, W, C)로 변경합니다.
    image = image.transpose(1, 2, 0)
    # Normalize한 이미지 데이터를 원상 복귀 합니다.
    image = image * sigma + mean
    # 이미지의 범위를 0과 1 사이로 고정시킵니다.
    image = image.clip(0, 1)
    return image  
```

<br>

- 앞의 코드를 보면 `training_loader`를 선언할 때 batch_size = 100 으로 설정하였습니다.
- 다음 코드를 보면 batch의 크기만큼 generator가 이미지를 불러오는 것을 확인할 수 있습니다.

```python
dataiter = iter(training_loader)
images, labels = dataiter.next()
>> print(images.shape)

torch.Size([100, 1, 28, 28]) # (배치사이즈, 채널수, 높이, 너비)
```

