---
layout: post
title: PyTorch 기본 문법 및 Tensor 사용법
date: 2019-05-18 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, pytorch 설치, colab] # add tag
---

+ 출처 : https://github.com/GunhoChoi/PyTorch-FastCampus
+ 이번 글에서는 Pytorch의 Tensor를 사용하는 간단한 방법에 대하여 알아보겠습니다.

<br> 

### Tensor 생성

+ 랜덤 넘버 생성

```python
import torch
x = torch.rand(2,3)
print(x)

 0.1330  0.3113  0.9652
 0.1237  0.4056  0.8464
[torch.FloatTensor of size 2x3]
```

<br>

+ 순열 생성

```python
torch.torch.randperm(5)

2
3
4
0
1
[torch.LongTensor of size 5]
```

<br>

+ zeros Tensor 생성

```python
torch.zeros(2,3)

: tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

<br>

+ ones Tensor 생성

```python
torch.ones(2,3)

: tensor([[1., 1., 1.],
        [1., 1., 1.]])

```

<br>

+ arange를 이용한 Tensor 생성
+ torch.arange(시작, 끝, step)을 인자로 사용하며 시작은 이상, 끝은 미만의 범위를 가지고 step만큼 간격을 두며 Tensor를 생성합니다. 

```python
# torch.arange(start,end,step=1) -> [start,end) with step
torch.arange(0,3,step=0.5)

: tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000])
```

<br>

### Tensor 데이터 타입

+ Float 타입의 m행 n열 Tensor 생성하기

```python
# 2행 3열의 Float 타입의 Tensor 생성
torch.cuda.FloatTensor(2,3)

: tensor([[2.0000e+00, 3.0000e+00, 1.4013e-45],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]], device='cuda:0')
```

<br>

+ 리스트를 입력하여 특정 리스트를 Tensor로 변환하기

```python
torch.cuda.FloatTensor([2,3])

: tensor([2., 3.], device='cuda:0')
```

<br>

+ Float 타입을 Int 타입으로 형변환

```python
x = torch.cuda.FloatTensor([2,3])
x.type_as(torch.cuda.IntTensor())

: tensor([2, 3], device='cuda:0', dtype=torch.int32)
```

<br>

### Numpy to Tensor 또는 Tensor to Numpy

+ Numpy를 생성한 후 Tensor로 변환한 후 다시 Numpy로 변환하기

```python
import numpy as np
x1 = np.ndarray(shape=(2,3), dtype=int,buffer=np.array([1,2,3,4,5,6]))
torch.from_numpy(x1)

: tensor([[1, 2, 3],
        [4, 5, 6]])
        
x2 = torch.from_numpy(x1)
x2.numpy()

: array([[1, 2, 3],
       [4, 5, 6]])
```
<br>

### CPU 타입과 GPU 타입의 Tensor

+ 딥러닝 프레임워크에서는 CPU와 GPU 두 타입에 대한 Tensor 생성이 가능합니다.
+ PyTorch에서는 어떻게 사용할 수 있는지 알아보겠습니다.

```python
x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_gpu = x.cuda()
x_gpu

: 1  2  3
 4  5  6
[torch.cuda.FloatTensor of size 2x3 (GPU 0)]

x_cpu = x_gpu.cpu()
x_cpu
 
: 1  2  3
 4  5  6
[torch.FloatTensor of size 2x3]
```

<br>

### Tensor 사이즈 확인하기

+ Tensor 사이즈를 확인하려면 `.size()`를 이용하여 확인하면 됩니다.

```python
x = torch.cuda.FloatTensor(10, 12, 3, 3)
x.size()

: torch.Size([10, 12, 3, 3])
```

<br>

### Index 기능 사용방법

+ Index는 Tensor에서 특정 값만 조회하는 것을 말합니다. 
+ 배열, 행렬에서도 인덱스 기능을 통하여 특정 값들을 조회하는 것 처럼 Tensor에서도 조회할 수 있습니다.
+ 먼저 `torch.index_select`함수를 이용해 보겠습니다. 파라미터는 input, dim, index가 차례로 입력됩니다.

```python
# torch.index_select(input, dim, index)
x = torch.rand(4,3)
print(x)

: tensor([[0.4898, 0.2505, 0.6500],
        [0.0976, 0.4117, 0.9705],
        [0.7069, 0.0546, 0.7824],
        [0.4921, 0.9863, 0.3936]])
    
# 3번째 인자에는 torch.LongTensor를 이용하여 인덱스를 입력해 줍니다.
torch.index_select(x,0,torch.LongTensor([0,2]))

: tensor([[0.4898, 0.2505, 0.6500],
        [0.7069, 0.0546, 0.7824]])
```  

<br>

+ 이번에는 좀더 파이썬스럽게 인덱싱을 해보겠습니다.
+ 아래 방법이 좀더 파이썬 유저에게 친숙한 방법입니다.

```python
print(x)

: tensor([[0.4898, 0.2505, 0.6500],
        [0.0976, 0.4117, 0.9705],
        [0.7069, 0.0546, 0.7824],
        [0.4921, 0.9863, 0.3936]])

print(x[:, 0])

: tensor([0.4898, 0.0976, 0.7069, 0.4921])

print(x[0, :])

: tensor([0.4898, 0.2505, 0.6500])

print(x[0:2, 0:2])

: tensor([[0.4898, 0.2505],
        [0.0976, 0.4117]])
```

<br>

+ 이번에는 `mask` 기능을 통하여 인덱싱 하는 방법에 대하여 알아보겠습니다.
+ `torch.masked_select(input, mask)` 함수를 이용하여 선택할 영역에는 `1`을 미선택할 영역은 `0`을 입력 합니다.
+ 인풋 영역과 마스크할 영역의 크기는 같아야 오류 없이 핸들링 할 수 있습니다.

```python
x = torch.randn(2,3)
print(x)

: tensor([[ 0.6122, -0.7963, -0.3964],
        [ 0.6030,  0.1522, -1.0622]])

# mask는 0,1 값을 가지고 ByteTensor를 이용하여 생성합니다.
mask = torch.ByteTensor([[0,0,1],[0,1,0]])
torch.masked_select(x,mask)

# (0,3)과 (1,1) 데이터 인덱싱
tensor([-0.3964,  0.1522])
```

<br>

### Join 기능 사용 방법

 
