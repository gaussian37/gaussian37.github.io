---
layout: post
title: PyTorch 기본 문법 및 Tensor 사용법
date: 2019-05-18 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, pytorch 설치, colab] # add tag
---

+ PyTorch 패키지의 구성 요소
    + `torch`
        + main namespace로 tensor등의 다양한 수학 함수가 패키지에 포함되어 있습니다.
        + NumPy와 같은 구조를 가지고 있습니다.
    + `torch.autograd`
        + 자동 미분을 위한 함수가 포함되어 있습니다. 
        + 자동 미분의 on, off를 제어하는 enable_grad 또는 no_grad나 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스인 Function등이 포함됩니다.
    + `torch.nn`
        + 신경망을 구축하기 위한 다양한 데이터 구조나 레이어가 정의되어 있습니다.
        + CNN, LSTM, 활성화 함수(ReLu), loss 등이 정의되어 있습니다.
    + `torch.optim`
        + SGD 등의 파라미터 최적화 알고리즘 등이 구현되어 있습니다.
    + `torch.utils.data`
        + Gradient Descent 계열의 반복 연산을 할 때, 사용하는 미니 배치용 유틸리티 함수가 포함되어 있습니다.
    + `torch.onnx`
        + ONNX(Open Neural Network Exchange) 포맷으로 모델을 export 할 때 사용합니다.
        + ONNX는 서로 다른 딥러닝 프레임워크 간에 모델을 공유할 때 사용하는 새로운 포맷입니다.
        

    
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

+ PyTorch에서 `torch.cat(seq, dim)`을 이용하여 concaternate 연산을 할 수 있습니다.
+ `dim`은 concaternate할 방향을 정합니다.

```python
x = torch.cuda.FloatTensor([[1, 2, 3], [4, 5, 6]])
y = torch.cuda.FloatTensor([[-1, -2, -3], [-4, -5, -6]])
z1 = torch.cat([x, y], dim=0)
print(z1)

tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [-1., -2., -3.],
        [-4., -5., -6.]], device='cuda:0')

z2 = torch.cat([x, y], dim=1)
print(z2)

tensor([[ 1.,  2.,  3., -1., -2., -3.],
        [ 4.,  5.,  6., -4., -5., -6.]], device='cuda:0')
```

<br>

+ `torch.stack` 을 이용하여도 concaternate를 할 수 있습니다.

```python
# torch.stack(sequence,dim=0) -> stack along new dim

x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_stack = torch.stack([x,x,x,x],dim=0)

x_stack

: tensor([[[1., 2., 3.],
         [4., 5., 6.]],

        [[1., 2., 3.],
         [4., 5., 6.]],

        [[1., 2., 3.],
         [4., 5., 6.]],

        [[1., 2., 3.],
         [4., 5., 6.]]])
```

<br>

### slicing 기능 사용 방법

+ slicing 기능은 Tensor를 몇개의 부분으로 나뉘는 기능입니다.
+ `torch.chunk(tensor, chunks, dim=0)` 또는 `torch.split(tensor,split_size,dim=0)`함수를 이용하여 Tensor를 나뉠 수 잇습니다

```python
# torch.chunk(tensor, chunks, dim=0) -> tensor into num chunks

x_1, x_2 = torch.chunk(z1,2,dim=0)
y_1, y_2, y_3 = torch.chunk(z1,3,dim=1)

print(z1)
:  1  2  3
  4  5  6
 -1 -2 -3
 -4 -5 -6
 [torch.FloatTensor of size 4x3]
 
 print(x1)
 :  1  2  3
  4  5  6
 [torch.FloatTensor of size 2x3], 
 
 print(x2)
 : -1 -2 -3
 -4 -5 -6
 [torch.FloatTensor of size 2x3]
 
 print(y1)
 :  1
  4
 -1
 -4
 [torch.FloatTensor of size 4x1], 
 
 print(y2)
 : 2
  5
 -2
 -5
 [torch.FloatTensor of size 4x1]
 
 print(y3)
:  3
  6
 -3
 -6
 [torch.FloatTensor of size 4x1]
```

<br>

### squeezing 기능 사용 방법

+ squeeze 함수를 사용하면 dimension 중에 1로 되어 있는 것을 압축할 수 있습니다.
+ dimension이 1이면 사실 불필요한 차원일 수 있기 때문에 squeeze를 이용하여 압축 시키는 것이 때론 필요할 수 있는데, 그 때 사용하는 함수 입니다.
+ `torch.squeeze(input, dim)`으로 사용할 수 있고, dim을 지정하지 않으면 dimeion이 1인 모든 차원을 압축하고 dim을 지정하면 지정한 dimension만 압축합니다.

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
torch.Size([2, 2, 1, 2])
```

<br>

+ 반면 unsqueeze 함수를 사용하면 dimension을 추가할 수 있습니다. squeeze와 정확히 반대라고 보시면 됩니다.
+ unsqueeze 함수는 dimension을 반드시 입력 받게 되어 있습니다.

```python
>>> x = torch.zeros(2,3,4)
>>> torch.unsqueeze(x, 0)
torch.Size([1, 2, 3, 4])

>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```

<br>

### Initialization, 초기화 방법

+ `init.uniform`함수를 사용하면 `uniform` 또는 `normal` 분포의 초기화 Tensor를 만들 수 있습니다.
+ 또는 상수 형태를 바로 만들 수도 있습니다. 예제는 아래와 같습니다.

```python
import torch.nn.init as init

x1 = init.uniform(torch.FloatTensor(3,4),a=0,b=9) 
>>> print(x1)

3.4121  0.9464  5.3126  4.3104
  7.3428  8.3997  7.2747  7.9227
  4.2563  0.1993  6.2227  5.7939
 [torch.FloatTensor of size 3x4]
 
x2 = init.normal(torch.FloatTensor(3,4),std=0.2)
>>> print(x2)

-0.1121  0.3684  0.0316  0.1426
  0.1499 -0.2384  0.0183  0.0429
  0.2808  0.1389  0.1057  0.1746
 [torch.FloatTensor of size 3x4]
 
x3 = init.constant(torch.FloatTensor(3,4),3.1415)
>>> print(x3)

3.1415  3.1415  3.1415  3.1415
  3.1415  3.1415  3.1415  3.1415
  3.1415  3.1415  3.1415  3.1415
 [torch.FloatTensor of size 3x4]
```

<br>

### Math Operation

+ Tensor의 산술 연산 방법에 대하여 알아보겠습니다.
+ `+`연산자 또는 `torch.add()`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
add = x1 + x2
# add = torch.add(x1,x2) 또한 가능합니다.
>>> print(add)
 2   4   6
 8  10  12
 [torch.FloatTensor of size 2x3]

```

<br>

+ `+`연산자를 이용한 broadcasting 또는 `torch.add() with broadcasting`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1 + 10 # torch.add(x1, 10) 또한 가능합니다.
11  12  13
14  15  16
[torch.FloatTensor of size 2x3]
```

<br>

+ `*`연산자 또는 `torch.mul()`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1*x2 # torch.mul(x1,x2)
1   4   9
16  25  36
[torch.FloatTensor of size 2x3]
```
<br>

+ `*`연산자를 이용한 broadcasting 또는 `torch.mul() with broadcasting`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1 * 10
10  20  30
40  50  60
[torch.FloatTensor of size 2x3]
```

<br>

+ `/`연산자 또는 `torch.div()`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1/x2 # torch.div(x1, x2)
 1  1  1
 1  1  1
[torch.FloatTensor of size 2x3]

```

<br>

+ `/`연산자를 이용한 broadcsting 또는 `torch.div() with broadcasting`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1 / 5
 0.2000  0.4000  0.6000
 0.8000  1.0000  1.2000
[torch.FloatTensor of size 2x3]
```

<br>

+ power 연산 : `torch.pow(input,exponent)`

```python
x1 = torch.FloatTensor([ [1,2,3], [4,5,6] ])
>>> x1**2 # torch.pow(x1,2)
tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]])
```
<br>

+ exponential 연산 : `torch.exp(tensor,out=None)`

```python
x1 = torch.FloatTensor([ [1,2,3], [4,5,6] ])
>>> torch.exp(x1)
tensor([[  2.7183,   7.3891,  20.0855],
        [ 54.5981, 148.4132, 403.4288]])
```

<br>

+ `torch.log(input, out=None) -> natural logarithm`

```python
x1 = torch.FloatTensor([ [1,2,3], [4,5,6] ])
>>> torch.log(x1)
tensor([[0.0000, 0.6931, 1.0986],
        [1.3863, 1.6094, 1.7918]])
```

<br>

### Matrix Multiplication

+ `torch.mm(mat1, mat2) -> matrix multiplication`
+ Tensor(행렬)의 곱을 연산하므로 shape이 잘 맞아야 연산이 가능합니다.

```python
x1 = torch.FloatTensor(3,4)
x2 = torch.FloatTensor(4,5)
torch.mm(x1,x2).size()
```

<br>

+ `torch.bmm(batch1, batch2) -> batch matrix multiplication`
+ Tensor(행렬)의 곱을 batch 단위로 처리합니다. `torch.mm`에서는 단일 Tensor(행렬)로 계산을 한 반면에 batch 단위로 한번에 처리하므로 좀 더 효율적입니다.

```python
x1 = torch.FloatTensor(10,3,4)
x2 = torch.FloatTensor(10,4,5)
torch.bmm(x1,x2).size()
```

<br>

+ `torch.dot(tensor1,tensor2)`는 두 tensor의 dot product 연산을 수행합니다.

```python
>>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)
```

<br>

+ `torch.t()`를 이용하면 transposed tensor를 구할 수 있습니다.

```python
x1 = torch.FloatTensor(2,3)
x2 = x1.t()
>>> print(x1.size())
torch.Size([2, 3])

>>> print(x2.size())
torch.Size([3, 2])
```

<br>

+ 반면 `torch.transpose()`를 이용하여 특정 dimension을 변경할 수 있습니다.

```python
x1 = torch.FloatTensor(10,3,4,5)
>>> x1.size()
torch.Size([10, 3, 4, 5])

>>> torch.transpose(x1,1,2).size()
torch.Size([10, 4, 3, 5])

>>> torch.transpose(x1,2,3).size()
torch.Size([10, 3, 5, 4])
```

<br>

+ `eigenvalue`와 `eigenvector`를 구하는 방법은 아래와 같습니다.
+ 출력은 각각 `eigenvalue`와 `eigenvector`입니다.

```python
x1 = torch.FloatTensor(4,4)
>>> torch.eig(x1,eigenvectors=True)

1.00000e-12 *
   -0.0000  0.0000
    0.0000  0.0000
    4.7685  0.0000
    0.0000  0.0000
  [torch.FloatTensor of size 4x2], 
  
  -6.7660e-13  6.6392e-13 -3.0669e-15  1.7105e-20
   7.0711e-01  7.0711e-01  1.0000e+00 -1.0000e+00
   2.1701e-39 -2.1294e-39 -2.8813e-10  1.8367e-35
   7.0711e-01  7.0711e-01  7.3000e-07  3.2207e-10
  [torch.FloatTensor of size 4x4]))
```
