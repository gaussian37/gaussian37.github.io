---
layout: post
title: PyTorch 기본 문법 및 Tensor 사용법
date: 2019-05-18 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, pytorch 설치, colab] # add tag
---

<br>

## **목차**

<br>

- ### [Pytorch 란?](#pytorch-란-1)
- ### [PyTorch 패키지의 구성 요소](#pytorch-패키지의-구성-요소-1)
- ### [PyTorch 기초 사용법](#pytorch-기초-사용법-1)
- ### [텐서의 생성과 변환](#텐서의-생성과-변환-1)
- ### [텐서의 인덱스 조작](#텐서의-인덱스-조작-1)
- ### [텐서 연산](#텐서-연산-1)
- ### [텐서의 차원 조작](#텐서의-차원-조작-1)
- ### [Tensor 생성](#tensor-생성-1)
- ### [Tensor 데이터 타입](#tensor-데이터-타입-1)
- ### [Numpy to Tensor 또는 Tensor to Numpy](#numpy-to-tensor-또는-tensor-to-numpy-1)
- ### [CPU 타입과 GPU 타입의 Tensor](#cpu-타입과-gpu-타입의-tensor-1)
- ### [Tensor 사이즈 확인하기](#tensor-사이즈-확인하기-1)
- ### [Index (slicing) 기능 사용방법](#index-slicing-기능-사용방법-1)
- ### [Join(cat, stack) 기능 사용 방법](#joincat-stack-기능-사용-방법-1)
- ### [slicing 기능 사용 방법](#slicing-기능-사용-방법-1)
- ### [squeezing 기능 사용 방법](#squeezing-기능-사용-방법-1)
- ### [Initialization, 초기화 방법](#initialization-초기화-방법-1)
- ### [Math Operation](#math-operation-1)
- ### [Gradient를 구하는 방법](#gradient를-구하는-방법-1)
- ### [벡터와 텐서의 element-wise multiplication](#)

<br>

## **Pytorch 란?**

<br>

- 출처 : http://www.itworld.co.kr/news/129659

<br>
<center><img src="../assets/img/dl/pytorch/basic/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>


<br>

## **PyTorch 패키지의 구성 요소**

<br>

- `torch`
    - main namespace로 tensor등의 다양한 수학 함수가 패키지에 포함되어 있습니다.
    - **NumPy와 같은 구조**를 가지고 있어서 numpy와 상당히 비슷한 문법 구조를 가지고 있습니다.
- `torch.autograd`
    - 자동 미분을 위한 함수가 포함되어 있습니다. 
    - 자동 미분의 on, off를 제어하는 enable_grad 또는 no_grad나 자체 미분 가능 함수를 정의할 때 사용하는 기반 클래스인 Function등이 포함됩니다.
- `torch.nn`
    - 신경망을 구축하기 위한 다양한 데이터 구조나 레이어가 정의되어 있습니다.
    - CNN, LSTM, 활성화 함수(ReLu), loss 등이 정의되어 있습니다.
- `torch.optim`
    - SGD 등의 파라미터 최적화 알고리즘 등이 구현되어 있습니다.
- `torch.utils.data`
    - Gradient Descent 계열의 반복 연산을 할 때, 사용하는 미니 배치용 유틸리티 함수가 포함되어 있습니다.
- `torch.onnx`
    - ONNX(Open Neural Network eXchange) 포맷으로 모델을 export 할 때 사용합니다.
    - ONNX는 서로 다른 딥러닝 프레임워크 간에 모델을 공유할 때 사용하는 새로운 포맷입니다.

<br>

## **PyTorch 기초 사용법**

<br>

```python
nums = torch.arange(9)
# : tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
```

<br>

- 먼저 위와 같이 pytorch의 torch라는 것을 통하여 생성할 수 있습니다. 여기서 보면 앞에서 언급한 바와 같이 numpy와 상당히 비슷한 것을 느낄 수 있습니다.

<br>

```python
nums.shape
# torch.Size([9])
```

<br>

```python
type(nums)
# torch.Tensor
```

<br>

- 여기 까지 보면 뭔가 상당히 numpy 스러운 문법 구조를 가지고 있음을 아실 수 있을 것입니다.

<br>

```python
# tensor를 numpy로 타입 변환
nums.numpy()
# array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype = int64)

nums.reshape(3, 3)
# tensor([ [0, 1, 2],
#           [3, 4, 5],
#           [6, 7, 8] ]])

nums = torch.arange(9).reshape(3, 3)
nums
# tensor([[0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]])
nums + nums
# tensor([[ 0,  2,  4],
#         [ 6,  8, 10],
#         [12, 14, 16]])
```

<br>

- 그 다음으로는 `Tensor`를 직접 생성하고 조작하는 방법들에 대하여 다루어 보도록 하겠습니다.

<br>

## **텐서의 생성과 변환**

<br>

- 텐서는 파이토치의 가장 기본이 되는 데이터 구조와 기능을 제공하는 다차원 배열을 처리하기 위한 데이터 구조입니다.
- API 형태는 `Numpy`의 ndarray와 비슷하며 GPU를 사용하는 계산도 지원합니다.
- 텐서는 각 데이터 형태별로 정의되어 있습니다.
    - `torch.FloatTensor` : 32bit float point
    - `torch.LongTensor` : 64bit signed integer
- GPU 상에서 계산할 때에는 torch.cuda.FloatTensor를 사용합니다. 일반적으로 Tensor는 FloatTensor라고 생각하면 됩니다.
- 어떤 형태의 텐서이건 `torch.tensor`라는 함수로 작성할 수 있습니다.

```python
import torch
import numpy as np

# 2차원 형태릐 list를 이용하여 텐서를 생성할 수 있습니다.
torch.tensor([[1,2],[3,4.]])
# : tensor([[1., 2.],
#         [3., 4.]])
        
# device를 지정하면 GPU에 텐서를 만들 수 있습니다.
torch.tensor([[1,2],[3,4.]], device="cuda:0")
# : tensor([[1., 2.],
#         [3., 4.]], device='cuda:0')
        
# dtype을 이용하여 텐서의 데이터 형태를 지정할 수도 있습니다.
torch.tensor([[1,2],[3,4.]], dtype=torch.float64)
# : tensor([[1., 2.],
#         [3., 4.]], dtype=torch.float64)
        
# arange를 이용한 1차원 텐서
torch.arange(0, 10)
# : tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 모든 값이 0인 3 x 5의 텐서를 작성하여 to 메소드로 GPU에 전송
torch.zeros(3, 5).to("cuda:0")
# :tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]], device='cuda:0')
        
# normal distribution으로 3 x 5 텐서를 작성
torch.randn(3, 5)
# : tensor([[-0.4615, -0.4247,  0.1998, -0.5937, -0.4767],
#         [ 0.7864,  0.3831, -0.7198, -0.0181, -1.1796],
#         [-0.4504, -1.3181,  0.2657,  0.6829, -1.1690]])
        
# 텐서의 shape은 size 메서드로 확인
t = torch.randn(3, 5)
t.size()
# : torch.Size([3, 5])
```

<br>

- 텐서는 Numpy의 ndarray로 쉽게 변환할 수 있습니다.
- 단, GPU상의 텐서는 그대로 변환할 수 없으며, CPU로 이동 후에 변환해야 합니다.

```python
# numpy를 사용하여 ndarray로 변환
t = torch.tensor([[1,2],[3,4.]])
x = t.numpy()

# GPU 상의 텐서는 to 메서드로 CPU의 텐서로 변환 후 ndarray로 변환해야 합니다.
t = torch.tensor([[1,2],[3,4.]], device="cuda:0")
x = t.to("cpu").numpy()
```

<br>

- `torch.linspace(시작, 끝, step)`
    - 시작과 끝을 포함하고 step의 갯수만큼 원소를 가진 등차 수열을 만듭니다.
    - 예를 들어 `torch.linspace(0, 10, 5)` 라고 하면 
    - `tensor([0.0, 2.5, 5.0, 7.5, 10.0])`의 값을 가집니다.
- torch에서 바로 이런 값들을 만들면 torch 내부적으로도 사용할 수 있지만 numpy와 호환되는 라이브러리에도 사용 가능합니다.
    - 왜냐하면 torch를 numpy로 바꿀 수 있기 때문입니다.

```python
x = torch.linspace(0, 10, 5)
y = torch.exp(x)

plt.plot(x.numpy(), y.numpy())
```  

<br>

## **텐서의 인덱스 조작**

<br>

- 텐서의 인덱스를 조작하는 방법은 여러가지가 있습니다.
- 텐서는 Numpy의 ndarray와 같이 조작하는 것이 가능합니다. 배열처럼 인덱스를 바로지정 가능하고 슬라이스, 마스크 배열을 사용할 수 있습니다.

```python
t = torch.tensor([
    [1,2,3],[4,5,6.]
])

# 인덱스 접근
t[0, 2]
: tensor(3.)

# 슬라이스로 접근
t[:, :2]
: tensor([[1., 2.],
        [4., 5.]])
        
# 마스크 배열을 이용하여 True 값만 추출
t[t > 3]
: tensor([4., 5., 6.])

# 슬라이스를 이용하여 일괄 대입
t[:, 1] = 10

# 마스크 배열을 사용하여 일괄 대입
t[t > 5] = 20
```

<br>

## **텐서 연산**

<br>

- 텐서는 Numpy의 ndarray와 같이 다양한 수학 연산이 가능하며 GPU를 사용할 시에는 더 빠른 연산이 가능합니다.
- 텐서에서의 사칙연산은 같은 타입의 텐서 간 또는 텐서와 파이썬의 스칼라 값 사이에서만 가능합니다.
    - 텐서간이라도 타입이 다르면 연산이 되지 않습니다. FloatTensor와 DoubleTensor간의 사칙연산은 오류가 발생합니다.
- 스칼라 값을 연산할 때에는 기본적으로 `broadcasting`이 지원됩니다.

```python
# 길이 3인 벡터
v = torch.tensor([1,2,3.])
w = torch.tensor([0, 10, 20])

# 2 x 3의 행렬
m = torch.tensor([
    [0, 1, 2], [100, 200, 300.]
])

# 벡터와 스칼라의 덧셈
v2 = v + 10

# 제곱
v2 = v ** 2

# 동일 길이의 벡터간 덧셈 연산
z =  v - w

# 여러 가지 조합
u = 2 * v - w / 10 + 6.0

# 행렬과 스칼라 곱
m2 = m * 2.0

# (2, 3)인 행렬과 (3,)인 벡터간의 덧셈이므로 브로드캐스팅 발생
m + v

:tensor([[  1.,   3.,   5.],
        [101., 202., 303.]])
        
# 행렬 간 처리
m + m

:tensor([[  0.,   2.,   4.],
        [200., 400., 600.]])
```

<br>

## **텐서의 차원 조작**

<br>

- 텐서의 차원을 변경하는 `view`나 텐서를 결합하는 `stack`, `cat`, 차원을 교환하는 `t`, `transpose`도 사용됩니다.
- `view`는 numpy의 reshape와 유사합니다. 물론 pytorch에도 `reshape` 기능이 있으므로 `view`를 사용하던지 `reshape`을 사용하던지 사용방법은 같으므로 선택해서 사용하면 됩니다. (**reshape를 사용하길 권장합니다.**)
- `cat`은 다른 길이의 텐서를 하나로 묶을 때 사용합니다. 
- `transpose`는 행렬의 전치 외에도 차원의 순서를 변경할 때에도 사용됩니다.

```python
x1 = torch.tensor([
    [1, 2], [3, 4.]
])

x2 = torch.tensor([
    [10, 20, 30], [40, 50, 60.]
])

# 2 x 2 행렬을 4 x 1로 변형합니다.
x1.view(4,1)
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])

x1.reshape(4,1)
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])

# 2 x 2 행렬을 1차원 벡터로 변형합니다.
x1.view(-1)
# tensor([1,2,3,4])

x1.reshape(-1)
# tensor([1,2,3,4])
        
# -1을 사용하면 shape에서 자동 계산 가능한 부분에 한해서 자동으로 입력 됩니다.
# 계산이 불가능 하면 오류가 발생합니다.
x1.view(1, -1)
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])

x1.reshape(1, -1)
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])

# 2 x 3 행렬을 전치해서 3 x 2 행렬을 만듭니다.
x2.t()
# tensor([[10., 40.],
#         [20., 50.],
#         [30., 60.]])
        
# dim = 1 로 결합하면 2 x 5의 행렬로 묶을 수 있습니다.
torch.cat([x1, x2], dim=1)
# tensor([[ 1.,  2., 10., 20., 30.],
#         [ 3.,  4., 40., 50., 60.]])
        
# transpose(dim0, dim1)을 사용하면 dim0의 차원과 dim1의 차원을 교환합니다.
# transpose(0, 3) 이라고 하면 0차원과 3차원을 교환하게 됩니다.
# 아래 예제는 HWC(높이, 너비, 컬러) 차원을 CHW(컬러, 높이, 너비)로 변형하는 예제입니다.
hwc_img_data = torch.rand(100, 64, 32, 3)
chw_img_data = hwc_img_data.transpose(1,2).transpose(1,3)
chw_img_data.size()
# torch.Size([100, 3, 64, 32])
```

<br>

## **Tensor 생성**

<br>

- Tensor를 생성할 때 대표적으로 사용하는 함수가 `rand`, `zeros`, `ones` 입니다. 이 때, 첫 인자는 dimension 입니다.
- 각 dimension은 tuple 형태로 묶어서 지정해 주어도 되고 콤마 형태로 풀어서 지정해 주어도 됩니다.
- 예를 들어 `torch.rand((2, 3))`와 `torch.rand(2, 3)` 모두 같은 shape인 (2, 3)을 가집니다.
- 먼저 랜덤 넘버 생성에 대하여 다루어 보겠습니다.

```python
import torch
x = torch.rand(2,3)
print(x)

 0.1330  0.3113  0.9652
 0.1237  0.4056  0.8464
[torch.FloatTensor of size 2x3]
```

<br>

- 다음과 같이 순열을 생성할 수도 있습니다.

<br>

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

- 다음은 모든 값이 0인 zeros tensor를 생성해 보도록 하겠습니다.

<br>

```python
zeros = torch.zeros(2,3)

: tensor([[0., 0., 0.],
        [0., 0., 0.]])

torch.zeros_like(zeros)
```

<br>

- 다음은 모든 값이 1인 ones Tensor 생성해 보도록 하겠습니다.

<br>

```python
torch.ones(2,3)

: tensor([[1., 1., 1.],
        [1., 1., 1.]])

```

<br>

- arange를 이용한 Tensor 생성
- torch.arange(시작, 끝, step)을 인자로 사용하며 시작은 이상, 끝은 미만의 범위를 가지고 step만큼 간격을 두며 Tensor를 생성합니다. 

```python
# torch.arange(start,end,step=1) -> [start,end) with step
torch.arange(0,3,step=0.5)

: tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000])
```

<br>

## **Tensor 데이터 타입**

<br>

- Float 타입의 m행 n열 Tensor 생성하기

```python
# 2행 3열의 Float 타입의 Tensor 생성
torch.cuda.FloatTensor(2,3)

: tensor([[2.0000e+00, 3.0000e+00, 1.4013e-45],
        [0.0000e+00, 0.0000e+00, 0.0000e+00]], device='cuda:0')
```

<br>

- 리스트를 입력하여 특정 리스트를 Tensor로 변환하기

```python
torch.cuda.FloatTensor([2,3])

: tensor([2., 3.], device='cuda:0')
```

<br>

- Float 타입을 Int 타입으로 형변환

```python
x = torch.cuda.FloatTensor([2,3])
x.type_as(torch.cuda.IntTensor())

: tensor([2, 3], device='cuda:0', dtype=torch.int32)
```

<br>

## **Numpy to Tensor 또는 Tensor to Numpy**

<br>

- Numpy를 생성한 후 Tensor로 변환한 후 다시 Numpy로 변환해 보고 추가적으로 변환하는 방법도 알아보겠습니다.

<br>

```python
import numpy as np
x1 = np.ndarray(shape=(2,3), dtype=int,buffer=np.array([1,2,3,4,5,6]))
# array([[1, 2, 3],
#        [4, 5, 6]])

torch.from_numpy(x1)
# tensor([[1, 2, 3],
#         [4, 5, 6]], dtype=torch.int32)
        
x2 = torch.from_numpy(x1)
x2.numpy()
# array([[1, 2, 3],
#        [4, 5, 6]])

x2.float()
# tensor([[1., 2., 3.],
#         [4., 5., 6.]])
```

<br>

## **CPU 타입과 GPU 타입의 Tensor**

<br>

- 딥러닝 프레임워크에서는 CPU와 GPU 두 타입에 대한 Tensor 생성이 가능합니다.
- PyTorch에서는 어떻게 사용할 수 있는지 알아보겠습니다.

<br>

```python
x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_gpu = x.cuda()
# 1  2  3
# 4  5  6
# [torch.cuda.FloatTensor of size 2x3 (GPU 0)]

x_cpu = x_gpu.cpu()
x_cpu
# 1  2  3
# 4  5  6
# [torch.FloatTensor of size 2x3]
```

<br>

## **Tensor 사이즈 확인하기**

<br>

- Tensor 사이즈를 확인하려면 `.size()`를 이용하여 확인하면 됩니다.

```python
x = torch.cuda.FloatTensor(10, 12, 3, 3)
x.size()
# torch.Size([10, 12, 3, 3])
```

<br>

## **Index (slicing) 기능 사용방법**

<br>

- Index 또는 slicing 기법은 Tensor에서 **특정 값만 조회**하는 것을 말합니다. 
- 배열, 행렬에서도 인덱스 기능을 통하여 특정 값들을 조회하는 것 처럼 Tensor에서도 조회할 수 있습니다.
- 먼저 `torch.index_select`함수를 이용해 보겠습니다. 파라미터는 input, dim, index가 차례로 입력됩니다. 이 함수는 torch에서 제공하는 인덱싱 방법입니다.

```python
# torch.index_select(input, dim, index)
x = torch.rand(4,3)
# tensor([[0.4898, 0.2505, 0.6500],
#         [0.0976, 0.4117, 0.9705],
#         [0.7069, 0.0546, 0.7824],
#         [0.4921, 0.9863, 0.3936]])
    
# 3번째 인자에는 torch.LongTensor를 이용하여 인덱스를 입력해 줍니다.
torch.index_select(x,0,torch.LongTensor([0,2]))
# tensor([[0.4898, 0.2505, 0.6500],
#         [0.7069, 0.0546, 0.7824]])
```  

<br>

- 하지만 위 처럼 인덱식 하는 방법은 뭔가 python이나 numpy와는 조금 이질적인 감이 있습니다.
- 이번에는 좀더 파이썬스럽게 인덱싱을 해보겠습니다. 아래 방법이 좀더 파이썬 유저에게 친숙한 방법입니다.

<br>

```python
print(x)
# tensor([[0.4898, 0.2505, 0.6500],
#         [0.0976, 0.4117, 0.9705],
#         [0.7069, 0.0546, 0.7824],
#         [0.4921, 0.9863, 0.3936]])

x[:, 0]
# tensor([0.4898, 0.0976, 0.7069, 0.4921])

x[0, :]
# tensor([0.4898, 0.2505, 0.6500])

x[0:2, 0:2]
# tensor([[0.4898, 0.2505],
#         [0.0976, 0.4117]])
```

<br>

- 이번에는 `mask` 기능을 통하여 인덱싱 하는 방법에 대하여 알아보겠습니다.
- `torch.masked_select(input, mask)` 함수를 이용하여 선택할 영역에는 `1`을 미선택할 영역은 `0`을 입력 합니다.
- 인풋 영역과 마스크할 영역의 크기는 같아야 오류 없이 핸들링 할 수 있습니다.

<br>

```python
x = torch.randn(2,3)
# tensor([[ 0.6122, -0.7963, -0.3964],
#         [ 0.6030,  0.1522, -1.0622]])

# mask는 0,1 값을 가지고 ByteTensor를 이용하여 생성합니다.
# (0,3)과 (1,1) 데이터 인덱싱
mask = torch.ByteTensor([[0,0,1],[0,1,0]])
torch.masked_select(x,mask)
# tensor([-0.3964,  0.1522])
```

<br>

## **Join(cat, stack) 기능 사용 방법**

<br>

- PyTorch에서 `torch.cat(seq, dim)`을 이용하여 concaternate 연산을 할 수 있습니다.
- `dim`은 concaternate할 방향을 정합니다.

<br>

```python
x = torch.cuda.FloatTensor([[1, 2, 3], [4, 5, 6]])
y = torch.cuda.FloatTensor([[-1, -2, -3], [-4, -5, -6]])
z1 = torch.cat([x, y], dim=0)
# tensor([[ 1.,  2.,  3.],
#         [ 4.,  5.,  6.],
#         [-1., -2., -3.],
#         [-4., -5., -6.]], device='cuda:0')

z2 = torch.cat([x, y], dim=1)
# tensor([[ 1.,  2.,  3., -1., -2., -3.],
#         [ 4.,  5.,  6., -4., -5., -6.]], device='cuda:0')
```

<br>

- `torch.stack` 을 이용하여도 concaternate를 할 수 있습니다.

<br>

```python
# torch.stack(sequence,dim=0) -> stack along new dim

x = torch.FloatTensor([[1,2,3],[4,5,6]])
x_stack = torch.stack([x,x,x,x],dim=0)
# tensor([[[1., 2., 3.],
#          [4., 5., 6.]],

#         [[1., 2., 3.],
#          [4., 5., 6.]],

#         [[1., 2., 3.],
#          [4., 5., 6.]],

#         [[1., 2., 3.],
#          [4., 5., 6.]]])
```

<br>

## **slicing 기능 사용 방법**

<br>

- slicing 기능은 Tensor를 몇개의 부분으로 나뉘는 기능입니다.
- `torch.chunk(tensor, chunks, dim=0)` 또는 `torch.split(tensor,split_size,dim=0)`함수를 이용하여 Tensor를 나뉠 수 있습니다.

<br>

```python
# torch.chunk(tensor, chunks, dim=0) -> tensor into num chunks

x_1, x_2 = torch.chunk(z1,2,dim=0)
y_1, y_2, y_3 = torch.chunk(z1,3,dim=1)

print(z1)
#   1  2  3
#   4  5  6
#  -1 -2 -3
#  -4 -5 -6
#  [torch.FloatTensor of size 4x3]
 
 print(x1)
#   1  2  3
#   4  5  6
#  [torch.FloatTensor of size 2x3], 
 
 print(x2)
#  -1 -2 -3
#  -4 -5 -6
#  [torch.FloatTensor of size 2x3]
 
 print(y1)
#   1
#   4
#  -1
#  -4
#  [torch.FloatTensor of size 4x1], 
 
 print(y2)
#   2
#   5
#  -2
#  -5
#  [torch.FloatTensor of size 4x1]
 
 print(y3)
#   3
#   6
#  -3
#  -6
#  [torch.FloatTensor of size 4x1]
```

<br>

## **squeezing 기능 사용 방법**

<br>

- squeeze 함수를 사용하면 dimension 중에 1로 되어 있는 것을 압축할 수 있습니다.
- dimension이 1이면 사실 불필요한 차원일 수 있기 때문에 squeeze를 이용하여 압축 시키는 것이 때론 필요할 수 있는데, 그 때 사용하는 함수 입니다.
- `torch.squeeze(input, dim)`으로 사용할 수 있고, dim을 지정하지 않으면 dimeion이 1인 모든 차원을 압축하고 dim을 지정하면 지정한 dimension만 압축합니다.

```python
>>> x = torch.zeros(2, 1, 2, 1, 2)
>>> x.size()
# torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x)
>>> y.size()
# torch.Size([2, 2, 2])
>>> y = torch.squeeze(x, 0)
>>> y.size()
# torch.Size([2, 1, 2, 1, 2])
>>> y = torch.squeeze(x, 1)
>>> y.size()
# torch.Size([2, 2, 1, 2])
```

<br>

- 반면 unsqueeze 함수를 사용하면 dimension을 추가할 수 있습니다. squeeze와 정확히 반대라고 보시면 됩니다.
- unsqueeze 함수는 dimension을 반드시 입력 받게 되어 있습니다.

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

## **Initialization, 초기화 방법**

<br>

- `init.uniform`함수를 사용하면 `uniform` 또는 `normal` 분포의 초기화 Tensor를 만들 수 있습니다.
- 또는 상수 형태를 바로 만들 수도 있습니다. 예제는 아래와 같습니다.

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

## **Math Operation**

<br>

- `dot` : 벡터 내적
- `mv` : 행렬과 벡터의 곱
- `mm` : 행렬과 행렬의 곱
- `matmul` : 인수의 종류에 따라서 자동으로 dot, mv, mm을 선택

```python
a = torch.tensor([1,2,3,4,5,6]).view(3,2)
b = torch.tensor([9,8,7,6,5,4]).view(2,3)
ab = torch.matmul(a,b)
ab = a@b # @ 연산자를 이용하여 간단하게 행렬곱을 표현할 수 있음
```

<br>

- Tensor의 산술 연산 방법에 대하여 알아보겠습니다.
- `+`연산자 또는 `torch.add()`

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

- `+`연산자를 이용한 broadcasting 또는 `torch.add() with broadcasting`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1 + 10 # torch.add(x1, 10) 또한 가능합니다.
11  12  13
14  15  16
[torch.FloatTensor of size 2x3]
```

<br>

- `*`연산자 또는 `torch.mul()`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1*x2 # torch.mul(x1,x2)
1   4   9
16  25  36
[torch.FloatTensor of size 2x3]
```
<br>

- `*`연산자를 이용한 broadcasting 또는 `torch.mul() with broadcasting`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1 * 10
10  20  30
40  50  60
[torch.FloatTensor of size 2x3]
```

<br>

- `/`연산자 또는 `torch.div()`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
x2 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1/x2 # torch.div(x1, x2)
 1  1  1
 1  1  1
[torch.FloatTensor of size 2x3]

```

<br>

- `/`연산자를 이용한 broadcsting 또는 `torch.div() with broadcasting`

```python
x1 = torch.FloatTensor([[1,2,3],[4,5,6]])
>>> x1 / 5
 0.2000  0.4000  0.6000
 0.8000  1.0000  1.2000
[torch.FloatTensor of size 2x3]
```

<br>

- power 연산 : `torch.pow(input,exponent)`

```python
x1 = torch.FloatTensor([ [1,2,3], [4,5,6] ])
>>> x1**2 # torch.pow(x1,2)
tensor([[ 1.,  4.,  9.],
        [16., 25., 36.]])
```
<br>

- exponential 연산 : `torch.exp(tensor,out=None)`

```python
x1 = torch.FloatTensor([ [1,2,3], [4,5,6] ])
>>> torch.exp(x1)
tensor([[  2.7183,   7.3891,  20.0855],
        [ 54.5981, 148.4132, 403.4288]])
```

<br>

- `torch.log(input, out=None) -> natural logarithm`

```python
x1 = torch.FloatTensor([ [1,2,3], [4,5,6] ])
>>> torch.log(x1)
tensor([[0.0000, 0.6931, 1.0986],
        [1.3863, 1.6094, 1.7918]])
```

<br>

- `torch.mm(mat1, mat2) -> matrix multiplication`
- Tensor(행렬)의 곱을 연산하므로 shape이 잘 맞아야 연산이 가능합니다.

```python
x1 = torch.FloatTensor(3,4)
x2 = torch.FloatTensor(4,5)
torch.mm(x1,x2).size()
```

<br>

- `torch.bmm(batch1, batch2) -> batch matrix multiplication`
- Tensor(행렬)의 곱을 batch 단위로 처리합니다. `torch.mm`에서는 단일 Tensor(행렬)로 계산을 한 반면에 batch 단위로 한번에 처리하므로 좀 더 효율적입니다.

```python
x1 = torch.FloatTensor(10,3,4)
x2 = torch.FloatTensor(10,4,5)
torch.bmm(x1,x2).size()
```

<br>

- `torch.dot(tensor1,tensor2)`는 두 tensor의 dot product 연산을 수행합니다.

```python
>>> torch.dot(torch.tensor([2, 3]), torch.tensor([2, 1]))
tensor(7)
```

<br>

- `torch.t()`를 이용하면 transposed tensor를 구할 수 있습니다.

```python
x1 = torch.FloatTensor(2,3)
x2 = x1.t()
>>> print(x1.size())
torch.Size([2, 3])

>>> print(x2.size())
torch.Size([3, 2])
```

<br>

- 반면 `torch.transpose()`를 이용하여 특정 dimension을 변경할 수 있습니다.

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

- `eigenvalue`와 `eigenvector`를 구하는 방법은 아래와 같습니다.
- 출력은 각각 `eigenvalue`와 `eigenvector`입니다.

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

<br>

## **Gradient를 구하는 방법**

<br>

- Pytorch를 이용하여 Gradient를 구하는 방법에 대하여는 제 블로그의 다른 글에서 자세하게 다루었습니다.
    - 링크 : https://gaussian37.github.io/dl-pytorch-gradient/

<br>

## **벡터와 텐서의 element-wise multiplication**

<br>

- 딥러닝에서 연산을 하다보면 다음과 같은 그림의 연산을 하는 경우가 종종 발생합니다.

<br>
<center><img src="../assets/img/dl/pytorch/basic/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다음 코드를 통해 `channel` 크기가 같은 벡터와 텐서를 생성해 보겠습니다.

<br>

```python
A = torch.ones(5, 3, 3)
# tensor([[[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]],

#         [[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]],

#         [[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]],

#         [[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]],

#         [[1., 1., 1.],
#          [1., 1., 1.],
#          [1., 1., 1.]]])

v = torch.arange(1, 5)
# tensor([1, 2, 3, 4, 5])
```

<br>
<center><img src="../assets/img/dl/pytorch/basic/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>