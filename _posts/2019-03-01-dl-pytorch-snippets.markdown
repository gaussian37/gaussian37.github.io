---
layout: post
title: pytorch 코드 snippets
date: 2019-03-01 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, snippets] # add tag
---

<br>

- 이 글은 pytorch 사용 시 참조할 수 있는 코드들을 모아놓았습니다.
- 완전히 기본 문법은 [이 글](https://gaussian37.github.io/dl-pytorch-pytorch-tensor-basic/)에서 참조하시기 바랍니다.
- 이 글에서는 `셋팅 관련` 내용, `자주 사용하는 함수` 그리고 `자주 사용하는 코드` 순으로 정리하였습니다.

<br>

## **목차**

<br>

- ### **--- 셋팅 관련 ---**
- ### pytorch import 모음
- ### pytorch 셋팅 관련 코드
- ### GPU 셋팅 관련 코드

<br>

- ### **--- 자주사용하는 함수 ---**
- ### torch.argmx(input, dim, keepdim)
- ### torch.from_numpy(numpy.ndarray)
- ### torch.unsqueeze(input, dim)
- ### torch.squeeze(input, dim)
- ### Variable(data)

<br>

- ### **--- 자주 사용하는 코드 모음 ---**
- ### weight 초기화 방법
- ### load와 save 방법

<br>

## **pytorch import 모음**

<br>

```python
import torch
import torchvision
import torch.nn as nn # neural network 모음. (e.g. nn.Linear, nn.Conv2d, BatchNorm, Loss functions 등등)
import torch.optim as optim # Optimization algorithm 모음, (e.g. SGD, Adam, 등등)
import torch.nn.functional as F # 파라미터가 필요없는 Function 모음
from torch.utils.data import DataLoader # 데이터 세트 관리 및 미니 배치 생성을 위한 함수 모음
import torchvision.datasets as datasets # 표준 데이터 세트 모음
import torchvision.transforms as transforms # 데이터 세트에 적용 할 수있는 변환 관련 함수 모음
from torch.utils.tensorboard import SummaryWriter # tensorboard에 출력하기 위한 함수 모음
import torch.backends.cudnn as cudnn # cudnn을 다루기 위한 값 모음

from torchsummary import summary # summary를 통한 model의 현황을 확인 하기 위함
import torch.onnx # model을 onnx 로 변환하기 위함
```

<br>

## **pytorch 셋팅 관련 코드**

<br>

```python
# pytorch 내부적으로 사용하는 seed 값 설정
torch.manual_seed(seed)

# cuda를 사용할 경우 pytorch 내부적으로 사용하는 seed 값 설정
torch.cuda.manual_seed(seed) 
```

<br>

## **GPU/CPU Device 세팅 코드**

<br>

```python
# cuda가 사용 가능한 지 확인
torch.cuda.is_available()

# cuda가 사용 가능하면 device에 "cuda"를 저장하고 사용 가능하지 않으면 "cpu"를 저장한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 멀티 GPU 사용 시 사용 가능한 GPU 셋팅 관련
# 아래 코드의 "0,1,2"는 GPU가 3개 있고 그 번호가 0, 1, 2 인 상황의 예제입니다.
# 만약 GPU가 5개이고 사용 가능한 것이 0, 3, 4 라면 "0,3,4" 라고 적으면 됩니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# 현재 PC의 사용가능한 GPU 사용 갯수 확인
torch.cuda.device_count()

# cudnn을 사용하도록 설정. GPU를 사용하고 있으면 기본값은 True 입니다.
import torch.backends.cudnn as cudnn
cudnn.enabled = True

# inbuilt cudnn auto-tuner가 사용 중인 hardware에 가장 적합한 알고리즘을 선택하도록 허용합니다.
cudnn.benchmark = True
```

<br>

- 위 코드와 같이 device의 유형을 선택하면 GPU가 존재하면 `cuda:0`에 할당되고 GPU가 없으면 `cpu`에 할당 되도록 할 수 있습니다.

<br>

## **torch.argmx(input, dim, keepdim)**

<br>

- 딥러닝 각 framework마다 존재하는 `argmax` 함수에 대하여 다루어 보겠습니다. keras, tensorflow와 사용법이 유사하니 참조하시면 됩니다. 아래 링크는 pytorch의 링크입니다.
    - pytorch 링크 : https://pytorch.org/docs/stable/torch.html
- argmax 함수가 받는 argument는 차례대로 `input`, `dim`, `keepdim` 입니다. `input`은 Tensor를 나타내고 `dim`은 몇번 째 축을 기준으로 argmax 연산을 할 지 결정합니다. 마지막으로 `keepdim`은 argmax 연산을 한 축을 생략할 지 그대로 둘 지에 대한 기준이 됩니다. argmax를 하면 각 축마다 값이 1개만 남게 되므로 필요 여부에 따라 남길 수도 있고 삭제 할 수도 있습니다.

<br>

- pytorch를 이용하여 이미지 처리를 할 때, 주로 사용하는 방법은 다음과 같습니다.

<br>

```python
# 1. input이 (channel, height, widht) 인 경우
torch.argmax(input, dim = 0, keepdim = True)

# 2. input이 (batch, channel, height, width) 인 경우
torch.argmax(input, dim = 1, keepdim = True) 
```

<br>

- 일반적으로 이미지 처리를 할 때, 출력의 `channel`의 갯수 만큼 클래스 label을 가지고 있는 경우가 많습니다. 이 때, 가장 큰 값을 argmax 함으로써 가장 큰 인덱스를 구할 수 있습니다.
- 예를 들어 segmentation을 하는 경우 위의 코드와 같은 형태가 그대로 사용될 수 있습니다. 1번 케이스의 경우 batch가 고려되지 않은 것이고 2번 케이스의 경우 batch가 고려된 것입니다. segmentation의 경우 이미지의 height, width의 크기에 channel의 갯수가 label의 갯수와 동일하게 되어 있습니다. 그 중 가장 큰 값을 가지는 channel이 그 픽셀의 label이 되게 됩니다.
- 따라서 argmax를 취하면 `channel`은 1로 되고 height와 width의 크기는 유지됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/snippets/1.png" alt="Drawing" style="height: 400px;"/></center>
<br>

- 위 그림을 예제로 이용하여 각 축에 대하여 argmax한 결과를 알아보도록 하겠습니다.
- 아래 예제 코드에서는 0번째 (channel), 1번째 (height), 2번째 (width) 방향으로 각각 argmarx를 한 것입니다. 또한 `keepdim`을 기본값인 False로 둘 경우와 True로 둘 경우를 구분하여 어떻게 `shape`이 변화하는 지 살펴보았습니다.

<br>

```python

# channel : 3, height : 4, width : 2로 가정합니다.
>> A = torch.randint(10, (3, 4, 2))
>> print(A)

tensor([[[6, 4],
         [8, 0],
         [8, 4],
         [2, 4]],

        [[2, 8],
         [3, 6],
         [4, 7],
         [0, 9]],

        [[8, 0],
         [3, 9],
         [0, 5],
         [7, 3]]])

# 0번째 축(channel) 기준 argmax w/o Keepdim
 >> torch.argmax(A, dim=0)
tensor([[2, 1],
        [0, 2],
        [0, 1],
        [2, 1]])

>> torch.argmax(A, dim=0).shape
torch.Size([4, 2])

# 0번째 축(channel) 기준 argmax w/ Keepdim
>> torch.argmax(A, dim=0, keepdim = True)
tensor([[[2, 1],
         [0, 2],
         [0, 1],
         [2, 1]]])

 >> torch.argmax(A, dim=0, keepdim = True).shape
 torch.Size([1, 4, 2])

# 1번째 축(height) 기준 argmax w/o Keepdim
>> torch.argmax(A, dim=1)
tensor([[2, 3],
        [2, 3],
        [0, 1]])
 
 >> torch.argmax(A, dim=1).shape
torch.Size([3, 2])


# 1번째 축(height) 기준 argmax w/ Keepdim
>> torch.argmax(A, dim=1, keepdim = True)
tensor([[[2, 3]],

        [[2, 3]],

        [[0, 1]]])

>> torch.argmax(A, dim=1, keepdim = True).shape
torch.Size([3, 1, 2])


# 2번째 축(width) 기준 argmax w/o Keepdim
>> torch.argmax(A, dim=2)
tensor([[0, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0]])

>> torch.argmax(A, dim=2).shape
torch.Size([3, 4])

# 2번째 축(width) 기준 argmax w/ Keepdim
>> torch.argmax(A, dim=2, keepdim=True)
tensor([[[0],
         [0],
         [0],
         [1]],

        [[1],
         [1],
         [1],
         [1]],

        [[0],
         [1],
         [1],
         [0]]])


>> torch.argmax(A, dim=2, keepdim=True).shape
torch.Size([3, 4, 1])

```

<br>
<center><img src="../assets/img/dl/pytorch/snippets/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 보면 왼쪽 부터 0번째 축, 1번째 축, 2번째 축 방향을 기준으로 argmax하였을 때 선택된 결과를 노란색 음영으로 표시하였습니다. 위 코드 결과를 그대로 표현한 것으로 이해하는 데 참조하시기 바랍니다.

<br>

- 이번에는 간단하게 height와 width만 고려하여 다루어 보겠습니다.

<br>

```python
# 간단하게 height, width의 크기만을 이용하여 다루어 보겠습니다.
>> B = torch.rand(3, 2)

tensor([[0.8425, 0.3970],
        [0.5268, 0.7384],
        [0.5639, 0.3080]])

# 1번 예제. 아래 첫번째 그림 참조
>> torch.argmax(B, dim=0)
tensor([0, 1])

>> torch.argmax(B, dim=0, keepdim=True)
tensor([[0, 1]])

# 2번 예제. 아래 두번째 그림 참조
>> torch.argmax(B, dim=1)
tensor([0, 1, 0])

>> torch.argmax(B, dim=1, keepdim=True)
tensor([[0],
        [1],
        [0]]) 
```

<br>

- 위의 1번 예제에 해당하는 그림입니다. 매트릭스에서 0번째 축은 세로(height)축입니다. 따라서 각 열에서 세로 방향으로 최대값이 선택됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/snippets/3.png" alt="Drawing" style="height: 400px;"/></center>
<br>

- 다음으로 2번 예제에 해당하는 그림입니다. 매트릭스에서 1번째 축은 가로(width)축입니다. 따라서 각 행에서 가로 방향으로 최댁밧이 선택됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/snippets/4.png" alt="Drawing" style="height: 400px;"/></center>
<br>

## **torch.from_numpy(numpy.ndarray)**

<br>

- torch에서 numpy를 이용해 선언한 데이터를 Tensor 타입으로 변환하려면 아래와 같이 사용할 수 있습니다.

<br>

```python
A = np.random.rand(3, 100, 100)
torch.from_numpy(A)
```

<br>

## **weight 초기화 방법**

<br>

- 이번 글에서는 딥러닝 네트워크 모델에서 각 layer 및 연산에 접근하여 weight를 초기화 하는 방법에 대하여 다루어 보도록 하겠습니다.
- 아래 코드에서 예제로 사용하는 [He initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_)은 링크를 참조하시기 바랍니다.

<br>

```python
# 모든 neural network module, nn.Linear, nn.Conv2d, BatchNorm, Loss function 등.
import torch.nn as nn 
# 파라미터가 없는 함수들 모음
import torch.nn.functional as F 

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=(3,3),
                stride=(1,1),
                padding=(1,1)
        )
        self.fc1 = nn.Linear(16*7*7, num_classes)
        # 예제의 핵심인 initialize_weights()로 __init__()이 호출될 때 실행됩니다.
        self.initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        
        return x
    
    # 각 지정된 연산(Conv2d, BatchNorm2d, Linear)에 따라 각기 다른 초기화를 줄 수 있습니다.
    def initialize_weights(self):
        for m in self.modules():
            # convolution kernel의 weight를 He initialization을 적용한다.
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                
                # bias는 상수 0으로 초기화 한다.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        
if __name__ == '__main__':
    model = CNN(in_channels=3,num_classes=10)
    
    # He initialization과 Constant로 초기화 한것을 확인할 수 있습니다.
    for param in model.parameters():
        print(param)
```

<br>

## **load와 save 방법**

<br>

- pytorch에서 학습한 결과를 저장하고 불러오는 방법에 대하여 간략하게 다루어 보겠습니다.
- 아래 코드에서 핵심 함수인 `save_checkpoint`와 `load_checkpoint`를 자세히 살펴보면 어떻게 저장하고 불러오는 지 알 수 있습니다.
- 이 부분의 구조를 알기 위해서는 `state_dict`에 대하여 이해해야 합니다. 먼저 `state_dict`는 dictionary 입니다. 따라서 이 형태에 맞게 데이터를 쉽게 저장하거나 불러올 수 있습니다.
- `state_dict`에는 각 계층을 매개변수 Tensor로 매핑합니다.(dictionary 이므로 mapping에 용이합니다.) 이 때, 학습 가능한 매개변수를 갖는 계층(convolution layer, linear layer 등)등이 모델의 `state_dict` 에 항목을 가지게 됩니다.
- 옵티마이저 객체(torch.optim) 또한 옵티마이저의 상태 뿐만 아니라 사용된 Hyperparameter 정보가 포함된 state_dict를 갖습니다. 
- `inference`를 위해 모델을 저장할 때는 학습된 모델의 학습된 **매개변수만 저장**하며 방법은 `torch.save()` 함수를 이용합니다.
- PyTorch에서는 모델을 저장할 때 `.pt` 또는 `.pth` 확장자를 사용하는 것이 일반적인 규칙이며 아래 코드와 같이 `tar`를 통한 압축 형태로 `*.pth.tar`와 같이 많이 사용합니다.
- 이 이후에 `inference` 용도로 사용하려면 반드시 `model.eval()`을 실행하여 dropout 및 batch normalization이 evaluation 모드로 설정되도록 해야 합니다.

<br>

```python
# Imports
import torch
import torchvision
# neural network modules
import torch.nn as nn 
# Optimization algorithms, SGD, Adam, etc.
import torch.optim as optim 
# parameter가 필요 없는 함수들
import torch.nn.functional as F
# dataset 관리와 mini batch 생성 관련
from torch.utils.data import DataLoader
# standard dataset 접근
import torchvision.datasets as datasets
# dataset transformation을 통한 augmentation
import torchvision.transforms as transforms 

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    # Initialize network
    model = torchvision.models.vgg16(pretrained=False)
    optimizer = optim.Adam(model.parameters())
    
    checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
    # Try save checkpoint
    save_checkpoint(checkpoint)
    
    # Try load checkpoint
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    # model.eval()

if __name__ == '__main__':
    main()
```

<br>

## **torch.unsqueeze(input, dim)**

<br>

- Tensor의 dimension을 맞추기 위해서 dimension을 변경해야 할 경우가 있습니다. 특히 이번에 알아볼 경우는 dimension을 축소하는 경우입니다.
- dimension 축소를 위해서는 `tensor.unsqueeze()` 함수를 이용하고 아래와 같이 응용할 수 있습니다.
- `torch.unsqueeze()` 함수는 **어떤 dimension의 값이 1일 때, 그 dimension을 제거**해 줍니다.

<br>

```python
import torch

tensor = torch.rand(1, 5, 5)
print(tensor.shape)
# torch.Size([5, 5])

tensor = torch.squeeze(tensor, 0)
print(tensor.shape)
# torch.Size([5, 5])

tensor = torch.rand(1, 5, 5)
tensor = torch.squeeze(tensor, 1)
print(tensor.shape)
# torch.Size([1, 5, 5])
```

<br>

- 위 예제와 같이 `torch.squeeze(tensor, 1)`에서는 dimension의 값이 1이 아니므로 dimension이 제거 되지 않았습니다.

<br>

## **torch.squeeze(input, dim)**

<br>

- 이번에는 바로 앞의 `unsqueeze` 예제를 이어 Tensor의 dimension을 늘려보도록 하겠습니다.
- `torch.unsqueeze(input, dim)`은 squeeze와는 반대로 **diemnsion을 늘려주고 그 값은 1**로 만듭니다.
-  이 때 선택할 수 있는 dimension은 0 부터 마지막 dimension 까지 입니다. 예를 들어 원래 input의 dimension이 2이면 0 (맨 앞), 1 (가운데), 2 (맨 끝)에 dimension을 늘려줄 수 있습니다.

<br>

```python
import torch

tensor = torch.rand(5, 5)
print(tensor.shape)
# torch.Size([5, 5])

tensor = torch.unsqueeze(tensor, 0)
print(tensor.shape)
# torch.Size([1, 5, 5])

tensor = torch.rand(5, 5)
tensor = torch.unsqueeze(tensor, 1)
print(tensor.shape)
# torch.Size([5, 1, 5])

tensor = torch.rand(5, 5)
tensor = torch.unsqueeze(tensor, 2)
print(tensor.shape)
# torch.Size([5, 5, 1])
```

<br>

## **Variable(data)**

<br>

- `Variable`은 `from torch.autograd import Variable`을 통해 import 할 수 있습니다. 
- Variable은 tensor에 데이터를 집어 넣을 때, `Variable` 타입으로 기존의 데이터를 변경하여 사용하곤 합니다.
- Variable을 생성할 때 가장 많이 사용하는 옵션 중 하나는 Variable이 학습이 필요한 weight 인 지 아닌 지 지정해 주는 옵션입니다. 다음과 같이 사용할 수 있습니다.

<br>

```python
import torch
from torch.autograd import Variable

v1 = Variable(torch.rand(3), requires_grad = True)
print(v1)
# tensor([0.8407, 0.9296, 0.6941], requires_grad=True)

with torch.no_grad():
    v2 = Variable(torch.rand(3))
print(v2)
# tensor([0.3445, 0.2108, 0.4271])
```

<br>

- 위 처럼  `with torch.no_grad():` 로 감싸준 경우 명확하기 학습이 필요 없음을 명시하므로 가독성에 좋습니다. 즉, `inference` 용도로만 사용한다는 뜻입니다.
- 이전에는 이를 `volatile` 옵션을 사용하기도 하였습니다. 예를 들어 `volatile = True`의 경우 inference 용도로만 사용한 경우로 위의 no_grad()와 동일한 목적의 Variable로 해석할 수 있습니다. 가끔씩 보이는 legacy 코드에서 volatile 파라미터가 있다면 True 인 경우 inference 용도인 것으로 해석하시면 됩니다.



