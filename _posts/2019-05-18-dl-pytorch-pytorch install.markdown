---
layout: post
title: PyTorch 설치 및 colab 사용 방법
date: 2019-05-18 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, pytorch 설치, colab] # add tag
---

+ 이번 글에서는 PyTorch를 사용할 때, 윈도우에서 설치하는 방법과 colab에서 설치하는 방법에 대하여 알아보겠습니다.

<br>

### 윈도우에서 PyTorch 설치하는 방법

+ GPU를 범용적으로 사용하기 위해 `CUDA`를 설치해줍니다.
+ 딥러닝을 사용하기 위해 `cuDNN`을 설치해줍니다.
+ 아나콘다를 설치해 줍니다.
+ 아나콘다를 설치하고 차례 차례 다음을 설치해 줍니다.
+ `conda create -n 가상환경이름 python=3.6`
+ `activate 가상환경이름`
+ `conda install pytorch -c pytorch`
+ `pip install torchvision` 
+ `conda install ipykernel`
+ `pip install jupyter`

+ 실행이 정상적으로 되는지 확인하기 위해 jupyter notebook에서 다음을 실행해 봅니다.

```python
import torch
x = torch.Tensor(3, 4).cuda()
print(x)

tensor([[-5.5635e-24,  0.0000e+00,  4.4842e-44,  0.0000e+00],
        [        nan,  0.0000e+00,  2.6104e-09,  2.1162e-07],
        [ 1.6899e-04,  2.1240e+20,  1.0919e-05,  1.6969e-07]], device='cuda:0')
```

<br>

### colab에서 PyTorch 사용하는 방법

+ colab에서는 기본적으로 PyTorch가 설치되어 있지 않습니다.
+ 따라서 매 세션마다 PyTorch를 설치를 해주어야 합니다. 금방 설치되니 설치하고 사용하면 문제가 없습니다.
+ 먼저 colab 메뉴의 `runtime` → `Change runtime type`을 선택해서 `GPU`를 선택해 줍니다.
    + 딥러닝을 사용하는 것이니 GPU를 선택해 주어야 합니다.
+ 아래 코드를 입력하여 PyTorch를 설치해줍니다.

```python
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip3 install torchvision
```
