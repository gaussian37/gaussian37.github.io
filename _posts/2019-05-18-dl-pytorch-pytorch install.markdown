---
layout: post
title: PyTorch 설치 및 colab 사용 방법
date: 2019-05-18 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, pytorch 설치, colab] # add tag
---

+ 이번 글에서는 PyTorch를 사용할 때, 윈도우에서 설치하는 방법과 colab에서 설치하는 방법에 대하여 알아보겠습니다.
- **가장 최신의 설치 방법**을 확인하고 싶으면 https://pytorch.org/get-started/locally/ 링크를 참조하는 것이 좋습니다.
- 아래 쓴 글의 내용은 시간이 지나면 틀릴 수도 있습니다. `pytorch` 공식 링크를 따르는 것을 추천드립니다.

<br>

### GPU 세팅하는 방법

<br>

- 아래 방법은 '17.11.18 시점 기준 GPU 세팅 방법입니다.
- CUDA를 설치해야 합니다. [CUDA](https://developer.nvidia.com/cuda-downloads)에서 본인 컴퓨터에 맞는 사양으로 설치하시면 됩니다.
- cuDNN 파일을 컴퓨터에 붙어넣어야 합니다.
    - [cuDNN](https://developer.nvidia.com/cudnn) 에서 컴퓨터 사양에 맞는 버전을 다운 받습니다.
    - 다운 받은 파일의 압축을 풀어 cuda/bin, cuda/incldue, cuda/lib의 폴더와 파일을 다음 경로에 붙여 넣습니다. (아래 경로 참조 하시어 컴퓨터 실제 경로에 맞도록 붙여넣습니다.)
        - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0 (또는 9.0)
- 제어판 내 시스템 => 고급 시스템 설정 => 시스템 속성 의 고급 탭 => 환경 변수 에서 아래 경로를 추가합니다. 경로 추가 시 어떤 path에서도 추가된 경로는 접근 가능해 집니다.
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0\bin
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0\libnvvp

<br>

### **윈도우에서 PyTorch 설치하는 방법**

<br>

+ GPU를 범용적으로 사용하기 위해 `CUDA`를 설치해줍니다.
+ 딥러닝을 사용하기 위해 `cuDNN`을 설치해줍니다.
+ 아나콘다를 설치해 줍니다.
+ 아나콘다를 설치하고 차례 차례 다음을 설치해 줍니다.
+ `conda create -n 가상환경이름 python=3.6`
+ `activate 가상환경이름`
+ `conda install pytorch -c pytorch`
    - 또는 `pip install torch` (윈도우에서는 pytorch가 아니라 torch로 pytorch를 설치합니다.)
+ `conda install -c pytorch torchvision`
    + 또는 `pip install torchvision` 
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

### **colab에서 PyTorch 사용하는 방법**

<br>

+ colab에서는 기본적으로 PyTorch가 설치되어 있지 않습니다.
+ 따라서 매 세션마다 PyTorch를 설치를 해주어야 합니다. 금방 설치되니 설치하고 사용하면 문제가 없습니다.
+ 먼저 colab 메뉴의 `runtime` → `Change runtime type`을 선택해서 `GPU`를 선택해 줍니다.
    + 딥러닝을 사용하는 것이니 GPU를 선택해 주어야 합니다.
+ 아래 코드를 입력하여 PyTorch를 설치해줍니다.

```python
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl
!pip3 install torchvision
```

<br>

### **PyTorch에서 model summary 보는 방법**

<br>

+ 정의한 model을 print 해서 보는 방법
+ Keras 형태의 summary를 보는 방법
    + `pip install torchsummary` 로 torchsummary를 설치합니다.
    + `summary(model.cuda(), input_size)` 를 실행합니다.
