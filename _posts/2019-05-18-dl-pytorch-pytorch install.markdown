---
layout: post
title: PyTorch 설치 및 colab 사용 방법
date: 2019-05-01 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, pytorch 설치, colab] # add tag
---

+ 이번 글에서는 PyTorch를 사용할 때, 윈도우에서 설치하는 방법과 colab에서 설치하는 방법에 대하여 알아보겠습니다.
- **가장 최신의 설치 방법**을 확인하고 싶으면 https://pytorch.org/get-started/locally/ 링크를 참조하는 것이 좋습니다.
- 아래 쓴 글의 내용은 시간이 지나면 틀릴 수도 있습니다. `pytorch` 공식 링크를 따르는 것을 추천드립니다.

<br>

## **GPU 세팅하는 방법**

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

## **윈도우에서 PyTorch 설치하는 방법**

<br>

+ GPU를 범용적으로 사용하기 위해 `CUDA`를 설치해줍니다.
+ 딥러닝을 사용하기 위해 `cuDNN`을 설치해줍니다. 여기까지는 앞의 과정을 그대로 따라하시면 됩니다.
- 그 다음 과정 부터는 `conda`를 설치한 후 진행하는 방법과 `pip`를 이용하는 방법이 있습니다.
- 1) `conda`를 설치하려면 다음 링크를 통해 설치 파일을 받은 후 설치하면 됩니다.
    - https://www.anaconda.com/products/individual
- 2) `pip`를 이용하여 설치하려면 `virtualenv`를 이용하여 가상환경을 독립적으로 만들고 그 환경에서 설치하는 것을 권장드립니다. 이 방법을 이용하려면 다음 링크를 참조하시기 바랍니다.
    - https://gaussian37.github.io/python-concept-initial_setting/
- 위의 `conda` 설치 또는 `pip` 설치 과정이 끝났으면 `pytorch`를 설치 할 가상 환경을 활성화 합니다.
- 1) `conda`의 경우 다음과 같이 가상 환경을 활성화 합니다.
    - 가상 환경 생성 : `conda create -n 가상환경이름 python=3.6`
    - 가상 환경 활성화 : `activate 가상환경이름`
- 2) `pip` 의 경우 다음과 같이 가상 환경을 활성화 합니다.
    - 가상 환경 생성 : `virtualenv 가상환경이름`
    - 가상 환경 활성화 : `가상환경이름\Scripts\activate`
- 위 두가지 방법 모두 가상 환경 활성화에 성공하면 command prompt의 가장 앞에 `(가상환경이름)`이 추가된 것을 확인할 수 있습니다.
- 그 다음에는 `pytorch`, `torchvision` 을 설치해 보도록 하겠습니다. 반드시 다음 링크를 접속 하시기 바랍니다.
    - https://pytorch.org/get-started/locally/
- 위 링크에서 현재 설치하려는 환경의 옵션을 차례대로 선택합니다. 그러면 설치에 필요한 명령어가 생성됩니다. 그 명령어를 커맨드에 입력하여 pytorch와 torchvision을 설치하면 됩니다.
- 저의 경우 예를 들어 stable, windows, pip, python, cuda10.2ㄹ를 옵션으로 선택하였습니다. 이 때 설치 커맨드는 다음과 같습니다.
    - pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html

<br>

- 실행이 정상적으로 되는지 확인하기 위해 다음 코드를 `print(x)` 까지 입력하였을 때, 출력이 정상정으로 되는 지 확인하면 됩니다.
- 여기서 `.cuda()` 명령어가 GPU를 이용하는 명령어 입니다. 만약 CUDA, cuDNN이 정상적으로 설치되지 않았다면 오류가 발생할 수 있습니다.
- 만약 오류가 발생하였을 경우 아래 명령어에서 `torch.Tensor(3, 4).cuda()` 대신 `torch.Tensor(3, 4)`을 사용하여 실행해 보시길 바랍니다. 이 경우 문제가 없다면 CUDA, cuDNN 이 설치가 잘못된 것입니다. 반면 이 경우에도 문제가 발생하였다면 pytorch 설치가 잘못된 것입니다.

```python
import torch
x = torch.Tensor(3, 4).cuda()
print(x)

tensor([[-5.5635e-24,  0.0000e+00,  4.4842e-44,  0.0000e+00],
        [        nan,  0.0000e+00,  2.6104e-09,  2.1162e-07],
        [ 1.6899e-04,  2.1240e+20,  1.0919e-05,  1.6969e-07]], device='cuda:0')
```

<br>

## **colab에서 PyTorch 사용하는 방법**

<br>

+ colab에서는 기본적으로 PyTorch가 설치되어 있지 않습니다.
+ 따라서 매 세션마다 PyTorch를 설치를 해주어야 합니다. 금방 설치되니 설치하고 사용하면 문제가 없습니다.
+ 먼저 colab 메뉴의 `runtime` → `Change runtime type`을 선택해서 `GPU`를 선택해 줍니다.
    + 딥러닝을 사용하는 것이니 GPU를 선택해 주어야 합니다.
+ 아래 코드를 입력하여 PyTorch를 설치해줍니다.

<br>

```python
!pip3 install torch
!pip3 install torchvision
```

<br>

- 만약 `colab`에서 google drive의 파일 또는 디렉토리를 연결해서 사용하려면 아래 명령어를 입력하여 실행하면 연결된 계정의 google drive를 사용할 수 있습니다.

<br>

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

<br>

## **PyTorch에서 model summary 보는 방법**

<br>

+ 정의한 model을 print 해서 보는 방법
+ Keras 형태의 summary를 보는 방법
    + `pip install torchsummary` 로 torchsummary를 설치합니다.
    + `summary(model.cuda(), input_size)` 를 실행합니다.
