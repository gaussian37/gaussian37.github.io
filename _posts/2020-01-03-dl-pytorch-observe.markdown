---
layout: post
title: Pytorch의 딥러닝 네트워크 및 학습 현황 확인
date: 2020-01-03 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch]
tags: [pytorch, summary, ] # add tag
---

<br>

- 이 글에서는 Pytorch에서 학습 현황이나 모델 현황들을 정리해 보겠습니다.

<br>

- **목차**

<br>

- ### torchsummary
- ### visdom

<br>

## **torchsummary**

<br>

- 먼저 네트워크의 현황을 쉽게 알아보기 위한 방법으로 `torchsummary` 기능을 사용하는 것입니다.

<br>

```python
from torchvision import models
from torchsummary import summary

vgg = models.vgg16()
summary(vgg, (3, 224, 224)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
         MaxPool2d-5         [-1, 64, 112, 112]               0
            Conv2d-6        [-1, 128, 112, 112]          73,856
              ReLU-7        [-1, 128, 112, 112]               0
            Conv2d-8        [-1, 128, 112, 112]         147,584
              ReLU-9        [-1, 128, 112, 112]               0
        MaxPool2d-10          [-1, 128, 56, 56]               0
           Conv2d-11          [-1, 256, 56, 56]         295,168
             ReLU-12          [-1, 256, 56, 56]               0
           Conv2d-13          [-1, 256, 56, 56]         590,080
             ReLU-14          [-1, 256, 56, 56]               0
           Conv2d-15          [-1, 256, 56, 56]         590,080
             ReLU-16          [-1, 256, 56, 56]               0
        MaxPool2d-17          [-1, 256, 28, 28]               0
           Conv2d-18          [-1, 512, 28, 28]       1,180,160
             ReLU-19          [-1, 512, 28, 28]               0
           Conv2d-20          [-1, 512, 28, 28]       2,359,808
             ReLU-21          [-1, 512, 28, 28]               0
           Conv2d-22          [-1, 512, 28, 28]       2,359,808
             ReLU-23          [-1, 512, 28, 28]               0
        MaxPool2d-24          [-1, 512, 14, 14]               0
           Conv2d-25          [-1, 512, 14, 14]       2,359,808
             ReLU-26          [-1, 512, 14, 14]               0
           Conv2d-27          [-1, 512, 14, 14]       2,359,808
             ReLU-28          [-1, 512, 14, 14]               0
           Conv2d-29          [-1, 512, 14, 14]       2,359,808
             ReLU-30          [-1, 512, 14, 14]               0
        MaxPool2d-31            [-1, 512, 7, 7]               0
           Linear-32                 [-1, 4096]     102,764,544
             ReLU-33                 [-1, 4096]               0
          Dropout-34                 [-1, 4096]               0
           Linear-35                 [-1, 4096]      16,781,312
             ReLU-36                 [-1, 4096]               0
          Dropout-37                 [-1, 4096]               0
           Linear-38                 [-1, 1000]       4,097,000
================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.59
Params size (MB): 527.79
Estimated Total Size (MB): 746.96
----------------------------------------------------------------
```

<br>

## **visdom**

<br>

- 참조 : https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/02_Regression%26NN/Visdom_Tutorial.ipynb
- `visdom`은 딥러닝 학습할 때, 학습 상황을 모니터링 하기 위한 기능 중에 하나입니다. 유사한 기능으로는 `tensorboard`가 있습니다.
- 이번 글에서는 `visdom`을 사용하여 학습 상황을 모니터링하는 방법에 대하여 간략하게 알아보도록 하겠습니다.
- 먼저 visdom을 실행하려면 커맨드 창에서 서버를 하나 띄워야 합니다. 서버를 띄우기 위해 다음 명령어를 실행합니다.
    - `python -m visdom.server`
- 그러면 커맨드 창에 `You can navigate to http://localhost:8097`와 같은 형태의 로그가 남게 됩니다. 
- 위 주소를 브라우저에 입력하면 아래와 같은 `visdom` 환경을 사용할 수 있습니다.

<br>
<center><img src="../assets/img/dl/pytorch/observe/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

```python
from visdom import Visdom

viz = Visdom()
textwindow = viz.text("Hello")
```

<br>

- 코드를 입력 가능한 환경에서 위와 같이 코드를 입력 한다면 브라우저에서 "Hello"라는 문자열이 출력된 것을 볼 수 있습니다.

<br>

```python
image_window = viz.image(
    np.random.rand(3,256,256),
    opts=dict(
        title = "random",
        caption = "random noise"
    )
)
```

<br>
<center><img src="../assets/img/dl/pytorch/observe/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 코드를 입력하면 이미지 형태로 보여줄 수 있습니다.

<br>

```python
images_window = viz.images(
    np.random.rand(10,3,64,64),
    opts=dict(
        title = "random",
        caption = "random noise"
    )
)
```

<br>
<center><img src="../assets/img/dl/pytorch/observe/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 여기서 3채널 이상의 채널을 입력하였을 때에는 위와 같이 펼쳐 10개의 RGB 이미지를 펼쳐주는 것을 볼 수 있습니다.

<br>
<center><img src="../assets/img/dl/pytorch/observe/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다양한 그래프를 깔끔하게 그려줄 수도 있습니다. 이 기능들은 `plot.ly`를 이용한 것입니다.

<br>

- 딥러닝 학습을 할 때, 학습의 변화를 관측하기 위하여 tensorboard나 visdom을 사용합니다.
- 그러면 주 목적인 측정 지표를 어떻게 실시간으로 업데이트 할 지 살펴보겠습니다.
- 먼저 측정한 지표는 `line` 형태로 주로 나타내기 때문에 아래와 같이 사용할 수 있습니다.

<br>

```python
plot = viz.line(
    X = np.array([0, 1, 2, 3, 4]),
    Y = torch.randn(5),
)  
```

<br>
<center><img src="../assets/img/dl/pytorch/observe/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위와 같이 코드를 입력하면 `visdom`에 line이 그려집니다.
- 코드를 해석하면 X축의 값과 Y축의 값이 대입되어 있는 것을 볼 수 있습니다.
- 여기서 중요한 것은 `viz.line()`을 plot으로 받았다는 것입니다.

<br>

```python
viz.line(X = np.array([5]), Y = torch.randn(1), win = plot, update = 'append')
```

<br>
<center><img src="../assets/img/dl/pytorch/observe/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 코드를 보면 `plot`에 `append` 방식으로 (X, Y)를 업데이틑 하는 것을 뜻합니다.
- 위 코드를 응용하면 **매 번의 epoch 마다 Accuracy, Precision, Recall 등의 지표가 어떻게 변하는 지 실시간으로 관찰할 수 있습니다.

<br>

- 다음은 실제로 사용할 수 있는 예제를 살펴보겠습니다.

<br>

```python
viz.line(
    X = np.column_stack((np.arange(0, 10), np.arange(0, 10))),
    Y = torch.randn(10, 2),
    opts = dict(
        title = "Test",
        legend = ["1번 라인", "2번 라인"],
        showlegend = True
    )
)
```

<br>
<center><img src="../assets/img/dl/pytorch/observe/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>