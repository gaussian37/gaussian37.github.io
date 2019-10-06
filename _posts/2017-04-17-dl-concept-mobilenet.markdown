---
layout: post
title: MobileNets - Efficient Convolutional Neural Networks for Mobile Vision Applications
date: 2017-04-17 00:00:00
img: dl/concept/mobilenet/mobilenet.PNG
categories: [dl-concept] 
tags: [python, deep learning, dl, MobileNet] # add tag
---

- 이번 글에서는 경량화 네트워크로 유명한 `MobileNet`에 대하여 알아보도록 하겠습니다.
    - 이번글은 `MobileNet` 초기 버전(v1) 입니다.
- MobileNet의 핵심 내용인 [`Depthwise Separable Convolution`]()의 내용을 퀵 하게 알고 싶으면 저의 다른 글도 추천 드립니다.
- 아래 글의 내용은 [PR12 모바일넷 설명](https://youtu.be/7UoOFKcyIvM) 내용을 글로 읽을 수 있게 옮겼습니다.

<br>

## **목차**

<br>

- ### 1. 추가 설명 자료
- ### 2. 논문 리뷰
- ### 3. Pytorch 코드 리뷰 

<br>

## **1. 추가 설명 자료**

<br> 


### 경량화 네트워크의 필요성

<br>

- 먼저 딥러닝의 상용화를 위하여 필요한 여러가지 제약 사항을 개선시키기 위하여 경량화 네트워크에 대한 연구가 시작되었습니다.
- 딥러닝을 이용한 상품들이 다양한 환경에서 사용되는데 특히, 고성능 컴퓨터가 아닌 상황에서 가벼운 네트워크가 필요하게 됩니다. 
- 예를 들어 데이터 센터의 서버나 스마트폰, 자율주행자동차 또는 드론과 같이 가격을 무작정 높일 수 없어서 제한된 하드웨어에 딥러닝 어플리케이션이 들어가는 경우입니다.
    - 이러한 경우에 실시간 처리가 될 정도 성능의 뉴럴넷이 필요하고 또한 얼마나 전력을 사용할 지도 고려를 해야합니다.    
- 이러한 제약 사항을 충분히 만족하면서 또한 아래와 같은 성능이 꽤 괜찮아야 어플리케이션에 적용을 할 수 있습니다.
    - 충분히 납득할만한 정확도
    - 낮은 계산 복잡도
    - 저전력 사용
    - 작은 모델 크기

<br>

- 그러면 왜 `Small Deep Neural Network`가 중요하게 되었을까요?
    - 네트워크를 작게 만들면 학습이 빠르게 될것이고 임베디드 환경에서 딥러닝을 구성하기에 더 적합해집니다.
    - 그리고 무선 업데이트로 딥 뉴럴 네트워크를 업데이트 해야한다면 적은 용량으로 빠르게 업데이트 해주어야 업데이트의 신뢰도와 통신 비용등에 도움이 됩니다. 
    
<br>

### Small Deep Neural Network 기법

<br>

- `Channel Reduction` : MobileNet 적용
    - Channel 숫자룰 줄여서 경량화
- `Depthwise Seperable Convolution` : MobileNet 적용
    - 이 컨셉은 `Xception`에서 가져온 컨셉이고 이 방법으로 경량화를 할 수 있습니다.
- `Distillation & Compression` : MobileNet 적용
- Remove Fully-Connected Layers
    - 파라미터의 90% 정도가 FC layer에 분포되어 있는 만큼 FC layer를 제거하면 경량화가 됩니다. 
    - CNN기준으로 필터(커널)들은 파라미터 쉐어링을 해서 다소 파라미터의 갯수가 작지만 FC layer에서는 파라미터 쉐어링을 하지 않기 때문에 엄청나게 많은 수의 파라미터가 존재하게 됩니다. 
- Kernel Reduction (3 x 3 → 1 x 1)
    - (3 x 3) 필터를 (1 x 1) 필터로 줄여서 연산량 또는 파라미터 수를 줄여보는 테크닉 입니다. 
    - 이 기법은 대표적으로 `SqueezeNet`에서 사용되었습니다.
- Evenly Spaced Downsampling
    - Downsampling 하는 시점과 관련되어 있는 기법입니다.
    - Downsampling을 초반에 많이 할 것인지 아니면 후반에 많이할 것인지 선택하게 되는데 그것을 극단적으로 하지 않고 균등하게 하자는 컨셉입니다.
    - 왜냐하면 초반에 Downsampling을 많이하게 되면 네트워크 크기는 줄게 되지만 feature를 많이 잃게 되어 accuracy가 줄어들게 되고
    - 후반에 Downsampling을 많이하게 되면 accuracy 면에서는 전자에 비하여 낫지만 네트워크의 크기가 많이 줄지는 않게 됩니다.
    - 따라서 이것의 절충안으로 적절히 튜닝하면서 Downsampling을 하여 Accuracy와 경량화 두 가지를 모두 획득하자는 것입니다.
- Shuffle Operation

<br>

- 특히 `MobileNet`에서 사용하는 핵심 아이디어는 `Depthwise Seperable Convolution`입니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet/1.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 `MobilNet`을 다루기 전에 간단하게 **Convolution Operation**에 대하여 다루어 보겠습니다.
- 위와 같이 인풋의 채널이 3개이면 convolution 연산을 하는 필터의 채널 또한 3개이어야 합니다.
- 이 때 **필터의 갯수**가 몇 개 인지에 따라서 **아웃풋의 채널의 숫자**가 결정되게 됩니다.
- 정리하면 `인풋의 채널 수 = 필터의 채널 수`이고 `필터의 갯수 = 아웃풀의 채널 수`가 됩니다.
- 이 때, **인풋 채널과 필터의 연산 과정**은 위의 오른쪽 그림과 같이 입력 채널에서는 필터의 크기 만큼 모든 채널의 값들이 **element-wise 곱**으로 연산하여 **최종적으로 한 개의 값**으로 모두 더해지게 됩니다.

<br>

- 먼저 모바일넷 이전의 네트워크인 `VGG`의 네트워크 구조를 살펴보면 어떤 크기의 필터를 사용하는 것이 좋은가에 대한 솔루션을 제공하였습니다.
- 3 x 3 크기의 필터를 여러번 사용하면 5 x 5 나 7 x 7 크기의 필터와 같은 `receptive field`를 가지기 때문에 3 x 3 필터만 쓰면 된다는 것을 제시하였습니다.
- 그리고 3 x 3 필터를 여러번 사용하는 것이 더 큰 필터를 조금 사용하는 것 보다 더 **non-linearity**가 많아지게 되는 장점이 있고 파라미터 수는 오히려 더 작아지게 되므로(ex. 2 x (3 x 3 x C) Vs. (5 x 5 x C)) regularization에서도 우수한 성능이 보입니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet/2.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>
  
- `VGG`의 `3 x 3 필터` 컨셉을 이용하여 `Inception v2, v3`에서는 5 x 5나 7 x 7 필터를 지우고 3 x 3 필터만 사용하게 됩니다.
- 더 나아가서 항상 (h, w) 크기의 필터를 사용해야 하는 점에 대하여 `Inception`에서 아이디어를 제공합니다.
- 행의 방향과 열의 방향으로 필터를 분리하여 필터를 적용하는 방법인데 예를 들어 (3 x 3 x C) 필터를 적용하는 대신 (3 x 1 x C)필터와 (1 X 3 X C) 필터를 적용하는 방법입니다.
- 이렇게 행과 열의 방향으로 필터를 분리하면 파라미터의 갯수를 줄일 수 있습니다. 위의 예에서 정사각형 필터의 파라미터의 수는 9C이지만 분리한 필터의 파라미터 수는 6C가 됩니다. 

<br>

- `VGG`에서는 `필터의 크기`에 대한 고찰이 있었고 `Inception`에서는 `정사각형 형태의 필터`에 대한 고찰이 있었습니다.
- `MobileNet`에서는 **입력 데이터에 필터를 적용할 때, 모든 채널을 한번에 계산해서 아웃풋을 만들어야 하는 것**에 대한 고찰을 합니다.
    - 예를 들어, (100(h) x 100(w) x 3(c)) 이미지가 있고 여기에 (3(h) x 3(w) x 3(c)) 필터를 적용하면 3채널 모두에 필터 연산이 적용 되고 그 값들의 합으로 하나의 스칼라 값이 출력되게 됩니다.
    - `MobileNet`은 왜 3채널 모두에 연산을 다 해야하지? 라는 의문점에서 시작합니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet/3.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 연산을 하는 것이 기존의 Convolution 연산입니다.

<br>
<center><img src="../assets/img/dl/concept/mobilenet/3.PNG" alt="Drawing" style="width: 800px;"/></center>
<br>

- 반면 위와 같이 연산을 하는 것이 `Depthwise Seperable Convolution`입니다.
    - 이 연산은 `Depthwise Convolution`과 `Pointwise Convolution`으로 나뉩니다.
- 왼쪽의 `Depthwise Convolution`을 보면 입력값의 가장 앞쪽 채널인 빨간색 3 x 3 영역만 필터와 연산이 되어 동일한 위치에 스칼라 값으로 출력이 됩니다.
    - 녹색과 파란색 채널도 각각 필터와 연산이 되어 동일한 위치의 채널에 출력값으로 대응됩니다.
- 이 계산 과정을 기존의 convolution과 비교하면 기존의 연산에서는 한번 필터를 적용하면 출력값을 모두 더하여 한 개의 출력값으로 만든 반면 `Depthwise Convolution`에서는 채널의 갯수 만큼의 출력값을 가진다는 것입니다. 즉, 채널 방향으로 합치지 않습니다. 
- 그리고 채널 방향의 연산을 하는 것은 `Pointwise Convolution`에서 합니다. 즉, 1 x 1 convolution을 적용하는 것입니다.

       

 

<br>

## **2. 논문 리뷰**

<br>

## **3. Pytorch 코드 리뷰**

<br>

```python
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

```
