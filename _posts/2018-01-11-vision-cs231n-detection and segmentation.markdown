---
layout: post
title: 11. Detection and Segmentation
date: 2018-01-11 01:00:00
img: vision/cs231n/cs231n.jpg
categories: [vision-cs231n] 
tags: [cs231n, detection, segmentation] # add tag
---

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/nDPWywWRIRo" frameborder="0" allowfullscreen="true" width="600px" height="400px"> </iframe>
</div>
<br>

<img src="../assets/img/vision/cs231n/11/1.png" alt="Drawing" style="width: 800px;"/>

+ 이 때까지의 수업에서는 주로 Image Classification 문제를 다루었습니다.
+ 입력 이미지가 들어오면 Deep ConvNet을 통과하고 네트워크를 통과하면 Feature Vector가 나옵니다.
    + AlexNet이나 VGG의 경우에는 4096차원의 Feature Vector가 생성되었습니다.
+ 그리고 최종 Fully Connected Layer에서는 1000개의 클래스 스코어를 나타냅니다.
    + 이 예제에서 1000개의 클래스는 ImageNet의 클래스를 의미합니다.
+ 즉, 전체 구조는 입력 이미지가 들어오면 전체 이미지가 속하는 카테고리의 출력입니다.
    + 위 과정은 가장 기본적인 Image Classification이고 Deep Learning으로 더 흥미로운 작업이 가능합니다.
+ 이번 강의에는 Deep Learning의 다양한 Task들에 대하여 알아보도록 하겠습니다.
     
<img src="../assets/img/vision/cs231n/11/2.png" alt="Drawing" style="width: 800px;"/>

+ 이번 강의에서 배울 내용은 크게 4가지 입니다.
    + Semantic Segmentation
    + Classification + Localization
    + Object Dectection
    + Instance Segmentation

<br>

## Semantic Segmentation

<img src="../assets/img/vision/cs231n/11/3.png" alt="Drawing" style="width: 800px;"/>

+ Semantic Segmentation 문제에서는 입력은 이미지이고 출력으로 이미지의 모든 픽셀에 카테고리를 정합니다.
    + 예를 들어 위 슬라이드의 예제를 보면 입력은 고양이 입니다. 
    + 출력은 모든 픽셀에 대하여 그 픽셀이 고양이, 잔디, 하늘, 나무, 배경인지를 결정합니다.
+ Semantic Segmentation에서도 Classification 처럼 카테고리가 있습니다.
+ 하지만 차이점은 Classification처럼 이미지 전체에 카테고리 하나가 아니라 모든 픽셀에 카테고리가 매겨집니다.


<img src="../assets/img/vision/cs231n/11/4.png" alt="Drawing" style="width: 800px;"/>

+ Semantic Segmentation은 개별 객체를 구별하지 않습니다.
+ 위 슬라이드의 Semantic Segmentation 결과를 보면 픽셀의 카테고리만 구분해 줍니다.
    + 즉 오른쪽 슬라이드의 결과를 보면 소가 2마리가 있는데 2마리 각각을 구분할 수는 없습니다.
+ 이것은 Semantic Segmentation의 단점이고 나중에 배울 Instance Segmentation에서 이 문제를 해결할 수 있습니다.

<img src="../assets/img/vision/cs231n/11/5.png" alt="Drawing" style="width: 800px;"/>

+ Semantic Segmentation 문제에 접근해 볼 수 있는 방법 중 하나는 Classification을 통한 접근 방법입니다.
+ Semantic Segmentation을 위해서 `Sliding Window`를 적용해 볼 수 있습니다.   
+ 먼저 입력 이미지를 아주 작은 단위로 쪼갭니다.
+ 위 슬라이드의 예제에서는 소의 머리 주변에서 영역 세개를 추출하였습니다. 이 작은 영역만을 가지고 classification 문제를 푼다고 가정하여 해당 영역이 어떤 카테고리에 속하는지 알아보겠습니다.
+ 이미지 한장을 분류하기 위해서 만든 모델을 이용해서 이미지의 작은 영역을 분류하게 할 수 있을 것입니다.
    + 이 방법이 어느정도는 동작할 지 모르겠지만 그렇게 좋은 방법은 아닙니다.
    + 왜냐하면 계산 비용이 굉장히 많이 들기 때문입니다.
+ 모든 픽셀에 대하서 작은 영역으로 쪼개고, 이 모든 영역을 forward/backward pass 하는 일은 상당히 비효율적입니다.
+ 그리고 서로 다른 영역이라도 인접해 있으면, 어느 정도 겹쳐져 있기 때문에 특징들을 공유할 수도 있을 것입니다.
    + 이렇게 영역을 분할하는 경우에도 영역들끼리 공유할만한 특징들이 아주 많을 것입니다.
    + 하지만 각 픽셀에 대하여 작은 영역으로 쪼개는 방법은 주변의 overlap 되는 부분에서 feature들을 공유해서 사용할 수 없기 때문에 상당히 비효율적입니다.
+ 따라서 각 픽셀마다 개별적으로 classification을 적용하는 방법은 아주 나쁜 방법입니다.
+ 하지만 semantic segmentation을 할 때 가장 쉽게 생각할 수 있는 아이디어 이기도 합니다.

<img src="../assets/img/vision/cs231n/11/6.png" alt="Drawing" style="width: 800px;"/>

+ 슬라이딩 윈도우 방법보다 조금 더 개선된 방법으로 `Fully Convolutional Network`가 있습니다.
+ 이 방법은 이미지 영역을 나누고 독립적으로 분류하는 방법과는 다릅니다. FC layer가 없고 Convolution layer로 구성된 네트워크를 상상해 볼 수 있습니다.
+ **3 x 3 zero padding**을 수행하는 Convolution Layer들을 쌓아올리면 이미지의 공간 정보를 손실하지 않을 수 있습니다.
+ 이 네트워크의 출력 Tensor는 `C x H x W`입니다. 여기서 `C`는 카테고리의 수 입니다.
+ 이 출력 Tensor는 입력 이미지의 모든 픽셀 값에 대하여 classification score를 매긴 값입니다.
    + 이는 Convolution Layer만 쌓아올린 네트워크를 이용해서 계산할 수 있습니다.
+ 이 네트워크를 학습시키려면 먼저 모든 픽셀의 `classification loss`를 계산하고 평균 값을 취합니다.
    + 그리고 기존처럼 back propagation을 수행하면 됩니다.

<br>

+ 이러한 방법을 사용하려면 training data을 만들 때 각 픽셀마다 classification을 한 데이터가 있어야 합니다.
    + 즉, training data를 만들 때에도 상당히 큰 비용이 발생하게 됩니다.
+ 또한 이러한 segmentaion 문제에서는 Classification을 하는 것입니다. 따라서 출력의 모든 픽셀에 대하여 Cross Entropy를 적용해야 합니다.
    + 따라서 출력의 모든 픽셀에는 Ground Truth가 반드시 존재합니다.
    + 출력의 모든 픽셀과 Ground Truth 간의 Cross Entropy를 계산합니다.
    + 이 때 계산한 값들을 모두 더하거나 평균화시켜서 `Loss`를 계산합니다. 또는 Mini-batch 단위로 계산할 수도 있습니다.
+ 따라서 우리는 모든 픽셀의 카테고리를 알고 있어야 데이터셋을 만들 수 있고 학습할 수도 있습니다.
    + 즉, Image Classification에서도 클래스의 수가 정해져 있고 이미지가 어떤 클래스에 들어가는 지 알 수 있습니다.
    + Semantic Segmentation에서도 클래스의 수가 고정되어 있습니다. 
+ 이 모델은 하이퍼파라미터만 잘 조절해서 학습시켜주면 비교젹 잘 동작합니다. 
+ 하지만 이 모델의 단점은 입력 이미지의 Spatial size를 계속 유지시켜주어야 합니다. 그래서 비용이 아주 큽니다.
+ 예를 들어 Convolution의 채널이 64/128/256일 수 있습니다. 이 정도는 보통 네트워크 크기에 해당하는 흔한 케이스 입니다.
    + 만약 이 네트워크에 고해상도 이미지가 입력으로 들어오게 되면 계산량과 메모리가 상당히 커져서 감당하기 어려운 계산량이 될 수 있습니다.
    
<img src="../assets/img/vision/cs231n/11/7.png" alt="Drawing" style="width: 800px;"/>
    
+ 따라서 `Spatial size`를 유지해야 하는 구조 대신에 위 슬라이드와 같은 구조를 사용하는 것이 대부분입니다.
    + 위 슬라이드를 보면 feature map을 downsampling/upsampling 을 합니다.
    + 즉, spatial resolution 전체를 가지고 Convolution을 수행하기 보다는 original resolution에서는 conv layer는 소량만 사용합니다.
    + 그리고 `max pooling`, `stride convolution` 등으로 특징맵을 downsample 합니다.
+ 위 구조는 classification network와 구조가 유사하게 보입니다. 하지만 차이점이 있습니다.
    + Image classification에서는 FC layer가 있었습니다. 하지만 위 슬라이드 방법은 Spatial resolution을 다시 키웁니다.
    + 이 방법으로 결국에는 다시 입력 이미지의 해상도와 같아집니다.
    + 출력의 해상도는 결국 입력의 해상도와 같지만 `계산 효율`이 좋아지는 장점을 갖습니다. 
    + 이 방법을 통하여 네트워크가 lower resolution을 처리하도록 하여 네트워크를 더 깊게 만들 수 있습니다.

<br>

+ 이전 강의에서 다루었던 내용을 살펴보면 Convolutional Networks에서의 `Downsampling`에 대해서 다루어 본 적이 있을 것입니다.
    + 예를 들어 이미지의 Spatial Size를 줄이기 위한 `stride conv` 또는 다양한 pooling들을 다루어 본 적이 있을 것입니다.
+ 하지만 `upsampling`은 처음일 것입니다. 따라서 upsampling이 네트워크에서 어떻게 동작하는지 궁금하실 수 있습니다.

<img src="../assets/img/vision/cs231n/11/8.png" alt="Drawing" style="width: 800px;"/>

+ Upsampling의 방법 중 하나는 `unpooling` 입니다.
    + Downsampling에서의 pooling에는 average/max pooling이 있습니다.
    + unpooling 방법 중에는 슬라이드 왼쪽과 같이 `nearest neighbor` unpooling 방법이 있습니다.
        + 왼쪽의 nearest neighbor unpooling 예제를 보면 입력은 2 x 2 그리드이고 출력은 4 x 4 그리드입니다.
            + 2x2 stride nearest neighbor unpooling은 해당하는 receptive field 값을 그냥 복사합니다.
    + 또 다른 방법으로는 `bed of nails` unpooling 이라는 것이 있습니다.
        + 이 방법은 unpooling region에만 값을 복사하고 다른 곳에는 모두 0을 채워넣습니다.
        + 이 경우 하나의 요소를 제외하고 모두 0으로 만듭니다. 이 예제에서는 왼쪽 위에만 값이 있고 나머지는 0입니다.
        + bed of nails 라는 표현의 유래는 zero region은 평평하고 non-zero region은 바늘처럼 뾰족하게 값이 튀기 때문입니다.

<img src="../assets/img/vision/cs231n/11/9.png" alt="Drawing" style="width: 800px;"/>

+ 또 다른 Upsampling 방법에는 `Max unpooling`이 있습니다.
    + 대부분의 네트워크는 대칭적인 경향이 있습니다. 예를 들어 Downsampling/Upsampling의 비율이 대칭적입니다. 
         
       
    



  
    


