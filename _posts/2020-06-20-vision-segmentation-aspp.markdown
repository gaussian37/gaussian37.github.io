---
layout: post
title: ASPP(Atrous Spatial Pyramid Pooling)
date: 2020-06-20 00:00:00
img: vision/segmentation/aspp/0.png
categories: [vision-segmentation] 
tags: [vision, deep learning, segmentation, aspp, atrous, spatial, pyramid, pooling] # add tag
---

<br>

[segmentation 글 목록](https://gaussian37.github.io/vision-segmentation-table/)

<br>

- 참조 : https://towardsdatascience.com/review-deeplabv1-deeplabv2-atrous-convolution-semantic-segmentation-b51c5fbde92d
- 참조 : https://m.blog.naver.com/laonple/221000648527
- 이번 글에서는 segmentation에서 자주 사용되는 `ASPP(Atrous Spatial Pyramid Pooling)`에 대하여 다루어 보도록 하겠습니다.
- `ASPP`는 [DeepLab_v2](https://arxiv.org/abs/1606.00915)에서 소개되었고 그 이후에 많은 Segmentation 모델에서 차용해서 사용하고 있습니다.

<br>

## **목차**

- ### Atrous convolution
- ### ASPP(Atrous Spatial Pyramid Pooling)
- ### Pytorch 코드

<br>

## **Atrous convolution**

<br>

- Atrous convolution 또는 dilated convolution에 대한 내용은 제 블로그의 다음 링크를 참조해 주시기 바랍니다. 상당히 자세하게 다루어 놓았습니다.
- 링크 : https://gaussian37.github.io/dl-concept-dilated_residual_network/

<br>

- Atrous convolution에 대한 자세한 내용은 위 링크를 참조하시고 이 글을 전개하기 위해 간단하게 살펴보도록 하겠습니다.
- 1-dimension의 Atrous convolution을 수식으로 나타내면 다음과 같습니다.

<br>

$$ y[i] = \sum_{k=1}^{K} x[i + r \cdot k]w[k] $$

$$ r > 1 \text{ : atrous convolution}, \quad r = 1 \text{ : standard convolution} $$

<br>

- 위 수식에서 $$ x $$가 input이고 $$ w $$가 filter입니다. 즉, $$ r $$의 값에 따라 input을 얼마나 띄엄 띄엄 filter와 곱 연산을 할 지가 결정됩니다.

<br>
<center><img src="../assets/img/vision/segmentation/aspp/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이를 이미지 convolution filter에 적용하면 위 그림과 같습니다. 그림의 위쪽 그림이 standard convolution인 반면 아래쪽 그림이 atrous convolution을 적용한 형태입니다.
- 같은 크기의 kernel을 사용하였음에도 불구하고 atrous convolution을 적용하였을 때, 더 넓은 범위의 input feature를 cover 할 수 있습니다.
- 즉, **atrous convolution은 input feature의 FOV(Field Of View)를 더 넓게 확장 할 수 있는** 장점을 가집니다.


<br>

## **ASPP(Atrous Spatial Pyramid Pooling)**

<br>

- ASPP에 대하여 제 블로그의 다음 링크도 참조하시기 바랍니다.
- 링크 : https://gaussian37.github.io/vision-segmentation-deeplabv3plus/

<br>
<center><img src="../assets/img/vision/segmentation/aspp/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `ASPP`가 소개된 Deeplab v2에서는 multi-scale에 더 잘 대응할 수 있도록 atrous convolution에 대한 `확장 계수`를 (6, 12, 18, 24)를 적용하여 위 그림과 같이 합쳐서 사용하였습니다.
- Spatial Pyramid Pooling 구조는 [SPPNet](https://gaussian37.github.io/vision-segmentation-sppnet/)을 통하여 얻을 수 있었고 이 구조에 atrous convolution을 적용하여 ASPP를 만들었습니다.
- 처음 제시되었던 ASPP는 `확장 계수`를 6 ~ 24 까지 다양하게 변화하면서 다양한 receptive field를 볼 수 있도록 적용하였습니다.

<br>
<center><img src="../assets/img/vision/segmentation/aspp/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 왼쪽 그림은 Deeplab v1의 모델의 형태이고 오른쪽 그림은 Deeplab v2의 모델 형태입니다.
- 양쪽 모두 FC6 layer 까지는 동일한 아키텍쳐를 이용하고 있습니다. 하지만 FC6에서 부터 두 모델이 달라집니다.
- 오른쪽 그림의 Deeplab v2의 경우 ASPP 구조에 확장 계수 6, 12, 18, 24가 적용되어 다양한 스케일을 볼 수 있도록 설계되어 있습니다.
- 반면 왼쪽 그림의 Deeplab v1의 경우 ASPP 구조와 비교해서 보면 단순히 확장 계수가 12로 고정되어 있다고 생각하면 됩니다.

<br>
<center><img src="../assets/img/vision/segmentation/aspp/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 성능 지표를 통해 고정된 확장 계수의 ASPP인 `Large FOV`와 `ASPP-S` (확장 계수 값이 작은 값들로 구성됨 r = 2, 4, 8, 12) 그리고 `ASPP-L` (r = 6, 12, 18, 24)를 이용하였을 때의 성능을 비교할 수 있습니다.
- 그 결과 **넓은 multi scale의 receptive field > 졻은 multi scale의 receptive field > 고정된 scale의 receptive field**인 것을 확인할 수 있습니다.
- 따라서 `ASPP`를 적용할 때 확장 계수는 `넓은 multi scale의 receptive field`를 사용할 수 있도록 하는 것이 성능을 높일 수 있습니다.

<br>

## **Pytorch 코드**

<br>

```python






<br>

[segmentation 글 목록](https://gaussian37.github.io/vision-segmentation-table/)

<br>