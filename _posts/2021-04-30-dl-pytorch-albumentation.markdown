---
layout: post
title: albumentation을 이용한 이미지 augmentation
date: 2021-04-30 00:00:00
img: dl/pytorch/albumentation/0.png
categories: [dl-pytorch]
tags: [deep learning, augmentation, albumentation, pytorch] # add tag
---

<br>

[pytorch 관련 글 목차](https://gaussian37.github.io/dl-pytorch-table/)

<br>

- 참조 : https://youtu.be/c3h7kNHpXw4
- 참조 : https://github.com/albumentations-team/albumentations
- 참조 : https://albumentations.ai/docs/
- 참조 : https://hoya012.github.io/blog/albumentation_tutorial/

<br>

## **목차**

<br>

- ### [albumentation 이란](#)
- ### [albumentation 설치 및 기본 사용 방법](#)
- ### [pytorch에서의 사용 방법](#)

<br>

- ## **자주 사용하는 이미지 augmentation 리스트**

<br>

- ### [ColorJitter](#)

<br>

- ### [GaussNoise](#)
- ### [ISONoise](#)
- ### [GaussianBlur](#)
- ### [MotionBlur](#)
- ### [ImageCompression](#)

<br>

- ### [RandomFog](#)
- ### [RandomGamma](#)
- ### [RandomRain](#)
- ### [RandomShadow](#)
- ### [RandomSnow](#)
- ### [RandomSunFlare](#)

<br>

- ### [Flip](#)
- ### [GridDistortion](#)
- ### [Perspective](#)

<br>

- ### [RandomResizedCrop](#)
- ### [RandomRotate90 & Resize](#)

<br>

## **albumentation 이란**

<br>

- `albumentation`은 **이미지 augmentation을 쉽고 효율적으로** 할 수 있도록 만든 라이브러리 입니다.
- 딥러닝 기반의 이미지 어플리케이션을 개발할 때에는 제한된 학습 이미지로 인하여 augmentation은 필수적으로 사용하고 있습니다. Pytorch를 기준으로 예시를 들면 torchvision이라는 패키지를 이용하여 다양한 transform을 할 수 있도록 지원하고 있습니다.
- 이번 글에서 설명할 `albumentation`은 torchvision에서 지원하는 transform 보다 더 효율적이면서도 다양한 augmentation 기법을 지원합니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 뿐만 아니라 이미지 기반 딥러닝 어플리케이션 중 대표적인 세그멘테이션(Masks), 2D object detection(BBoxes) 그리고 Keypoints를 자동으로 augmentation에 적용시켜 줍니다.
- 예를 들어 위 예제와 같이 Crop을 하면 Mask 또한 Crop이되고 BBox와 Keypoint 또한 Crop을 반영한 좌표가 적용됩니다. 이 점을 잘 이용하면 굉장히 쉽게 augmentation을 적용할 수 있습니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 수치는 [벤치마크](https://github.com/albumentations-team/albumentations/blob/master/benchmark/README.md)를 이용하여  Intel Xeon E5-2650 v4 CPU에서 초당 얼만큼의 이미지를 처리할 수 있는 지 나타냅니다. 초당 처리할 수 있는 이미지의 양이기 때문에 숫자가 클 수록 성능이 더 좋다고 말할 수 있습니다.
- albumentation은 다른 패키지에서 제공하지 않는 augmentation 방법을 지원할 뿐 아니라 같은 방법이라도 초당 더 많은 augmentation 성능을 확보하였습니다.
- 이번 글에서는 albumentation을 사용하는 방법과 pytorch에서는 어떻게 사용하는 지 알아보고 자주 사용하는 이미지 augmentation에 대하여 하나씩 살펴보겠습니다.

<br>

## **albumentation 설치 및 기본 사용 방법**

<br>

- albumentation은 python 3.6 버전 이상을 사용하여야 하며 `pip`를 통하여 다음과 같이 설치 하기를 권장합니다. `-U`는 최신 버전을 받기 위하여 지정하였습니다.
    - 명령어 : `pip install -U albumentations`
- albumentation을 위한 전체 document는 아래 링크를 참조하시면 됩니다.
    - 링크 : [https://albumentations.ai/docs/](https://albumentations.ai/docs/)
- albumentation은 pytorch의 `DataSet`과 `DataLoader`의 역할을 합니다. 따라서 이 2가지 기능에 대해서는 반드시 숙지하시길 바랍니다.
    - 관련 링크 : [https://gaussian37.github.io/dl-pytorch-dataset-and-dataloader/](https://gaussian37.github.io/dl-pytorch-dataset-and-dataloader/)

<br>

## **pytorch에서의 사용 방법**

<br>



<br>


## **자주 사용하는 이미지 augmentation 리스트**

<br>

## **ColorJitter**

<br>

<br>


<br>

## **GaussNoise**

<br>

<br>

## **ISONoise**

<br>

<br>

## **GaussianBlur**

<br>

<br>

## **MotionBlur**

<br>

<br>

## **ImageCompression**

<br>

<br>


<br>

## **RandomFog**

<br>

<br>

## **RandomGamma**

<br>

<br>

## **RandomRain**

<br>

<br>

## **RandomShadow**

<br>

<br>

## **RandomSnow**

<br>

<br>

## **RandomSunFlare**

<br>

<br>


<br>

## **Flip**

<br>

<br>

## **GridDistortion**

<br>

<br>

## **Perspective**

<br>

<br>


<br>

## **RandomResizedCrop**

<br>

<br>

## **RandomRotate90 & Resize**

<br>

<br>




<br>

[pytorch 관련 글 목차](https://gaussian37.github.io/dl-pytorch-table/)

<br>
