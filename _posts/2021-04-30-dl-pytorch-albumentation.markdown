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
- 참조 : https://youtu.be/rAdLwKJBvPM?list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz
- 참조 : https://github.com/albumentations-team/albumentations
- 참조 : https://albumentations.ai/docs/
- 참조 : https://hoya012.github.io/blog/albumentation_tutorial/

<br>

## **목차**

<br>

- ### [albumentation 설치](#albumentation-설치-1)
- ### [albumentation 이란](#albumentation-이란-1)
- ### [albumentation의 기본적인 사용 방법](#albumentation의-기본적인-사용-방법-1)
- ### [albumentation의 pytorch에서의 사용 방법](#albumentation의-pytorch에서의-사용-방법-1)
- ### [albumentation 사용 시 tip](#albumentation-사용-시-tip-1)
- ### [멀티 데이터 셋의 albumentation 적용](#멀티-데이터-셋의-albumentation-적용-1)

<br>

- ## **자주 사용하는 이미지 segmentation augmentation 리스트**

<br>

- ### [Normalize](#normalize-1)
- ### [RandomResizedCrop](#randomresizedcrop-1)
- ### [RandomRotate90 & Resize](#randomrotate90--resize-1)

<br>

- ### [ColorJitter](#colorjitter-1)

<br>

- ### [GaussNoise](#gaussnoise-1)
- ### [ISONoise](#isonoise-1)
- ### [GaussianBlur](#gaussianblur-1)
- ### [MotionBlur](#motionblur-1)
- ### [ImageCompression](#imagecompression-1)

<br>

- ### [RandomFog](#randomfog-1)
- ### [RandomGamma](#randomgamma-1)
- ### [RandomRain](#randomrain-1)
- ### [RandomShadow](#randomshadow-1)
- ### [RandomSnow](#randomsnow-1)
- ### [RandomSunFlare](#randomsunflare-1)

<br>

- ### [Flip](#flip-1)
- ### [GridDistortion](#griddistortion-1)
- ### [Perspective](#perspective-1)

<br>

## **albumentation 설치**

<br>

- albumentation은 python 3.6 버전 이상을 사용하여야 하며 `pip`를 통하여 다음과 같이 설치 하기를 권장합니다. `-U`는 최신 버전을 받기 위하여 지정하였습니다.
    - 명령어 : `pip install -U albumentations`
- albumentation을 위한 전체 document는 아래 링크를 참조하시면 됩니다.
    - 링크 : [https://albumentations.ai/docs/](https://albumentations.ai/docs/)
- albumentation은 pytorch의 `DataSet`과 `DataLoader`의 역할을 합니다. 따라서 이 2가지 기능에 대해서는 반드시 숙지하시길 바랍니다.
    - 관련 링크 : [https://gaussian37.github.io/dl-pytorch-dataset-and-dataloader/](https://gaussian37.github.io/dl-pytorch-dataset-and-dataloader/)

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

- 위 수치는 [벤치마크](https://github.com/albumentations-team/albumentations/blob/master/benchmark/README.md)를 이용하여  Intel Xeon E5-2650 v4 CPU에서 초당 얼만큼의 이미지를 처리할 수 있는 지 나타냅니다. 초당 처리할 수 있는 이미지의 양이기 때문에 **숫자가 클 수록 성능이 더 좋다**고 말할 수 있습니다.
- albumentation은 다른 패키지에서 제공하지 않는 augmentation 방법을 지원할 뿐 아니라 같은 방법이라도 초당 더 많은 augmentation 성능을 확보하였습니다.

<br>

## **albumentation의 기본적인 사용 방법**

<br>

- 이번 글에서는 albumentation을 사용하는 방법과 pytorch에서는 어떻게 사용하는 지 알아보고 자주 사용하는 이미지 augmentation에 대하여 하나씩 살펴보겠습니다. 특히 `semantic segmentation task`를 기준으로 글을 작성할 예정임 처리 순서는 다음과 같습니다.

- ① `opencv`를 이용하여 이미지와 라벨을 불러 옵니다. (필요 시, BGR → RGB로 변환합니다.)
- ② `transform = A.Compose([])`을 이용하여 이미지와 라벨 각각에 Augmentation을 적용하기 위한 객체를 생성합니다.
- ③ `augmentations = transform(image=image, mask=mask)`를 이용하여 실제 Augmentation을 적용합니다.
- ④ `augmentation_img = augmentations["image"]`를 이용하여 Augmentation된 이미지를 얻을 수 있습니다.
- ⑤ `augmentation_mask = augmentations["mask"]`를 이용하여 Augmentation된 마스크를 얻을 수 있습니다.

<br>

- 아래 코드에서 사용된 샘플 이미지는 다음 링크에서 받을 수 있습니다.
    - [baidu_img.png 다운로드](https://github.com/gaussian37/gaussian37.github.io/blob/master/assets/img/dl/pytorch/albumentation/city_image.png)
    - [baidu_mask.png 다운로드](https://github.com/gaussian37/gaussian37.github.io/blob/master/assets/img/dl/pytorch/albumentation/city_mask.png)

<br>

```python
import albumentations as A
import cv2

image = cv2.imread("city_image.png")
mask = cv2.imread("city_mask.png")

height = 150
width = 300

# Declare an augmentation pipeline
transform = A.Compose([
    A.Resize(height=height, width=width),
    A.RandomResizedCrop(height=height, width=width, scale=(0.3, 1.0)),
])

augmentations = transform(image=image, mask=mask)
augmentation_img = augmentations["image"]
augmentation_mask = augmentations["mask"]

cv2.imwrite("city_image_augmented.png", augmentation_img)
cv2.imwrite("city_mask_augmented.png", augmentation_mask)
```

<br>

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 코드에서 사용된 `RandomResizedCrop`을 통하여 원본 영상이 resize 된 것을 알 수 있습니다.
- segmentation에서 사용되는 `mask`는 각 픽셀 별 클래스 정보를 나타내므로 1 채널 8bit 이미지이지만 시각화를 위하여 컬러 이미지를 사용하였습니다.
- 여기서 핵심은 `augmentations = transform(image=image, mask=mask)` 입니다. `image`와 `mask`에 해당하는 데이터를 단순히 `transform`의 `image`와 `mask`라는 파라미터로 넘겨주기만 하면 **segmentation 목적에 맞게 data가 변형됩니다.** 이 점이 바로 `albumentation`의 핵심이라고 말할 수 있습니다.
- 기본적으로 `image`는 `bilinear interpolation`이 적용되어 이미지의 사이즈가 변형이 되고 `mask`는 `nearest`가 적용이 되어 이미지의 사이즈가 변형이 됩니다.

<br>

- `albumentation`에서 다루고 있는 대표적인 Task는 `segmentation`, `2d detection`, `keypoints estimation`이며 [PyConBy 20](https://youtu.be/c3h7kNHpXw4)에서 설명한 슬라이드를 통해 간단히 사용법을 살펴보면 아래와 같습니다.

<br>

- 먼저 `segmentation`의 경우 앞에서 설명한 것과 같으며 아래 코드에서의 핵심은 `transform(image=image, mask=mask)` 부분이 됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다음으로 `2d detection`의 경우 먼저 `transform`을 선언할 때, 어떤 형식으로 좌표값을 입력하였는 지 정해주어야 합니다. 따라서 아래 코드와 같이 `transform` 선언 시 `bbox_params={'format', : 'coco'}`와 같은 형태로 어떤 오픈 데이터셋의 포맷을 따랐는 지 정해줍니다.
- `2d detection`의 경우 `transform(image=image, bboxes=bboxes)`과 같이 `bboxes`라는 옵션을 사용하면 augmentation 방법에 따라 자동적으로 bbox의 값이 변경됩니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `keypoints estimation`의 경우 `2d detection` 사례와 사용 방법이 유사합니다. `transform` 선언 시 `keypoint_params={'format' : 'xy'}`과 같이 데이터셋의 포맷을 입력하고 `transform(image=image, keypoints=keypoints)`와 같이 `keypoints`라는 옵션을 사용하여 augmentation 방법에 따라 자동적으로 keypoint값이 변경되도록 설정해줍니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `albumentation`의 다양한 augmentation 기법은 공식 문서를 통해 확인이 가능하며 대략적으로 아래와 같은 범주로 augmentation을 지원합니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 다양한 augmentation 방법을 `A.Compose()`로 묶어서 `transform`을 생성합니다. 여러가지 방법을 조합하는 방법은 다음과 같습니다.

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 예시에서 다양한 augmentation 방법이 List 형식으로 차례대로 이어져 있습니다.
- 여기서 `A.OneOf`라는 방법을 살펴보면 `A.OneOf([...])`의 List에 있는 방법 중 하나가 적용된다는 의미입니다. 이 때, 각 augmentation의 발생 확률을 상세하게 정할 수 있어 어떤 augmentation이 좀 더 잘 선택 될수 있도록 할 수도 있습니다. 만약 `A.OneOf([...], p)`와 같이 `A.OneOf` 자체에 `p` 값을 입력하면 `A.OneOf` 자체가 적용될 확률 또한 적용할 수 있습니다.

<br>

## **albumentation의 pytorch에서의 사용 방법**

<br>

- 링크 : [https://albumentations.ai/docs/api_reference/pytorch/transforms/#albumentations.pytorch.transforms.ToTensorV2](https://albumentations.ai/docs/api_reference/pytorch/transforms/#albumentations.pytorch.transforms.ToTensorV2)

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `albumentation`을 pytorch에서 사용하려면 augmentation이 적용된 데이터 타입을 `torch`로 변환을 해야 합니다.
- `from albumentations.pytorch import ToTensorV2`를 import 한 다음에 `ToTensorV2`를 `A.Compose` 마지막에 추가하면 augmentation이 적용된 데이터를 `torch` 타입으로 변환할 수 있습니다.
- 아래 2가지 예시를 통하여 `ToTensorV2`를 적용한 것과 아닌 것의 차이점을 살펴보겠습니다.

<br>

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

image = cv2.imread("city_image.png")
mask = cv2.imread("city_mask.png")

height = 150
width = 300

# Declare an augmentation pipeline
transform = A.Compose([
    A.Normalize(),
    A.Resize(height=height, width=width),
    A.RandomResizedCrop(height=height, width=width, scale=(0.3, 1.0)),
#     ToTensorV2()
])

augmentations = transform(image=image, mask=mask)
augmentation_img = augmentations["image"]
augmentation_mask = augmentations["mask"]

print(augmentation_img.shape)
# (150, 300, 3)

print(type(augmentation_img))
# <class 'numpy.ndarray'>
```

<br>

- 위 예시에서 `image`는 opencv를 통하여 입력되었으므로 numpy 형태이고 이 값을 augmentation 하더라도 그대로 numpy 값을 가집니다. 
- opencv로 이미지를 읽었을 때, (H, W, C) 순서의 shape을 가지는 것 또한 확인할 수 있습니다.

<br>

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

image = cv2.imread("city_image.png")
mask = cv2.imread("city_mask.png")

height = 150
width = 300

# Declare an augmentation pipeline
transform = A.Compose([
    A.Normalize(),
    A.Resize(height=height, width=width),
    A.RandomResizedCrop(height=height, width=width, scale=(0.3, 1.0)),
    ToTensorV2()
])

augmentations = transform(image=image, mask=mask)
augmentation_img = augmentations["image"]
augmentation_mask = augmentations["mask"]

print(augmentation_img.shape)
# torch.Size([3, 150, 300])

print(type(augmentation_img))
# <class 'torch.Tensor'>
```

<br>

- `A.Compose`의 마지막에 `ToTensorV2()`를 추가하였을 때, shape과 class가 모두 `torch` 타입으로 바뀌어져 있는 것을 알 수 있습니다. shape의 순서가 (H, W, C) → (C, H, W) 가 되고 class도 `torch.Tensor`가 되었습니다.

<br>

## **albumentation 사용 시 tip**

<br>

- ① 멀티 GPU를 사용한다면 다음 2가지의 `opencv` 설정을 해주면 효과적입니다.
    - `cv2.setNumThreads(0)`
    - `cv2.ocl.setUseOpenCL(False)`
    - 이유는 다음과 같습니다. In some systems, in the multiple GPU regime, PyTorch may deadlock the DataLoader if OpenCV was compiled with OpenCL optimizations. Adding the following two lines before the library import may help. For more details https://github.com/pytorch/pytorch/issues/1355
- ② transform을 한 이후에 `image`, `label`의 값이 유효한 지 한번 더 체크하는 것이 좋습니다. 저의 경우 transform 이후 label이 이상한 값을 가지게 되어 Loss 에서 에러가 발생하는 경우가 있었습니다. 아래와 같이 해결하였습니다.

<br>
    
```python
augmentations = transform(image=image, mask=mask)
image = augmentations['image']
mask = augmentations['mask']

no_use_class = 99
mask[mask >= num_class] = no_use_class
mask[mask < 0] = no_use_class
```

<br>

## **멀티 데이터 셋의 albumentation 적용**

<br>




<br>

## **자주 사용하는 이미지 augmentation 리스트**

<br>

## **Normalize**

<br>

- 링크 : [https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize)

<br>
<center><img src="../assets/img/dl/pytorch/albumentation/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `Normalize`는 입력 받은 이미지 값의 범위를 (0, 255) → (-1, 1) 범위로 줄여주는 역할을 합니다. 이와 같이 하는 이유는 입력 값의 범위를 줄여줌으로써 학습이 빨리 수렴되게 하고 특정 입력값이 커짐으로써 특정 weight값이 커지는 문제를 개선할 수 있기 때문입니다.
- `Normalize`를 할 때, `mean,` `std` 값이 필요하며 albumentation에서는 추가적으로 `max_pixel_value`라는 값을 필요로 합니다. 그 이유는 `Normalize`하는 식이 다음과 같기 때문입니다.

<br>

- `img = (img - mean * max_pixel_value) / (std * max_pixel_value)`

<br>

- 보통 `mean`, `std` 값을 (0, 255) 사이의 값으로 저장하거나 (-1, 1) 사이의 값으로 저장합니다. 이 때, `max_pixel_value` 값은 다음과 같이 설정합니다. 참고로 `mean`, `std`는 리스트, 튜플 또는 넘파이 배열로 입력하면 됩니다.

<br>

- ① `mean`, `std` 값의 범위 (0, 255) : `max_pixel_value = 1.0`으로 설정 합니다. 

<br>

```python
mean1 = [90, 100, 100]
std1 = [30, 32, 28]

transform = A.Compose([
    A.Normalize(mean=mean1, std=std1, max_pixel_value=1.0),
])
```

<br>

- ②`mean`, `std` 값의 범위 (-1, 1) : `max_pixel_value = 255`으로 설정

<br>

```python
mean2 = [mean1[0]/255, mean1[1]/255, mean1[2]/255]
std2 = [std1[0]/255, std1[1]/255, std1[2]/255]

transform = A.Compose([
    A.Normalize(mean=mean2, std=std2, max_pixel_value=255),
])
```

<br>

- 위와 같이 `max_pixel_value`를 설정해야 정확한 Normalization 수식 `(변량 - 평균) / 표준편차`를 적용할 수 있습니다.

<br>

## **RandomResizedCrop**

<br>

<br>

## **RandomRotate90 & Resize**

<br>

<br>

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

[pytorch 관련 글 목차](https://gaussian37.github.io/dl-pytorch-table/)

<br>
