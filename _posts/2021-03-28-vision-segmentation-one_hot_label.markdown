---
layout: post
title: segmentation 학습을 위한 one hot label 생성 및 Loss 연산
date: 2021-03-28 00:00:00
img: vision/segmentation/one_hot_label/0.png
categories: [vision-segmentation] 
tags: [vision, deep learning, segmentation, one hot label] # add tag
---

<br>

## **목치**

<br>

- ### [one hot label 생성 방법](#one-hot-label-생성-방법-1)
- ### [segmentation 모델의 출력과 one hot label로 Loss 구하기](#segmentation-모델의-출력과-one-hot-label로-loss-구하기-1)


<br>

## **one hot label 생성 방법**

<br>

- segmetation을 딥러닝으로 학습할 때, Loss 함수로 Pytorch에서 제공하는 `Cross Entropy Loss`를 많이 사용하곤 합니다.
- `Cross Entropy Loss` 이외에도 `Dice Loss`, `IoU Loss`, `Focal Loss` 등 다양한 Loss 함수가 있으며 이러한 Loss 함수들은 경우에 따라서 Custom Loss Function 따로 작성하여 써야 하는 경우가 종종 발생합니다.
- 이번 글에서 다루는 `one hot label`은 Custom Loss Function을 작성하여 사용할 때, 필요한 부분입니다.

<br>

- segmentation 모델의 입력으로 들어가는 한 장의 이미지는 `(Batch, Channel, Height, Width) = (1, 3, Height, Width)`의 크기를 가집니다.
- 일반적으로 이미지를 읽어들였을 때, `(Channel, Height, Width) = (3, Height, Width)` 또는 `(Height, Width, Channel) = (Height, Width, 3)`와 같이 Height, Width, Channel 3가지 정보를 가지게 됩니다. 이 때, 한번에 학습할 이미지의 양 까지 추가되어 최종적으로는 `(Batch, Channel, Height, Width)`와 같은 형태를 가집니다.

<br>

- 반면 모델의 출력과 비교가 되는 `label`의 경우 여러 개의 Channel을 가지지 않으므로 `Pytorch`와 같은 Framework에서는 `(Batch, Height, Width)`의 형태로 입력 받습니다.
- 따라서 한 장의 `label` 이미지를 입력받는 경우 단순히 2차원 행렬인 `(Height, Width)` 형태로 입력 받도록 작성되어야 합니다.

<br>
<center><img src="../assets/img/vision/segmentation/one_hot_label/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림에서 왼쪽 이미지는 일반적인 `RGB` 이미지 입니다. 이와 같이 segmentation 모델의 입력으로 들어가게 될 이미지는 `(Height, Width, Channel)` 또는 `(Channel, Height, Width)`의 크기를 가집니다.
- 반면 위 그림에서 오른쪽 이미지는 단순히 행렬 형태의 `label` 이미지 입니다. 각 값은 픽셀 별 클래스 값을 가지므로 일반적으로 0 부터 시작하고 (클래스 갯수 - 1)의 값을 최댓값으로 가집니다. 이미지를 읽었을 때에는 `(Height, Width)`만을 가지는 2차원 데이터 입니다.

<br>

- `Pytorch`에서 제공하는 Loss Function은 앞에서 설명한 데이터 타입을 따릅니다.
    - `image` : (B, C, H, W)의 shape을 가져야 합니다.
    - `label` : (B, H, W)의 shape을 가져야 합니다.
- Pytorch의 Loss Function 내부에서 Loss 계산을 할 때, `segmentation 모델의 출력`과 `label`이 연산이 가능하도록 `label`을 변경해야 합니다.

<br>
<center><img src="../assets/img/vision/segmentation/one_hot_label/0.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- `segmentation 모델의 출력`은 위 그림과 같이 픽셀 단위 별로 class의 갯수 만큼의 확률 값을 가지는 벡터가 생기게 되고 (height, width) 크기로 묶어서 보았을 때, 위 그림과 같습니다. 따라서 Loss를 구하기 위해 `label`을 `segmentation 모델의 출력`과 같은 형태로 만들어 주어야 합니다.
- 따라서 `label`을 위 그림과 같이 Channel 방향으로 `one-hot` 형태로 만듭니다. 즉, Channel은 클래스의 갯수 만큼 사이즈를 가지고 해당 해당 클래스에 해당하는 인덱스에 1의 값을 가지도록 `label`을 변경합니다. 따라서 `label`의 shape은 `(B, H, W) → (B, #class, H, W)`로 바뀌게 되며 클래스 dimension은 0 (또는 0에 매우 가까운 값)과 1 (또는 1에 매우 가까운 값)을 가지게 됩니다.

<br>

- 이와 같이 `(B, H, W)` 형태의 `label`을 `(B, C=#class, H, W)` 형태의 `one-hot` 형태로 바꾸어 주는 코드는 다음과 같습니다.

<br>

```python
import torch
from typing import Optional

def segmetaion_label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
        >>> segmetaion_label_to_one_hot_label(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=num_classes, H, W)
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # ret : (B, C=num_classes, H, W)
    ret = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps    
    return ret
```

<br>

```python
labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
segmetaion_label_to_one_hot_label(labels, num_classes=3)

# tensor([[[[1.0000e+00, 1.0000e-06],
#           [1.0000e-06, 1.0000e+00]],

#          [[1.0000e-06, 1.0000e+00],
#           [1.0000e-06, 1.0000e-06]],

#          [[1.0000e-06, 1.0000e-06],
#           [1.0000e+00, 1.0000e-06]]]])

```

<br>

- 위 예제를 살펴보면 (2, 2) 크기의 `labels`을 임시로 만들었습니다. 클래스는 총 3개입니다. 함수 `segmetaion_label_to_one_hot_label`를 이용하여 실행을 하면 위 주석 부분의 출력과 같이 나타나는 것을 확인할 수 있습니다.
- 위 출력을 보면 `labels`의 값이 `one-hot`의 인덱스이고 그 인덱스 부분에 해당하는 값은 1을 가지고 그 이외의 부분은 0에 가까운 값을 가지는 것을 확인할 수 있습니다. channel 방향으로 인덱스를 적용해 보면 쉽게 이해할 수 있습니다.
- 위 코드에서 `.scatter_`의 동작이 `one-hot`의 핵심이며 동작 방식은 다음 링크에서 확인 가능합니다.
    - 링크 : [torch.scatter 사용 방법](https://gaussian37.github.io/dl-pytorch-snippets/#torchscatter-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9-%EC%98%88%EC%A0%9C-1)

<br>

## **segmentation 모델의 출력과 one hot label 로 Loss 구하기**

<br>

- 앞의 과정을 통하여 `image`와 `label` 각각의 연산은 다음과 같은 과정을 거쳐서 shape이 결정되는 것을 확인하였습니다.
- ① `image` : `(B, C=3, H, W)` → segmentation 모델 → `(B, C=#class, H, W)`
- ② `label` : `(B, H, W)` → one hot label 생성 → `(B, C=#class, H, W)`

<br>

- `(B, C=#class, H, W)`로 크기가 같아졌으므로 두 텐서를 단순히 곱하면 element-wise로 곱을 하게 됩니다. 곱의 결과를 살펴보면 one-hot에서 hot(1)에 해당하는 클래스 부분의 확률 값은 유지되고 나머지 부분은 0 또는 0에 가까운 값이 곱해져서 0에 수렴하게 됩니다. 이 결과를 이용하여 Loss를 구하게 되며 그 순서는 다음과 같습니다.
- ① segmentation 모델을 이용하여 prdict를 구합니다. (`(B, C=3, H, W)` → segmentation 모델 → `(B, C=#class, H, W)`)
- ② label을 ont hot label로 생성합니다. (`(B, C=#class, H, W)`)
- ③ (B, C, H, W)에서 C (Channel) 방향으로 sum을 합니다. (`(B, H, W)`)
- ④ 최종 Loss 값이 스칼라 값이 되도록 하기 위하여 `mean` (또는 `sum`)을 적용하여 스칼라 값을 구합니다.

<br>

```python
import cv2

# image : (1, 3, H, W)
image = cv2.imread("input.png")
label = cv2.imread("label.png")

predict = segmentation(image)
one_hot_label = segmetaion_label_to_one_hot_label(label, num_classes=10)
loss_temp = torch.sum(predict * one_hot_label, dim=1)
loss = torch.mean(loss_temp)
# loss = torch.sum(loss_temp)
```

<br>
