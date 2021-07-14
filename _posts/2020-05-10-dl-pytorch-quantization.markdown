---
layout: post
title: Pytorch Quantization 정리
date: 2020-05-10 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [딥러닝, deep learning, quantization, pytorch] # add tag
---

<br>

- 참조 : https://pytorch.org/docs/stable/quantization.html#torch.nn.intrinsic.ConvBnReLU2d
- 참조 : https://pytorch.org/blog/introduction-to-quantization-on-pytorch/
- 참조 : https://pytorch.org/docs/stable/quantization.html
- 참조 : https://youtu.be/c3MT2qV5f9w
- 참조 : https://leimao.github.io/blog/PyTorch-Static-Quantization/
- 참조 : https://leimao.github.io/blog/PyTorch-Quantization-Aware-Training/

<br>

- 이번 글에서는 Pytorch를 이용한 Quantization 방법들에 대하여 정리해 보도록 하겠습니다.

<br>


<br>

## **Pytorch Quantization Aware Training 예시**

<br>

- TensorFlow는 2~16 bit의 quantization을 지원하는 반면에 Pytorch (1.7.0 기준)에서는 int8 quantization을 지원하고 있습니다.
- QAT를 적용하는 전체 workflow는 간단합니다. 단순히 QAT wrapper를 모델에 적용하면 되기 때문입니다. 하지만 추가적으로 고려해야할 점이 있습니다. 바로 `layer fusion`입니다. 경우에 따라서 `layer fusion`을 하지 않으면 QAT를 하더라도 좋은 성능이 나오지 않는 경우가 발생하곤 합니다.
- 이번 예제에서는 TorchVision 모델 중 `ResNet18`을 이용할 예정이며 `layer fusion`과 `skip connections replacement`또한 적용할 예정입니다.

<br>

- QAT의 전체적인 Flow는 다음과 같습니다.

<br>

- ① floating point 타입으로 모델을 학습하거나 pre-trained 모델을 불러옵니다.
- ② 모델을 `CPU` 상태로 두고 학습 모드로 변환합니다. (`model.train()`)
- ③ `layer fusion`을 적용합니다.
- ④ 모델을 평가 모드로 변환 후 (`model.eval()`) layer fusion이 잘 적용되었는 지 확인합니다. 확인 후에는 다시 학습 모드로 변경해 줍니다.
- ⑤ `input`에는 `torch.quantization.QuantStub()`를 적용시키고 `output`에는 `torch.quantization.DeQuantStub()`을 적용시킵니다.
- ⑥ quantization configuration을 지정합니다. (ex. symmetric quantization, asymmetric quantization)
- ⑦ QAT를 하기 위하여 quantization 모델을 준비합니다.
- ⑧ 모델을 다시 `CUDA`가 상태로 적용하고 CUDA를 이용하여 QAT를 모델 학습을 진행합니다.
- ⑨ 모델을 다시 `CPU` 상태로 두고 QAT가 적용된 floating point 모델을 quantized integer model로 변환합니다.
- ⑩ quantized integer model의 정확도 및 성능을 확인합니다.
- ⑪ quantized integer model을 저장합니다.