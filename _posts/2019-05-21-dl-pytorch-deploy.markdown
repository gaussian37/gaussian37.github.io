---
layout: post
title: pytorch 모델 저장과 ONNX 사용
date: 2019-05-18 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, deploy] # add tag
---

<br>

- 이 글에서는 pytorch를 이용하여 학습한 결과를 저장하거나 다른 프레임워크에서 사용하는 방법에 대하여 다루어 보겠습니다.

## **목차**

- ### 모델 저장과 불러오기

<br>

## **모델 저장과 불러오기**

<br>

- 먼저 모델의 저장과 불러오는 방법에 대하여 다루어 보도록 하겠습니다. 
- 딥러닝이나 머신러닝에서 학습한 모델을 저장해 두었다가 나중에 재사용하는 것은 필수적입니다. 
- 이 때, `모델 저장`하는 것은 `모델 구조 자체 + 학습한 파라미터` 두가지를 함께 저장하는 것을 뜻합니다.
- 모델 구조는 코드를 그대로 사용하면 되지만 학습한 파라미터는 대량의 수치 데이터이므로 파일로 저장할 필요가 있습니다.

<br>

- 먼저 파이토치에서는 `state_dict` 메서드를 사용하여 파라미터의 텐서를 사전 형식으로 추출할 수 있고, `torch.save`라는 `pickle`의 wrapper 함수를 사용하여 파일로 저장할 수 있습니다.
- 참고로 `pickle_protocol = 4` 옵션은 pickle 형식으로 큰 객체를 효율적으로 저장할 수 있습니다.

<br>

```python
# 학습이 끝난 신경망 모델
params = net.state_dict()
# net.prm라는 파일로 저장
torch.save(params, "net.prm", pickle_protocol = 4)
```

<br>

- 저장한 파라미터 파일을 불러올 때에는 `torch.load` 함수를 사용하고 `nn.Module`의 `load_state_dict` 함수에 전달하면 딥러닝 모델에 파라미터를 설정할 수 있습니다.

<br>

```python
# net.prm 불러오기
params = torch.load("net.prm", map_location = "cpu")
net.load_state_dict(params)
```

<br>

- 여기서 map_location은 읽은 파라미터를 어디에 저장할 것인지를 나타냅니다. 
- 예를 들어 GPU로 학습한 모델의 파라미터를 그대로 저장하면 `torch.load`로 읽는 경우 우선 CPU로 불러온 후에 GPU로 전송합니다.
- 이 경우에 GPU가 없는 서버에서 사용할 때에는 오류가 발생해 버립니다. 
- 이런 오류를 방지하기 위하여 map_location 인수를 사용하여 읽은 후의 동작을 지정합니다.
- 파라미터를 저장할 때에는 어떻게 사용할 지를 고려해서 GPU 또는 CPU 버전으로 저장해 두어야 합니다. 만약 CPU에서 사용한다면 다음 예와 같이 일단 모델을 CPU로 전송해 두는 것도 좋은 방법입니다. 

<br>

```python
# 일단 CPU로 모델 이동
net.cpu()
# 파라미터 저장
params = net.state_dict()
torch.save(params, "net.prm", pickle_protocol=4)
```
