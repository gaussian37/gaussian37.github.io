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

<br>

## **목차**

- ### 모델 저장과 불러오기
- ### ONNX를 사용한 다른 프레임워크와의 연계

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

<br>

## **ONNX를 사용한 다른 프레임워크와의 연계**

<br>

-  ONNX란 Open Neural Network eXchange의 줄임말입니다.
- 이 글에서는 ONNX라는 신경망 모델의 표준 포맷에 대하여 설명하고, `pytorch`로 학습한 모델을 ONNX를 경유해서 `Caffe2`로 실행해 보겠습니다. (pytorch와 caffe2 모두 페이스북에서 개발한 것으로 비교적 안정적으로 동작합니다.)
- 먼저 ONNX에 대하여 간단하게 살펴보면 Facebook과 MS가 개발한 신경망 모델의 호환 포맷입니다. 
- 특정 프레임워크에서 작성한 모델을 다른 프레임워크에서 사용할 수 있게 하는 도구로 `pytorch, caffe, CNTK, MXNet`등을 지원하고 caffe나 MXNet 같은 C++ 기반의 프레임워크를 지원하기 때문에 모바일 기기에 배포할 수 있어서 응용 범위가 넓어집니다.
- 예를 들면 Pytorch 작성 → ONNX 변환 → Caffe에서 임포트 하는 순서로 사용 가능합니다.

<br>

### pytorch에서 학습 완료된 모델 불러오기

<br>

- 먼저 학습이 완료된 모델을 불러오겠습니다. 아래 코드는 예제로 참고만 하시기 바랍니다.
- 물론 모델을 저장할 때에는 `평가`모드로 저장해야 합니다.

<br>

```python
from torchvision import models

# resnet18의 binary classification 예제
def CreateNetwork():
    net = models.resnet18()
    fc_input_dim = net.fc.in_features
    net.fc = nn.Linear(fc_input_dim, 2)
    return net

# 모델 생성
net = CreateNetwork()

# 파라미터 읽기 및 모델에 설정
prm = torch.load("net.prm", map_location="cpu")
net.load_state_dict(prm)

# 평가 모드로 설정
net.eval()
```

<br>

### ONNX로 export

<br>

- 다음은 ONNX로 export를 해보겠습니다. 
- pytorch는 동적인 계산 그래프를 이용해야 하므로 export할 때에는 네트워크 계산을 직접 한번 해야 합니다. 
- 이를 위해서는 실제 이미지를 사용할 필요는 없고 입력 데이터와 크기가 같은 더미용 데이터를 사용해도 됩니다.
- 예를 들어 입력 데이터의 크기가 (3, 224, 224)의 이미지라고 가정해보고 배치 차원도 포함해서 (1, 3, 224, 224)라고 하면 더미 데이터도 이 크기로 만들면 됩니다.

<br>

```python
import torch.onnx

dummy_data = torch.empty(1, 3, 224, 224, dtype = torch.float32)
torch.onnx.export(net, dummy_data, "output.onnx")
```

<br>

- 위와 같이 코드를 입력하면 export가 완료됩니다. ONNX의 제약 중 하나는 pytorch 등의 동적 계산 프레임워크로부터 export할 경우에 1회 계산을 해야하므로 네트워크 내에서 if문 등으로 분기가 이루어지지 않는 경우에는 제대로 export가 되지 않을 수 있습니다. 
 
 <br>

### Caffe2에서 ONNX 모델 사용

 <br>

 - ONNX 모델을 불러오면 onnx 패키지가 필요하며, caffe2에서 ONNX 모델을 사용하려면 caffe2에 포함되어 있는 caffe2.python.onnx라는 패키지를 사용하여 변환해야 합니다.
 - 다음은 caffe2에서 ONNX를 이용하여 임포트 하는 코드 입니다.

 <br>

 ```python
import onnx
from caffe2.python.onnx import backend as caffe2_backend

# ONNX 모델 불러오기
onnx_model = onnx.load("output.onnx")

# ONNX 모델을 caffe2 모델로 변환하기
backend = caffe2_backend.prepare(onnx_model)
 ```
 
 <br>

- 그리고 난 후 `backend.run`에 이미지 데이터를 numpy의 ndarray 형태로 넣으면 계산할 수 있습니다.
- 원래 pytorch 모델과 ONNX 경유로 caffe2에서 사용한 모델이 동일한 결과를 보여주는 것을 확인할 수 있습니다.
- 다음은 pytorch 모델과 ONNX 경유 caffe2 모델을 비교하는 코드입니다.

<br>

```python
from PIL import Image
from torchvision import transforms

# 이미지를 잘라서 텐서로 변환하는 함수
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 이미지 불러오기
img = Image.open("test.jpg")

# 텐서로 변환해서 배치 차원을 더함
img_tensor = transform(img).unsqueeze(0)

# ndarray로 변환
img_ndarray = img_tensor.numpy()

# 파이토치로 실행할 때에는 tensor를 입력
net(img_tensor)
: tensor([[ 1.1262, -1.8448]])

# ONNX와 caffe2로 실행할 때에는 numpy를 입력
output = backend.run(img_ndarray)
output[0]
: array([[ 1.126245 , -1.8447802]], dtype=float32)
```

<br>

- 내부 처리가 다르므로 세세한 오차는 있지만, 동일 계산 결과가 반환됩니다. 
- pytorch로 작성한 신경망이 ONNX를 경유하여 caffe2에서 제대로 실행되는 것을 확인하실 수 있을 것입니다.

<br>

### ONNX 모델을 caffe2 모델로 저장

<br>

- 다음 예에선 ONNX 모델을 실행하기 위하여 caffe2를 백엔드로 사용하고, 딥러닝 계산 자체는 caffe2의 API를 사용하는 예제입니다.
- 먼저 ONNX에 의존하지 않고 단순한 caffe2 모델로 변환하는 예제를 살펴보겠습니다.

<br>

```python
from caffe2.python.onnx.backend import Caffe2Backend
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model)
```

<br>

- `onnx_graph_to_caffe2_net`은 caffe2의 뉴럴넷과 파라미터를 생성합니다. 이 두가지 결과물을 다음과 같이 파일로 저장해보겠습니다.
- 따로 파일을 저장한다면 ONNX를 더 이상 사용하지 않고 caffe2만으로도 inference를 할 수 있습니다.
- 즉, 학습은 pytorch로 시작하고 그 결과를 ONNX 형태로 저장한 다음에 caffe2에서 사용할 수 있도록 하였고 더 나아가 caffe2만으로 사용할 수 있도록 caffe2 형태로 다 변환할 수 있음을 보여줍니다.

<br>

```python
with open('init_net.pb', "wb") as fopen: 
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb'，"wb") as fopen: 
    fopen.write(predict_net.SerializeToString())
```

<br>