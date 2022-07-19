---
layout: post
title: 포인트 클라우드 처리를 위한 pptk 사용법 정리
date: 2021-06-30 00:00:00
img: autodrive/lidar/pptk/0.png
categories: [autodrive-lidar] 
tags: [라이다, pptk, 포인트 클라우드] # add tag
---

<br>

- 참조 : https://heremaps.github.io/pptk/viewer.html
- 참조 : https://heremaps.github.io/pptk/tutorials/viewer/semantic3d.html
- 참조 : https://github.com/heremaps/pptk

<br>

- 포인트 클라우드를 다룰 때, `open3d`를 이용하여 포인트 클라우드를 처리하고 시각화 할 때, 시각화 측면에서 아쉬운 점이 몇가지 있었습니다.
- 예를 들어 포인트 클라우드의 좌표를 바로 확인하기 어렵다는 점과 XY, YZ, ZX 평면 관점에서 포인트 클라우드를 2차원상에서 확인하고 싶을 때, 수동으로 매번 각도를 변경해 주어야 하였습니다.
- `pptk (point processing toolkit)` 에서는 `open3d` 보다 좀 더 세련된 방식으로 데이터를 보여주며 앞에서 언급한 문제인 좌표를 확인하는 기능과 XY, YZ, ZX 평면에서 보는 기능을 제공합니다.
- 추가적으로 `pptk`에서 제공하는 다양한 기능을 살펴보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [pptk를 통해 포인트 클라우드를 읽고 시각화](#pptk를-통해-포인트-클라우드를-읽고-시각화-1)
- ### [pptk의 기본 마우스 및 키보드 조작방법](#pptk의-기본-마우스-및-키보드-조작방법-1)

<br>

## **pptk를 통해 포인트 클라우드를 읽고 시각화**

<br>

- 먼저 `pptk`를 설치하려면 아래 링크를 참조하시면 됩니다. 일반적인 상황에서는 `pip install pptk`를 통해 설치 가능합니다.
    - 링크 : https://heremaps.github.io/pptk/install.html
- `pptk`는 기본적으로 numpy의 `(N, 3)` 형태를 입력으로 받습니다. 3D 정보를 표시해야 하므로 열 방향의 차원이 3이 아니면 입력이 되지 않습니다. 추가적으로 `numpy.asarray()`를 통해 numpy로 변환될 수 있는 모든 타입을 입력으로 받을 수 있습니다.

<br>

- 아래 코드는 Numpy를 이용하여 랜덤 포인트를 생성하고 시각화 및 시각화 조건 셋팅 하는 방법입니다. `pptk`는 아래와 같이 `pptk.viewer()`에서 초기 셋팅을 조절하는 방법이 있고 시각화 툴이 실행된 다음에 `v.set()`을 이용하여 interactive 하게 즉각적으로 셋팅을 바꾸는 방법이 있습니다.

<br>

```python
import numpy as np
import pptk

# generate random points
x = np.random.rand(10000, 3)

# visualize points
# input type is numpy with (N, 3) size
v = pptk.viewer(x)
v.set(point_size=0.01)
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 10000 개의 임의의 점이 생성된 것을 확인할 수 있습니다.

<br>

- 앞에서 언급한 바와 같이 `numpy.asarry()`를 통해 (N, 3) 으로 변경 가능한 모든 파이썬의 데이터 타입은 모두 입력으로 사용할 수 있습니다. 다음 예제를 참조하시면 됩니다.

<br>

```python
import pptk
import random

# generate random points
x = []
for _ in range(10000):
    x.append([random.random(), random.random(), random.random()])

# visualize points
# input type is numpy with (N, 3) size
v = pptk.viewer(x)
v.set(point_size=0.01)
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>

## **pptk의 기본 마우스 및 키보드 조작방법**

<br>

##### **화면 회전 및 RGB축 이동**

<br>

- viewer 자체는 Python에서 실행 시 별도 프로세스로 실행됩니다. 따라서 viewer를 실행한 이후에 `pptk.viewer()`에 의해 반환된 핸들을 통해 뷰어를 쿼리하고 조작할 수 있습니다.
- 현재 조작할 수 있는 포인트의 갯수는 GPU 메모리에 달려있고 대략 1억개 미만의 포인트를 다룰 수 있다고 생각하시면 됩니다.
- viewer에서 카메라의 시점의 위치는 `red, green, blue` 축 (이하 `RGB 축`)에 따라 확인할 수 있으며 각 색은 `x, y, z` 축에 대응됩니다. 여기서 주의할 점은 RGB 축은 현재 카메라가 보는 시점이지 좌표계의 중심축이 아닙니다. 즉, `RGB 축`이 이동하여도 각 포인트의 좌표는 변하지 않고 입력 받은 좌표 그대로 표시 됩니다.
- viewer에서 `LMB(Left Mouse Button)`을 이용하여 원하는 방향으로 회전을 할 수 있고 (3D 뷰어 프로그램에서 `pan`의 기능에 해당함) `LMB`을 포인트 근처에서 더블 클릭하면 `RGB 축`이 근처 포인트로 이동합니다. 또한 `Shift + LMB`으로 드래그 하면 `RGB 축`을 옮길 수 있습니다.
- 키보드의 방향 전/후/좌/우 키를 이용하여 화면을 회전할 수 있으며 이는 `LMB`를 이용하여 회전하는 것과 동일합니다.

<br>

##### **포인트 선택 및 정보 추출**

<br>

- viewer의 포인트 클라우드에서 특정 점을 선택해서 위치 정보를 확인하거나 특정점의 정보를 추출하는 방법에 대하여 알아보도록 하겠습니다.
- 먼저 `Ctrl + LMB`를 이용하여 클릭을 하면 특정 점을 선택 할 수 있습니다. 이 때 왼쪽 하단에 점의 시각화 정보와 위치 정보가 함께 표출됩니다.

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 복수 개의 점을 클릭하면 점의 위치 정보는 사라지고 몇 개의 점이 선택되었는 지 정보가 나타나게 됩니다. 선택된 점은 다른 색으로 표시되며 좀 더 쉽게 확인하기 위해서는 점의 크기를 `v.set(point_size=0.01)`와 같이 조절하는 것이 좋습니다.
- 만약 선택된 점을 취소하고 싶으면 `RMB`를 클릭하면 됩니다. 그러면 선택된 모든 점이 선택 취소됩니다.

<br>

- 한번에 영역을 선택하고 싶으면 `Ctrl + LMB`로 드래그하여 영역을 선택 하면 됩니다.

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 이 때, 선택된 영역에서 배제하고 싶은 영역이 있다면 `Ctrl + Shift + LMB`로 배제할 영역을 선택하면 됩니다.

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같이 가운데 영역을 배제할 수 있습니다.

<br>

- 최종적으로 선택한 영역은 파이썬 코드에서 `selected_indexes = v.get('selected')`와 같은 형태로 바로 확인할 수 있습니다. 앞에서 말씀드린 바와 같이 별도 프로세스로 동작하기 때문에 바로 확인할 수 있습니다.

<br>

```python
selected_indexes = v.get('selected')
print(selected_indexes)
# array([5795, 5999, 6854, ..., 4042, 4627, 9207])
print(selected_indexes.shape)
# (3386,)
```

<br>

- 제가 선택한 점은 3386개가 선택되었고 선택된 점들의 정보는 원본 데이터의 인덱스 형태로 반환됩니다. 원본 데이터가 numpy array라면 다음과 같이 사용할 수 있습니다.

<br>

```python
selected_points = x[selected_indexes]
v_selected = pptk.viewer(selected_points)
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 선택된 노란색 영역의 점들만 추출되어 표시된 것을 확인할 수 있습니다. viewer에서는 2차원 사각형을 드래그해서 선택하기 때문에 그 2차원 사각형에 대응되는 3차원 정보의 모든 점들이 선택되는 것을 알 수 있습니다.

<br>

#### **pptk 기본 view 관점 조작 기능**

<br>

- 아래는 pptk의 viewer에서 기본적으로 제공하는 view의 관점 (viewpoint)을 변경하는 기능입니다.

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/7.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 먼저 키보드의 1을 눌렀을 때 기능을 살펴보겠습니다. 

<br>

```python
import numpy as np
import pptk

# generate random points
x = np.random.rand(10000, 3)

# visualize points
# input type is numpy with (N, 3) size
v = pptk.viewer(x, x[:, 1])
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/8.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 키보드 1번을 누르면 `+y` 방향으로 `XZ` 평면을 보여줍니다. 위 코드에서 `pptk.viewer(x, x[:, 1])`의 `x[:, 1]`을 통해 y 방향의 크기에 따라 색을 다르게 표시하였고 y 값이 커질수록 빨간색, 작아질수록 파란색에 가까워집니다.
- `+y` 방향으로 `XZ` 평면을 보았기 때문에 y값이 작은 점이 가장 가깝에 표현되고 y값이 큰 점일 수록 사용자에서 거리가 먼 쪽에 위치되어 보이게 됩니다. (모니터 안쪽 방향으로 점점 증가하는 방향)
- 위 그림은 왼쪽 부터 실행된 기본 뷰, 키보드 1번을 눌러서 `XZ` 평면으로 보인 뷰, `+y` 축 방향인 지 확인하기 위해 손으로 움직인 뷰가 됩니다. (아래 뷰의 예시도 동일하게 진행하겠습니다.)

<br>

- 다음으로 키보드 3을 눌렀을 때 기능을 살펴보겠습니다.

<br>

```python
import numpy as np
import pptk

# generate random points
x = np.random.rand(10000, 3)

# visualize points
# input type is numpy with (N, 3) size
v = pptk.viewer(x, x[:, 0])
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/9.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 키보드 3을 눌렀을 때에는 `-x` 방향으로 모니터 안쪽으로 점점 감소하는 방향으로 보이게 되는 `YZ` 평면을 확인할 수 있습니다.

<br>

```python
import numpy as np
import pptk

# generate random points
x = np.random.rand(10000, 3)

# visualize points
# input type is numpy with (N, 3) size
v = pptk.viewer(x, x[:, 2])
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/10.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 키보드 7을 누르면 `-z` 방향으로 모니터 안쪽으로 점점 감소하는 방향으로 보이게 되는 `XY` 평면을 확인할 수 있습니다. 일반적으로 전체 형상을 확인하기 위해 많이 사용 하는 뷰입니다.

<br>

- 키보드에서 `C`를 누르면 모든 포인트들의 X, Y, Z 값의 평균에 해당하는 위치로 `RGB 축`을 변경합니다. 이는 기본값에 해당하며 만약 `RGB 축`을 옮겼을 경우 원상 복귀를 위하여 사용할 수 있습니다.

<br>

- 필요에 따라서 RGB 축을 원점으로 옮겨야 할 경우도 있습니다. 아래 코드와 같이 `v.set(lookat=(0, 0, 0)`으로 두면 됩니다. 아쉽게도 이러한 셋팅을 했음에도 불구하고 축을 돌리다 보면 이전 셋팅으로 복귀되는 버그가 종종 발생하곤 합니다. 구글링하면 유사한 문제의 글들이 확인되나 아직 고쳐지지 않은 것 같습니다. 그 점 고려하여 사용하면 다음과 같이 사용할 수 있습니다.

<br>

```python
import numpy as np
import pptk

# generate random points
x = np.random.rand(10000, 3)

# visualize points
# input type is numpy with (N, 3) size
v = pptk.viewer(x)
v.set(lookat=(0, 0, 0))
```

<br>
<center><img src="../assets/img/autodrive/lidar/pptk/11.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

