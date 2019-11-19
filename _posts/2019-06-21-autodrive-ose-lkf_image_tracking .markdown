---
layout: post
title: (칼만필터는 어렵지 않아) 칼만 필터 이미지 물체 추적
date: 2019-06-21 01:00:00
img: ad/kalmanfilter/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [컴퓨터 비전, 칼만 필터, kalman filter] # add tag
---

<br>

- 이번 글에서는 2차원 평면 위에서 움직이는 물체를 추적하는 방법에 대하여 알아보겠습니다.
- 물론 움직이는 물체를 찾는 역할은 칼만 필터가 아니라 컴퓨터 비전의 알고리즘을 이용하여 알아내야 하고 칼만 필터는 영상 처리 기법으로 알아낸 물체의 위치를 받아서 정확한 위치를 추정하는 역할을 합니다.
- 칼만 필터는 **노이즈를 제거**하거나 **측정하지 않은 값을 추정**해 내는 역할을 하므로 이미지 내에서의 칼말 필터는 `위치의 오차를 제거`하고 `이동 속도를 추정`하는 역할을 합니다.  

<br>

- 평면 상의 표적 추적에서 칼만 필터를 적용하기 위해서는 `2차원의 위치 및 속도 모델`이 필요합니다.

<br>

$$ x = 
    \begin{Bmatrix}
    p_{x} \\
    v_{x} \\
    p_{y} \\
    v_{y} \\
    \end{Bmatrix}
$$

<br>

- 위 식에서 $$ p $$는 위치를 나타내고 $$ v $$는 속도를 나타냅니다. $$ x, y $$ 각각 축의 방향을 나타냅니다.
- 물론 위 상태 변수의 순서는 중요하지 않습니다. 위치 변수와 속도 변수를 따로 모아도 되며 이 때에는 시스템 모델을 구성할 때 그에 맞춰서 맞추어 주면 됩니다.
    - [칼만 필터의 전체 플로우](https://gaussian37.github.io/vision-kalmanfilter-basic-kalman-filter/)는 이 링크를 참조하시기 바랍니다. 

<br>

$$ A = 
    \begin{bmatrix}
    1 & \Delta t & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & \Delta t \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}, \ \     
    
$$

$$
    H = 
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    \end{bmatrix},
$$

<br>

$$ x_{k+1} = Ax_{k} + w_{k} $$

<br>

$$ z_{k} = Hx_{k}  + v_{k} $$

<br>

- 그러면 $$ x $$축과 $$ y $$축 방향으로 각각 다음과 같은 관계를 행렬식으로 표현할 수 있습니다.

<br>

$$ 
    \begin{bmatrix}
    p_{k+1} \\
    v_{k+1} \\
    \end{bmatrix}
    = 
    \begin{bmatrix}
    p_{k} + v_{k} \cdot \Delta t  \\
    v_{k} \\
    \end{bmatrix} + w_{k}
$$

<br>

- 위 식에서 $$ w $$를 노이즈로 보면 $$ x, y $$축 방향의 위치만 측정하고 속도는 측정하지 않는다는 의미를 가지고 있습니다.
- 즉, 처음에 설명한 칼만 필터의 역할에 따라 `노이즈를 제거`하고 `측정하지 않은 값인 속도를 추정`하는 역할을 하고 있습니다.

<br>

```python
import numpy as np
from numpy import transpose
from numpy.linalg import inv

# 파라미터 초기화
def init():
    global dt, A, H, Q, R, P, x

    dt = 1
    
    A = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1],
    ])

    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    
    # Q와 R은 튜닝 파라미터
    Q = np.eye(4) 

    R = np.array([
        [50, 0],
        [0, 50]
    ])

    P = 100 * np.eye(4)

    x = np.array([0, 0, 0, 0])

# 선형 칼만필터 함수
# input: 객체 검출한 측정 좌표 (x_, y_)
# output: 칼만 필터 적용한 객체 좌표와 속도 (추정 x좌표, x축 속도, 추정 y좌표, y축 속도)
def KalmanTracking(x_, y_):
    global A, H, Q, R, P, x
    
    xp = A@x
    Pp = A@P@transpose(A) + Q
    
    K = Pp@transpose(H)@inv(H@Pp@transpose(H) + R)
    
    z = (x_, y_)
    x = xp + K@(z - H@xp)
    P = Pp - K@H@Pp 
    
    return x


init()
KalmanTracking(1, 1)
```

<br>

- 다른 글에서 설명한 바와 같이 `Q`와 `R`을 변경하면서 칼만 필터의 성능을 개선할 수 있습니다. 
- moving average filter, low pass filter에서도 간략하게 살펴본 바와 같이 파라미터를 변경하는 것에 따라 **측정값에 민감하게 할 것인지** 또는 **추정값에 민감하게 할 것인지** 가중치를 줄 수 있습니다.
- 칼만 필터 알고리즘에서도 동일한 역할을 할 수 있습니다. 자세한 내용은 앞의 글을 참조하시기 바랍니다.
- 결과만 살펴보면 **Q가 커지면 측정값에 가까워지고 작아지면 노이즈가 줄어들어 추정값에 가까워집니다.**
- 반대로 **R은 작아지면 측정값에 가까워지고 커지면 추정값에 가까워집니다.**
- 파라미터 Q와 R의 역할은 반대입니다. Q와 R의 값을 적당히 변경해가면서 실험하면 전체적으로 검출한 좌표의 `노이즈를 제거 하고 속도를 추정`할 수 있습니다.

<br>

- 다음은 cpp 기반으로 만든 함수 입니다.
- cpp 기반으로 만들면 transpose, inverse 그리고 행렬곱을 하는 함수를 따로 만들어야 합니다.