---
layout: post
title: Vehicle Dynamics와 Bicycle Model
date: 2023-09-01 00:00:00
img: autodrive/ose/vehicle_dynamics_and_bicycle_model/0.png
categories: [autodrive-ose] 
tags: [차량 동역학, vehicle dynamics, bicycle model, tire model, dynamic bicycle model] # add tag
---

<br>

[Optimal State Estimation 글 목차](https://gaussian37.github.io/autodrive-ose-table/)

<br>

- 이번 글은 Andreas Geiger의 Self-Driving Cars 강의 내용을 참조하여 작성하였습니다.

<br>

## **목차**

<br>

- ### Vehicle Dynamics
- ### Kinematic Bicycle Model
- ### Tire Models
- ### Dynamic Bicycle Model

<br>

## **Vehicle Dynamics**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/wuUUN_DvYP4" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/6.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/7.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 먼저 7, 8 슬라이드에서 설명하는 `holonomic`과 `non-holonomic`의 정의를 쉽게 설명한 글을 아래 링크를 통해 참조할 수 있습니다.
    - 링크 : [holonomic과 non-holonomic의 정의 설명](https://konnect.news/[%EB%B0%95%EC%9A%B0%EB%9E%8C-%EA%B5%90%EC%88%98%EC%9D%98-%EC%9D%BC%EC%83%81-%EC%86%8D-%EA%B3%BC%ED%95%99%EA%B8%B0%EC%88%A0-%EC%9D%B4%EC%95%BC%EA%B8%B0]-%ED%8F%89%ED%96%89-%EC%A3%BC%EC%B0%A8%EB%8A%94-%EC%99%9C-%EC%96%B4%EB%A0%A4%EC%9A%B8%EA%B9%8C--p853-128.htm)

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/8.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/9.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/10.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/11.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/12.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/13.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/14.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/14_1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

## **Kinematic Bicycle Model**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/D4AgX1zjx54" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/16.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/17.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/17_1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/17_2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/18.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/18_1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/18_2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/18_3.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/18_4.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/18_5.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/19.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/19_1.png" alt="Drawing" style="width: 1000px;"/></center>
<br>


<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/19_2.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- 아래는 19 페이지의 슬라이드를 기준으로 $$ \beta, \delta $$ 를 모두 이용하여 `Kinematic Bicycle Model`을 정의한 내용입니다. 
- `initialize parameters`에서 `L`, `lr`, `dt`는 실제 차량 환경을 고려한 길이를 넣어주어야 합니다. 
- 아래 코드의 `constant inputs`으로 정의한 `v` 와 `delta` 또한 실제 차량의 움직임의 센서값들을 넣어야 정상 동작합니다. 아래 코드는 고정된 `v`, `delta` 상수값을 예시로 작성한 결과입니다.

<br>

```python
import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters
L = 2.0  # Wheelbase length
lr = 1.0  # Distance from center of mass to the rear axle
dt = 0.1  # Time step (s)
T = 10.0  # Total simulation time (s)
N = int(T / dt)  # Number of time steps

# Initialize state variables
x = np.zeros(N)
y = np.zeros(N)
psi = np.zeros(N)

# Initial conditions
x[0] = 0.0
y[0] = 0.0
psi[0] = 0.0

# Constant inputs for this example
v = 1.0  # Velocity (m/s)
delta = np.pi / 6  # Steering angle (rad)

# Time-discretized equations of motion
for t in range(N - 1):
    beta = np.arctan((lr / L) * np.tan(delta))
    x[t + 1] = x[t] + dt * v * np.cos(psi[t] + beta)
    y[t + 1] = y[t] + dt * v * np.sin(psi[t] + beta)
    psi[t + 1] = psi[t] + dt * (v / L) * np.cos(beta) * np.tan(delta)

# Plotting the trajectory
plt.figure()
plt.plot(x, y)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectory')
plt.grid(True)
plt.show()
```

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/20.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/21.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

## **Tire Models**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/gikM0m3AWIk" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/23.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/24.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/25.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/26.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/27.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/28.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/29.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

## **Dynamic Bicycle Model**

<br>
<div style="text-align: center;">
    <iframe src="https://www.youtube.com/embed/6fyUnoRxPvs" frameborder="0" allowfullscreen="true" width="800px" height="400px"> </iframe>
</div>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/31.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/32.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/33.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/34.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/35.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/36.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/37.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/38.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/39.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/40.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/41.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/42.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

<br>
<center><img src="../assets/img/autodrive/ose/vehicle_dynamics_and_bicycle_model/43 .png" alt="Drawing" style="width: 1000px;"/></center>
<br>




<br>

[Optimal State Estimation 글 목차](https://gaussian37.github.io/autodrive-ose-table/)

<br>