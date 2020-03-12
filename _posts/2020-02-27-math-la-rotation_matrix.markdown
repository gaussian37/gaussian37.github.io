---
layout: post
title: 회전 변환 행렬 (2D, 3D)
date: 2020-01-02 00:00:00
img: math/la/rotation_matrix/0.png
categories: [math-la] 
tags: [선형대수학, 회전 변환, rotation, rotation matrix] # add tag
---

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

- 참조 : https://en.wikipedia.org/wiki/Rotation_matrix
- 참조 : https://ko.wikipedia.org/wiki/회전변환행렬

<br>

- 이번 글에서는 2D와 3D 상태에서의 좌표의 회전 변환하는 방법에 대하여 알아보도록 하겠습니다.

<br>

## **2D에서의 회전 변환**

<br>

- 2D 좌표계에서 회전 변환을 할 때 사용하는 변환 행렬은 다음과 같습니다.

<br>

$$ R(\theta) = \begin{bmatrix} \text{cos}\theta & -\text{sin}\theta \\ \text{sin}\theta & \text{cos}\theta \end{bmatrix} $$

<br>

- 여기서 $$ \theta $$는 각도에 해당합니다. 반시계 방향으로 회전하는 방향이 + 각도가 됩니다.
- 위 회전 행렬을 이용하여 $$ (x, y) $$ 좌표를 회전 변환을 하면 다음과 같습니다.

<br>

$$ \begin{bmatrix} \text{cos}\theta & -\text{sin}\theta \\ \text{sin}\theta & \text{cos}\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}  = \begin{bmatrix} x \text{cos}\theta - y \text{sin}\theta \\ x \text{sin}\theta + y \text{cos}\theta \end{bmatrix} = \begin{bmatrix} x' \\ y' \end{bmatrix} $$

<br>

- 위 식을 이용하여 회전 변환한 좌표를 구하면 다음과 같습니다.

<br>
<center><img src="../assets/img/math/la/rotation_matrix/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 자주 사용하는 회전인 90도 회전 / 180도 회전 / 270도 회전은 다음과 같습니다.

<br>

$$ R(\frac{\pi}{2}) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $$

$$ R(\frac{\pi}{2}) = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix} $$

$$ R(\frac{3\pi}{2}) = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} $$

<br>

### **회전 변환 행렬 유도**

<br>

- 회전 변환을 다루는 방법에 대해서는 위 글에서 다루었습니다. 그러면 왜 저런 형태의 행렬식이 유도되었는 지에 대하여 다루어 보겠습니다.

<br>
<center><img src="../assets/img/math/la/rotation_matrix/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 먼저 앞에서 다룬 회전 변환은 원점을 기준으로 회전을 하게 됩니다. 따라서 위 그림에서도 원점을 중심으로 `P`가 `P'`로 어떻게 변환되는 지 다루어 보도록 하겠습니다.
- 아래 식에서 $$ P, \overline{OP}, \text{cos}(\alpha), \text{sin}(\alpha) $$를 정의해 보겠습니다.

<br>

$$ P = (x, y) $$

$$ \overline{OP} = l = \sqrt{(x - 0)^{2} + (y - 0)^{2})} = \sqrt{x^{2} + y^{2}} $$

$$ \text{cos}(\alpha) = \frac{x}{\overline{OP}} = \frac{x}{\sqrt{x^{2} + y^{2}}} $$

$$ \text{sin}(\alpha) = \frac{y}{\overline{OP}} = \frac{y}{\sqrt{x^{2} + y^{2}}} $$

<br>

- 위 식을 그대로 이용하여 $$ P' $$에 적용해 보도록 하겠습니다. $$ P' = (x', y') $$는  $$ P = (x, y) $$를 $$ +\theta$$ 만큼 회전 시킨 것이므로 회전 각도 만큼 반영해여 식을 적어보겠습니다.

<br>

$$ x' = \sqrt{x^{2} + y^{2}} \text{cos}(\alpha + \theta) $$

$$ y' = \sqrt{x^{2} + y^{2}} \text{sin}(\alpha + \theta) $$

<br>
<center><img src="../assets/img/math/la/rotation_matrix/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 삼각함수의 덧셈 정리를 이용하여 식을 풀어보도록 하겠습니다.

<br>

$$ x' = \sqrt{x^{2} + y^{2}}(\text{cos}(\alpha)\text{cos}(\theta) -\text{sin}(\alpha)\text{sin}(\theta)) $$

$$ x' = \Biggl(\sqrt{x^{2} + y^{2}}\frac{x}{\sqrt{x^{2} + y^{2}}}\text{cos}(\theta) -  \sqrt{x^{2} + y^{2}}\frac{y}{\sqrt{x^{2} + y^{2}}}\text{sin}(\theta) \Biggr) $$

$$ x' = x\text{cos}(\theta) - y\text{sin}(\theta) $$

$$ y' = \sqrt{x^{2} + y^{2}}\text{sin}(\alpha + \theta) $$

$$ y' = \sqrt{x^{2} + y^{2}}(\text{sin}(\alpha)\text{cos}(\theta) + \text{cos}(\alpha)\text{sin}(\theta)) $$

$$ y' = \Biggl(\sqrt{x^{2} + y^{2}}\frac{y}{\sqrt{x^{2} + y^{2}}}\text{cos}(\theta) + \sqrt{x^{2} + y^{2}}\frac{x}{\sqrt{x^{2} + y^{2}}}\text{sin}(\theta) \Biggr) $$

$$ y' = y\text{cos}(\theta) + x\text{sin}(\theta) = x\text{sin}(\theta) + y\text{cos}(\theta) $$

<br>

- 위에서 유도한 식을 정리하면 다음과 같습니다.

<br>

$$ \begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \text{cos}\theta & -\text{sin}\theta \\ \text{sin}\theta & \text{cos}\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} $$

<br>

### **임의의 점을 중심으로 회전 변환**

<br>


<br>

## **3D에서의 회전 변환**

<br>

- 3D에서의 회전 변환은 2차원에서 사용한 회전 변환 행렬을 유사하게 사용합니다. 다만 이 때, 3차원에 맞춰서 행렬의 차원이 늘어나게 되고 각 차원별로 회전을 고려해 주어야 합니다.
- 예를 들어서 $$ R_{x}(\theta) $$는 x축을 중심으로 

<br>

$$ R_{x}(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \text{cos}\theta & -\text{sin}\theta \\ 0 & \text{sin}\theta & \text{cos}\theta \end{bmatrix} $$ 

$$ R_{y}(\theta) = \begin{bmatrix} \text{cos}\theta & 0 & \text{sin}\theta \\ 0 & 1 & 0 \\  -\text{sin}\theta & 0 & \text{cos}\theta \end{bmatrix} $$ 

$$ R_{z}(\theta) = \begin{bmatrix} \text{cos}\theta & -\text{sin}\theta & 0 \\ \text{sin}\theta & \text{cos}\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} $$ 

<br>

- 이 행렬을 정리해 보려고 하는데, 그 전에 `roll`, `yaw`, `pitch`에 대하여 알아보겠습니다.

<br>
<center><img src="../assets/img/math/la/rotation_matrix/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 일반적으로 `roll`은 x축을 기준으로 회전한 양을 뜻하고 `pitch`는 y축을 기준으로 회전한 양 그리고 `yaw`는 z축을 기준으로 회전한 양을 뜻합니다. 위 그림처럼 생각하시면 됩니다.
    - 예를 들어 자동차가 좌회전 또는 우회전을 한다면 z축을 기준으로 회전을 하는 것이므로 `yaw`의 변화가 있게 됩니다.
- 그러면 $$ R_{x}(\theta) $$, $$ R_{y}(\theta) $$ 그리고 $$ R_{z}}(\theta) $$ 각각 x축, y축, z축을 기준으로 회전하는 회전 변환 행렬이 됩니다.
- x축을 기준으로 회전한 `roll angle`을 $$ \gamma $$, y축을 기준으로 회전한 `pitch angle`을 $$ \beta $$ 마지막으로 z축을 기준으로 회전한 `yaw angle`을 $$ \alpha $$로 두겠습니다.

<br>

$$ R = R_{z}(\alpha)R_{y}(\beta)R_{x}(\gamma) = \begin{bmatrix} \text{cos}\alpha & -\text{sin}\alpha & 0 \\ \text{sin}\alpha & \text{cos}\alpha & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} \text{cos}\beta & 0 & \text{sin}\beta \\ 0 & 1 & 0 \\  -\text{sin}\beta & 0 & \text{cos}\beta \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & \text{cos}\gamma & -\text{sin}\gamma \\ 0 & \text{sin}\gamma & \text{cos}\gamma \end{bmatrix} $$

<br>

- 위 변환 행렬을 모두 곱하면 roll, pitch, yaw angle을 모두 고려한 회전을 나타낼 수 있습니다.
- 위 식을 풀어서 나타내면 다음과 같습니다.

<br>

$$ R = \begin{bmatrix} \text{cos}\alpha \ \text{cos}\beta & \text{cos}\alpha \ \text{sin}\beta \ \text{sin}\gamma - \text{sin}\alpha \ \text{cos}\gamma & \text{cos}\alpha \ \text{sin}\beta \ \text{cos}\gamma + \text{sin}\alpha \ \text{sin}\gamma \\ \text{sin}\alpha \ \text{cos}\beta & \text{sin}\alpha \ \text{sin}\beta \ \text{sin}\gamma + \text{cos}\alpha \ \text{cos}\gamma & \text{sin}\alpha \ \text{sin}\beta \ \text{cos}\gamma - \text{cos}\alpha \ \text{sin}\gamma \\ -\text{sin}\beta & \text{cos}\beta \ \text{sin} \gamma & \text{cos}\beta \ \text{cos} \gamma \\ \end{bmatrix} $$

<br>

[선형대수학 글 목차](https://gaussian37.github.io/math-la-table/)

<br>

