---
layout: post
title: 카메라 캘리브레이션의 이해와 Python 실습
date: 2022-01-28 00:00:00
img: vision/concept/calibration/0.png
categories: [vision-concept] 
tags: [vision, concept, calibaration, 캘리브레이션] # add tag
---

<br>

- 참조 : https://ms-neerajkrishna.medium.com/
- 참조 : https://darkpgmr.tistory.com/32
- 참조 : https://www.mathworks.com/help/vision/ug/camera-calibration.html

<br>

- 이번 글에서는 컴퓨터 비전을 위한 카메라 내용 및 카메라 캘리브레이션 관련 내용과 파이썬을 이용하여 실습을 해보도록 하겠습니다.

<br>

## **목차**

<br>

- ### [이미지 형성과 핀홀 모델 카메라](#이미지-형성과-핀홀-모델-카메라-1)
- ### [Camera Extrinsic Matrix with Example in Python](#camera-extrinsic-matrix-with-example-in-python-1)
- ### [Camera Intrinsic Matrix with Example in Python](#camera-intrinsic-matrix-with-example-in-python-1)
- ### [Find the Minimum Stretching Direction of Positive Definite Matrices](#)
- ### [Camera Calibration with Example in Python](#camera-calibration-with-example-in-python-1)

<br>

## **이미지 형성과 핀홀 모델 카메라**

<br>

- 이미지 형성의 기본 아이디어는 `object`에서 `medium`으로 반사되는 광선(Rays)을 포착하는 것에서 부터 시작합니다.
- 가장 단순한 방법은 `object` 앞에 `medium`을 놓고 반사되어 들어오는 광선을 캡쳐하면 됩니다. 하지만 단순히 이러한 방식으로 하면 필름 전체에 회색만 보일 수 있습니다. 왜냐하면 object의 다른 지점에서 나오는 광선이 필름에서 서로 겹쳐서 엉망이 되기 때문입니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림을 살펴보면 object에서 부터 반사되어 나온 광선이 Medium의 같은 위치에서 수집될 수 있습니다. 이런 경우 object로 부터 반사되어 나온 위치의 형상을 정확히 알 수 없습니다. 

<br>
<center><img src="../assets/img/vision/concept/calibration/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 위 그림과 같이 `pin hole` 구조를 이용하여 object의 어떤 위치가 medium의 픽셀 간 일대일 대응이 될 수 있도록 만듭니다. 이와 같은 방법을 이용하여 medium의 한 픽셀에 object의 여러 부분이 겹치는 문제를 개선할 수 있습니다.
- 하지만 위 그림에서 한가지 문제점이 있습니다. 광선이 medium에 맺히는 위치가 반전이 되어 있습니다. 예를 들어 나무 윗부분의 물체가 medium의 아랫 부분에 위치해 있는 것을 알 수 있습니다. 이것이 `pin hole` 카메라의 특성이며 이 성질을 이용하여 수학적으로 모델링을 해야 합니다.

<br>

#### **Pinhoe camera**

<br>
<center><img src="../assets/img/vision/concept/calibration/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서 설명한 핀홀 카메라 모델은 대부분의 컴퓨터 비전 분야에서 기본적으로 사용됩니다. 하지만 핀홀 카메라 모델은 매우 이상적인 카메라 모델이며 실제로는 렌즈의 특성이 반영되어야 하므로 영상 왜곡 등도 같이 고려되어야 합니다.

<br>

- 그러면 핀홀 카메라의 원리를 이용하여 수학적으로 모델링 하기 위하여 아래 2가지 가정을 전제로 두겠습니다.
- ① 앞의 그림에서 광선을 캡쳐하는 medium 또는 필름 구조를 `image plane` 이라고 하며 촬영자 - `image plane` - 핀홀- 물체 순서로 위치해 있습니다. 편의상 핀홀 앞쪽이라고 말하겠습니다. 하지만 실제로는 `image plane`은 핀홀 뒤쪽에 위치해 있습니다. 이와 같은 구조로 설계하는 것이 물체로부터 반사된 광선을 image plane에 projection하기 쉽기 때문이며 앞서 언급한 반전된 이미지 문제도 해소할 수 있기 때문입니다. 자세한 구조는 생략하며 여기서 짚고 넘어갈 점은 촬영자 - 핀홀 - `image plane` - `world space` 순서로 위치한 점입니다.
- ② world space에서 부터 반사된 수많은 광선은 핀홀로 수렴된다고 가정합니다. 이 지점을 `center of projection` 또는 `camera center`라고 합니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 이러한 가정을 이용하면 `world space`에 있는 물체들이 `image plane`에 투영되고 최종적으로 `center of projection (camera center)`로 수렴되는 구조로 이해할 수 있습니다.
- `image plane`은 XY plane과 평행하고 `center of projection`과 일정 거리 떨어져 있습니다. 이 거리를 `focal length`라고 하며 이후에 다룰 예정입니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 좌표계는 위 그림과 같으며 오른손으로 엄지, 검지, 중지로 나타내었을 때, 각각 y, z, x 축을 가리킵니다. 이를 참조하여 좌표계를 이해하시면 됩니다.

<br>

#### **Geometric camera calibration**

<br>

- 카메라는 어느 위치에나 장착될 수 있고 카메라가 보는 방향도 제각각 입니다. 여기서 해결해야 할 점은 `world space`에서 반사되어 들어오는 광선을 어떻게 `image plane`에 투영할 지 정해야 한다는 것입니다. 바꿔 말하면 `world space`와 `image plane` 간의 관계를 알아내어야 실제 이미지를 만들어 낼 수 있습니다.
- 이 관계는 `world space` → `image planem`으로 변환하는 행렬을 구해야 하며 이 때 필요한 2가지 행렬을 `extrinsic`, `intrinsic` 이라고 합니다.
- `extrinsic` : `world space`의 좌표계를 `world coordinate system`이라고 하고 앞에서 $$ X, Y, Z $$ 축으로 표현한 좌표계를 `camera coordinate system`이라고 합니다. 이 때, `world coordinate system` → `camera coordinate system`으로 좌표계를 변환할 때 사용하는 행렬을 `extrinsic`이라고 합니다. 이 행렬은 카메라가 실제 장착된 위치 등의 환경과 관련이 있습니다.
- `intrinsic` : `camera coordinate system`의 점들을 `image plane`의 좌표로 변환하는 행렬을 `intrinsic`이라고 합니다. 이 행렬은 카메라 내부 환경과 관련이 있습니다.
- `extrinsic`과 `intrinsic`을 확인하는 것을 카메라 캘리브레이션이라고 합니다.

<br>

## **Camera Extrinsic Matrix with Example in Python**

<br>

- 카메라가 설치되는 위치와 방향에 따라 `world coordinate system`에서 `camera coordinate system`으로 변형하기 위하여 `extrinsic`이 필요하다고 앞에서 설명하였습니다.
- `extrinsic`을 구하기 위해서는 world space 상에서 카메라의 방향과 위치를 알아야 하며 이것을 알기 위해서는 ① `rotation`과 ② `translation`에 대한 변환이 어떻게 되어있는 지 알아야 합니다. world space 상에서의 좌표 기준이 있고 그 좌표계에서 카메라가 얼만큼 회전(rotation)이 되었는 지를 알고 카메라가 얼만큼 이동(translation)하였는 지 알면 카메라 좌표계 상에서의 위치 변화를 알 수 있습니다.

<br>

- 지금부터 살펴볼 내용은 `rotation`과 `translation` 각각에 대하여 기저 (basis) 변환을 어떻게 하는 지 살펴보려고 합니다. 기저 변환을 확인하기 위하여 먼저 ① 같은 기저 내에서 점 $$ P \to P' $$ 로 `rotation`과 `translation`을 하는 방법에 대하여 알아보고 ② 기저1의 점 $$ P $$ 가 기저2에서는 어떤 좌표값을 가지는 지 살펴도록하곘습니다.
- 이와 같이 기저 변환을 통하여 좌표가 어떻게 바뀌는 지 알아보는 이유는 world space 상의 `world coordinate system`에서 `camera coordinate system`으로 기저 변환을 하기 위함입니다.

<br>

#### **Change of coordinates by rotation**

<br>

- 점 $$ P $$가 $$ \theta $$ 만큼 회전하였을 때 좌표를 구하기 위하여 다음 그림을 참조해 보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/6.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 예시는 2차원 ($$ $$ \mathbb{R}^{2} $$) XY 평면에서 점 $$ P $$ 를 $$ \theta $$ 만큼 회전하여 $$ P' $$ 를 얻을 때 사용하는 행렬을 나타냅니다. 그러면 위 그래프를 기준으로 식을 전개해 보도록 하겠습니다.
- 먼저 $$ \alpha $$ 각도에 대하여 다루어 보도록 하곘습니다.

<br>

- $$ \sin{(\alpha)} = \frac{y}{r}, \cos{(\alpha)} = \frac{x}{r} \tag{1} $$

<br>

- 위 식에서 $$ r $$ 을 기준으로 등식을 만들어 정리하면 다음과 같이 두 식을 정리할 수 있습니다.

<br>

- $$ x\sin{(\alpha)} = y \cos{(\alpha)} \tag{2} $$

<br>

- 이번에는 $$ \theta + \alpha $$ 각도에 대하여 다루어 보도록 하겠습니다.

<br>

- $$ x' = r\cos{(\theta + \alpha)} \tag{3} $$

- $$ \cos{(\theta + \alpha)} = \cos{(\theta)}\cos{(\alpha)} - \sin{(\theta)}\sin{(\alpha)} \tag{4} $$

<br>

- 식(4)를 이용하여 식(3)을 전개해 보도록 하겠습니다.

<br>

- $$ \Rightarrow r\cos{(\theta + \alpha)} = r(\cos{(\theta)}\cos{(\alpha)} - \sin{(\theta)}\sin{(\alpha)}) \tag{5} $$

- $$ = r(\cos{(\theta)}\frac{x}{r} - \sin{(\theta)}\frac{y}{r}) = x\cos{(\theta)} - y\sin{(\theta)} \tag{6} $$

- $$ \therefore x' = x\cos{(\theta)} - y\sin{(\theta)} \tag{7} $$

<br>

- 이번에는 다른 식을 살펴보도록 하겠습니다.

<br>

- $$ y' = r\sin{(\theta + \alpha)} \tag{8} $$

- $$ \sin{(\theta + \alpha)} = \sin{(\theta)}\cos{(\alpha)} + \cos{(\theta)}\sin{(\alpha)} \tag{9} $$

<br>

- 식(7)을 전개하는 과정과 동일한 방식으로 식(8)을 식(9)를 이용하여 정리하면 다음과 같습니다.

<br>

- $$ y' = x\sin{(\theta)} + y\cos{(\theta)} \tag{10} $$

<br>

- 식 (7)과 식(10)을 묶어서 행렬로 나타내면 다음과 같습니다.

<br>

- $$ \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos{(\theta)} & -\sin{(\theta)} \\ \sin{(\theta)} & \cos{(\theta)} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} \tag{11} $$

<br>

- 식(11)을 이용하면 점 $$ P $$를 $$ P' $$ 로 변환할 수 있습니다.

<br>

- 위 예시는 2차원 평면에서의 회전 변환을 나타냅니다. 만약 3차원 평면에서의 회전이 발생하면 어떻게 될까요? 각 $$ X, Y, Z $$ 축 방향으로 회전 변환 행렬을 적용하면 됩니다. 변환 행렬은 다음과 같습니다.

<br>

- $$ R_{x}(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \text{cos}\theta & -\text{sin}\theta \\ 0 & \text{sin}\theta & \text{cos}\theta \end{bmatrix} \tag{12} $$

- $$ R_{y}(\theta) = \begin{bmatrix} \text{cos}\theta & 0 & \text{sin}\theta \\ 0 & 1 & 0 \\  -\text{sin}\theta & 0 & \text{cos}\theta \end{bmatrix} \tag{13} $$

- $$ R_{z}(\theta) = \begin{bmatrix} \text{cos}\theta & -\text{sin}\theta & 0 \\ \text{sin}\theta & \text{cos}\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \tag{14} $$

<br>

- 전체 변환 행렬 $$ R $$ 은 $$ R = R_{z}(\alpha)R_{y}(\beta)R_{x}(\gamma) $$ 로 행렬 곱을 통해 나타낼 수 있습니다. $$ \alpha, \beta, \gamma $$ 각각은 각 축으로 회전한 각도를 의미합니다.

<br>

- $$ R = R_{z}(\alpha)R_{y}(\beta)R_{x}(\gamma) = \begin{bmatrix} \text{cos}\alpha & -\text{sin}\alpha & 0 \\ \text{sin}\alpha & \text{cos}\alpha & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} \text{cos}\beta & 0 & \text{sin}\beta \\ 0 & 1 & 0 \\  -\text{sin}\beta & 0 & \text{cos}\beta \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & \text{cos}\gamma & -\text{sin}\gamma \\ 0 & \text{sin}\gamma & \text{cos}\gamma \end{bmatrix} \tag{15} $$

<br>

- $$ R = \begin{bmatrix} \text{cos}\alpha \ \text{cos}\beta & \text{cos}\alpha \ \text{sin}\beta \ \text{sin}\gamma - \text{sin}\alpha \ \text{cos}\gamma & \text{cos}\alpha \ \text{sin}\beta \ \text{cos}\gamma + \text{sin}\alpha \ \text{sin}\gamma \\ \text{sin}\alpha \ \text{cos}\beta & \text{sin}\alpha \ \text{sin}\beta \ \text{sin}\gamma + \text{cos}\alpha \ \text{cos}\gamma & \text{sin}\alpha \ \text{sin}\beta \ \text{cos}\gamma - \text{cos}\alpha \ \text{sin}\gamma \\ -\text{sin}\beta & \text{cos}\beta \ \text{sin} \gamma & \text{cos}\beta \ \text{cos} \gamma \\ \end{bmatrix} \tag{16} $$

<br>

#### **Change of basis by rotation**

<br>

- 지금까지 살펴본 내용은 한 점 $$ P $$ 가 각 축의 방향으로 회전하였을 때 새로운 위치를 계산하는 방법에 대하여 알아보았습니다.
- 앞으로 살펴볼 내용은 `basis`가 회전할 때 각 좌표들이 어떻게 변경되는 지 살펴보도록 하겠습니다. 앞의 좌표 변환과 유사하지만 다소 차이점이 있으니 그 점을 유의해서 살펴보시면 됩니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그래프를 살펴보면 기존의 $$ X, Y $$ 축이 이루는 평면을 $$ X', Y' $$ 평면이 이루는 축으로 변경을 해야 합니다.
- XY 평면 상의 점 P가 X'Y'평면 상에서 어떤 좌표값을 가지는 지 알면 XY → X'Y'의 변환 관계를 알 수 있습니다.
- 결과는 위 그림의 행렬 식과 같이 회전 변환 행렬의 역행렬을 곱하면 됩니다.

<br>

- $$ \sin{(\alpha)} = \frac{y'}{r}, \cos{(\alpha)} = \frac{x'}{r} \tag{17} $$

- $$ \Rightarrow x'\sin{(\alpha)} = y'\cos{(\alpha)} \tag{18} $$

- $$ x = r\cos{(\theta + \alpha)} = \frac{x'}{\cos{(\alpha)}}\cos{(\theta + \alpha)} \tag{19} $$

<br> 

- 식 (4)의 코사인 법칙을 이용하여 식을 전개합니다.

<br>

- $$ x = \frac{x'}{\cos{(\alpha)}}\cos{(\theta + \alpha)} = \frac{x'}{\cos{(\alpha)}}(\cos{(\theta)}\cos{(\alpha)} - \sin{(\theta)}\sin{(\alpha)}) \tag{20} $$

- $$ x = x'\cos{(\alpha)} - x'\sin{(\alpha)}\frac{\sin{(\theta)}}{\cos{(\alpha)}} \tag{21} $$

<br>

- 식 (21)에 식 (17)을 이용하여 $$ x'\sin{(\alpha)} $$ 을 $$ y'\cos{(\alpha)} $$ 로 대체한다.

<br>

- $$ x =  x'\cos{(\alpha)} -  y'\cos{(\alpha)}\frac{\sin{(\theta)}}{\cos{(\alpha)}} \tag{22} $$

- $$ x =  x'\cos{(\alpha)} -  y'\sin{(\theta)} \tag{23} $$

<br>

- 이 방법과 유사하게 아래 식 (24)를 식 (9)의 sin법칙과 식 (17)을 이용하여 전개하면 식 (25)와 같이 정리 됩니다.

<br>

- $$ y = r\sin{(\theta + \alpha)} = \frac{y'}{\sin{(\alpha)}}\sin{(\theta + \alpha)} \tag{24} $$

- $$ \Rightarrow x'\sin{(\theta)} + y'\cos{(\theta)} \tag{25} $$

<br>

- 따라서 basis를 회전하였을 때, 회전 변환 행렬은 다음과 같이 정리할 수 있습니다.

<br>

- $$ \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} \cos{(\theta)} & -\sin{(\theta)} \\ \sin{(\theta)} & \cos{(\theta)} \end{bmatrix} \begin{bmatrix} x' \\ y' \end{bmatrix} \tag{26} $$

<br>

- 변환의 최종 목적은 (x, y) → (x', y')로 변환하기 위한 행렬을 찾는 것이므로 아래와 같이 행렬식을 변경합니다.

<br>

- $$ \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos{(\theta)} & -\sin{(\theta)} \\ \sin{(\theta)} & \cos{(\theta)} \end{bmatrix}^{-1} \begin{bmatrix} x \\ y \end{bmatrix} \tag{27} $$

<br>

- 식 (27)과 같이 basis 행렬의 변환은 기존의 회전 변환 행렬을 역행렬 한 것으로 확인할 수 있습니다. 따라서 카메라의 `extrinsic`의 요소인 Rotation 정보를 안다면 역행렬을 이용하여 카메라 좌표계의 basis와 실제 세계의 basis 간의 변환을 할 수 있습니다.

<br>

- 지금까지 특정 점이 회전하는 경우와 기저(basis)가 회전하는 경우에 대하여 살펴보았습니다. 그러면 특정 점이나 기저가 이동 (translation)하는 경우에 대하여 살펴보도록 하겠습니다.

<br>

#### **Change of coordinates by translation**

<br>

- 먼저 translation에 의하여 특점 점의 좌표값이 바뀌는 경우에 대하여 살펴보도록 하겠습니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/8.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 그림과 같이 점 $$ P $$ 가 점 $$ P' $$ 로 이동한 경우입니다. 아래 식과 같이 이동할 수 있습니다.

<br>

- $$ x' = x + a, y' = y + b \tag{28} $$

<br>

- 위 식을 행렬 곱으로 나타내려면 차원이 맞지 않기 때문에 차원을 하나 늘려주는 트릭을 통해 행렬의 곱으로 나타낼 수 있습니다. 이와 같이 행렬의 곱셈으로 나타내려는 이유는 앞에서 살펴본 rotation과 translation을 한번에 표현하기 위함입니다.

<br>

- $$ \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix}1 & 0 & a \\ 0 & 1 & b \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1\end{bmatrix} \tag{29} $$

<br>

- 위 식과 같이 차원을 추가하여 좌표를 표현한는 것을 `homogeneous coordinates`라고 합니다. 행렬의 곱으로만 표현된 형태를 뜻합니다.
- 만약 위 식에서 $$ x, y $$ 좌표값을 구하고 싶으면 다음과 같이 마지막 상수항을 나눠서 구할 수 있습니다.

<br>

- $$ \begin{bmatrix} x & y & 1 \end{bmatrix} \approx \begin{bmatrix} x/1 & y/1 \end{bmatrix} = \begin{bmatrix} x & y \end{bmatrix} \tag{30} $$

<br>

- 실제 연산을 할 때에는 homogeneous coordinate 상에서 연산을 하고 좌표를 구할 때에는 식 (30)을 이용하여 구합니다. 상세 내용은 글 이후에서 다룰 예정입니다.

<br>

#### **Change of coordinates by translation**

<br>

- 앞에서 다룬 것과 마찬가지로 XY 평면을 X'Y' 평면으로 translation 해보도록 하겠습니다. 두 기전 간의 관계는 점 $$ P $$ 를 고정으로 둔 다음에 좌표가 어떻게 바뀌는 지 확인하여 살펴보겠습니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/9.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- $$ x' = x - a, y' = y - b \tag{31} $$

- $$ \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix}1 & 0 & -a \\ 0 & 1 & -b \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1\end{bmatrix} \tag{32} $$

<br>

- 식 (27)의 rotation에서 살펴보았듯이 좌표 위치 변환을 위한 linear transformation과 basis transformation 간에 `역(inverse) 관계`가 있었듯이 translation에서도 역관계가 있음을 확인할 수 있습니다. 

<br>
<center><img src="../assets/img/vision/concept/calibration/10.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 풀이 과정을 통하여 식 (29)의 우변의 행렬과 식 (32)의 우변의 행렬은 역행렬 관계임을 알 수 있습니다.
- 정리하면 **rotation과 translation에서의 basis transformation과 coordinate transformation에는 역행렬 관계가 있습니다.**

<br>

#### **Extrinsic Camera Matrix**

<br>

- 지금까지 `rotation`과 `translation`을 각각 살펴보았습니다. 그러면 `homogeneous coordinate` 형태로 나타내어 본 이유가 한번에 행렬곱으로 연산하기 위함이었듯이 행렬 곱으로 나타내어 보겠습니다.

<br>

- $$ E^{-1} = \begin{bmatrix} R & T \\ 0^{T} & 1 \end{bmatrix} = \begin{bmatrix} 1 & T \\ 0^{T} & 1  \end{bmatrix} \begin{bmatrix} R & 0 \\ 0^{T} & 1  \end{bmatrix} \tag{33} $$

<br>

- 위 식에서 $$ R $$ 은 rotation을 의미하고 $$ T $$ 는 translation을 의미합니다. $$ R $$ 은 (2, 2), (3, 3)과 같은 정사각행렬의 크기를 가집니다. 이 때 차원이 결정되면 $$ R $$의 차원과 동일한 차원의 $$ T $$ 열벡터가 크기 2, 3과 같은 사이즈를 가지게 됩니다. $$ 0^{T} $$ 는 열벡터를 행벡터 형태로 나타내기 위함입니다.
- 3차원 공간에서의 rotation과 translation을 위한 행렬에서 $$ R $$ 은 (3, 3)의 크기의 행렬을 가지고 $$ T $$ 는 (3, 1)의 크기의 열벡터를 가지므로 최종적으로 (4, 4) 크기의 행렬이 됩니다.
- 어떤 점 $$ P \to P' $$ 로 `coordinate transform` 할 때 사용한 행렬을 $$ A $$ 라고 하면 $$ A^{-1} $$ 은 `basis transformation` 이라고 하였습니다. 따라서 식 (33)의 행렬의 역행렬이 `basis transformation`을 위한 행렬이 됩니다. 이 `basis transformation`을 `extrinsic camera matrix` $$ E $$ 라고 합니다.

<br>

- $$ P^{C} = E \times P^{W} \tag{34} $$

<br>

- 식 (34)와 같이 extrinsic camera matrix $$ E $$ 를 이용하여 `world coordinate system`에서 `camera coordinate system`으로의 기저 변환을 할 수 있습니다.

<br>

#### **Degrees of Freedom**

<br>

- 지금까지 살펴본 `Extrinsic Camera matrix`를 선언할 때, 필요한 파라미터는 6가지가 있었습니다. X, Y, Z 축으로 얼만큼 회전하였는 지와 X, Y, Z 축으로 부터 얼마 만큼의 translation이 발생하였는 지 입니다.
- 이 필요한 파라미터를 `DoF(Degree of Freedom)`이라고 하며 따라서 Extrinsic을 구할 때에는 6개의 DoF가 필요하다고 말합니다.

<br>

- 지금까지 배운 내용을 파이썬으로 실습해 보도록 하겠습니다. 아래 링크의 예제는 world coordinate system → camera coordinate system으로 기저 변환이 되었을 때, $$ y $$ 축으로 45도 회전과 -8만큼 translation이 발생하였다고 가정하고 변환하였습니다.
- 그래프 출력 결과는 colab에서 생성이 안되어 local의 jupyter notebook에서 실행하시길 바랍니다.

<br>

- colab 링크 : [Image formation and camera extrinsics](https://colab.research.google.com/drive/1PKCBW46wrPyK-V9gMfmEX7oc-uowf9l0?usp=sharing)

<br>
<center><img src="../assets/img/vision/concept/calibration/11.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/concept/calibration/12.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 먼저 결과부터 살펴보면 기존의 파란색 평면 (world 좌표계)이 Y축 (초록색 축)을 기준으로 45도 rotation과 Y축 기준으로 -8 만큼 translation이 발생한 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/vision/concept/calibration/13.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 코드 부분에서 `R`과 `T`는 각각 Y축 방향으로 45도 회전과 -8만큼 translation이 발생함을 나타낸 것입니다.
- `R_`과 `T_`는 

<br>

## **Camera Intrinsic Matrix with Example in Python**

<br>


<br>

## **Find the Minimum Stretching Direction of Positive Definite Matrices**

<br>


<br>

## **Camera Calibration with Example in Python**

<br>


<br>


