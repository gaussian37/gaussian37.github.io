---
layout: post
title: 영상의 edge detection (Sobel, Canny Edge) 원리와 사용법
date: 2021-02-17 00:00:00
img: vision/concept/edge_detection/0.png
categories: [vision-concept] 
tags: [edge detection, gradient, sobel, canny edge] # add tag
---

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-vision-table/)

<br>

- 이번 글에서는 컴퓨터 비전에서 중요한 특징 추출의 방법을 `edge detection`에 대한 내용을 다루어 보도록 하겠습니다.
- 이 글의 최종 목적은 `auto canny edge detection`을 사용하기 위함이며 이 방식에는 어떤 파라미터도 포함되지 않는 것에 장점이 있습니다.

<br>

## **목차**

<br>

- ### [영상의 미분과 sobel filter](#영상의-미분과-sobel-filter-1)
- ### [gradient와 edge detection](#gradient와-edge-detection-1)
- ### [canny edge detection](#canny-edge-detection-1)
- ### [auto canny edge edtection](#auto-canny-edge-edtection-1)

<br>

## **영상의 미분과 sobel filter**

<br>

- 영상에서의 `edge`란 영상에서 **픽셀의 밝기 값이 급격하게 변하는 부분을 의미**합니다. 일반적으로 배경과 객체 또는 객체와 객체의 경계를 의미합니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/1.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 위 그림과 같이 f 축의 픽셀의 값이 갑자기 변화가 발생하는 지점에서 `edge`가 검출됩니다.
- 왼쪽 상단의 형태는 급격한 형태로 표현한 것이며 실제 영상에서 값이 급격하게 바뀌더라도 우측 상단의 그래프와 같이 어느 정도는 부드럽게 값이 바뀌게 됩니다.
- 좀 더 현실적으로 보면 왼쪽 하단과 같이 어느 정도 노이즈가 섞여 있기 때문에 지글지글한 형태의 그래프를 확인할 수 있습니다. 이와 같은 경우 gaussian blur 등을 이용하면 최종적으로 우측 하단과 같이 부드럽게 값이 바뀌면서 노이즈도 적은 형태의 그래프를 얻을 수 있습니다.
- 만약 픽셀 값이 감소하게 되면 반대 형상의 그래프를 얻을 수 있습니다.

<br>

- 기본적인 `edge 검출 방법`은 영상을 (x, y) 변수의 함수로 간주하였을 때, **함수의 1차 미분 값이 크게 나타나는 부분을 검출**합니다.
- 미분을 이용하면 앞의 그래프의 예시와 같이 값이 급격하게 바뀌는 부분 즉, `변화율`을 찾을 수 있기 때문입니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/2.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 왼쪽 상단의 그래프가 $$ f(x, y) $$ 의 함수 중 $$ x $$ 축만을 고려하여 값의 변화를 살펴본 그래프라고 하면 왼쪽 하단의 그래프는 $$ x $$ 축에 대하여 미분을 하여 변화율을 살펴본 그래프입니다. 
- 그래프의 양 끝 지점에서는 변화가 없기 때문에 변화율은 0이고 $$ a $$ 부근에서만 변화가 발생하기 때문에 $$ a $$ 부근에서만 변화율이 있는 것을 확인할 수 있습니다.
- 단, 디지털 영상에서 이러한 미분 값을 처리할 때에는 연속적인 값이 아닌 이산적인 값이기 때문에 최종적으로 오른쪽 하단과 같은 형태의 이산적인 값을 처리하게 됩니다.
- 위 그래프에서 가로선 방향의 $$ T $$ 값이 있는데 이 값을 `threshold`라고 적용하며 이 값을 기준으로 `edge`를 검출할 수 있습니다. 위 오른쪽 하단 그래프의 예시에서 제공한 $$ T $$ 값을 초과한 값은 3개이며 edge라고 검출된 픽셀이 3개라고 볼 수 있습니다. 따라서 $$ T $$ 값을 크게 설정하거나 작게 설정하는 것에 따라 엣지 검출의 결과가 달라지게 됩니다.

<br>

- 1차 미분을 하는 방법은 흔히 알려져 있는 방식을 사용합니다. 아래는 1차 미분을 근사화 하는 방법이며 기본적으로 3가지 방식이 알려져 있습니다.

<br>

- $$ \text{Forward Difference : } \frac{\partial I}{\partial x} = \frac{I(x + h) - I(x)}{h} $$

- $$ \text{Backward Difference : } \frac{\partial I}{\partial x} = \frac{I(x) - I(x - h)}{h} $$

- $$ \text{Centered Difference : } \frac{\partial I}{\partial x} = \frac{I(x + h) - I(x - h)}{2h} $$

<br>

- 위 3가지 방식 중 마지막의 중앙 차분이 가장 근사화가 잘되는 것으로 알려져 있으므로 edge detection을 위한 방법 또한 중앙 차분을 사용합니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/3.png" alt="Drawing" style="width:400px;"/></center>
<br>

- 따라서 edge detection을 위하여 위 그림과 같은 `미분 마스크`를 사용하게 됩니다.
- 미분 마스크의 -1 부분의 픽셀 값의 위치가 가장 작고 0, 1 방향으로 픽셀 값의 위치가 커집니다. 따라서 미분 마스크를 이용하여 각 픽셀의 영역과 곱한 뒤 모두 더하는 연산을 하면 미분을 근사화 할 수 있습니다.

<br>

- $$ (x_{-1} * -1 + x_{0} * 0 + x_{1} * 1) $$

<br>

- 중앙 차분의 분모 $$ 2h $$ 는 모든 픽셀에 동일하게 곱해지므로 변화율을 계산하기 때문에 생략해도 무관합니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/4.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 흔히 사용하는 마스크의 형태는 위 그림과 같이 $$ x, y $$ 방향 계산에 적용하는 정사각형 형태의 마스크를 사용합니다.
- 위 그림과 같이 정사각형 형태의 마스크를 통해 $$ x, y $$ 방향 모두에 평균값을 적용하여 마스크를 사용하면 노이즈에 좀 더 강건해지는 효과가 있으며 가로 방향과 세로 방향 각각 마스크를 적용함으로써 가로 방향과 세로 방향의 edge를 모두 검출할 수 있습니다.
- 뿐만 아니라 단순히 -1, 0, 1 형태가 아닌 다른 값을 사용하기도 하며 변화율을 어떤 크기로 반영하여 나타낼 지에 따라서 마스크의 값이 달라질 뿐 성격은 모두 동일합니다. `prewitt` 필터는 단순히 평균값을 사용하는 필터이지만 `sobel`이나 `scharr`는 기준 픽셀에 좀 더 큰 가중치를 주는 방식이며 이와 같은 방식을 사용할 때 효과가 더 좋다고 알려져 있습니다. (`sobel`은 1:2:1 의 가중치로 기준 픽셀에 더 큰 가중치가 있습니다.)
- `scharr` 필터는 3:10:3 의 가중치 비율을 가지며 이와 같은 방식이 좀 더 가우시안 분포에 가깝게 필터링 할 수 있다고 제안합니다. 하지만 성능 차이가 크지 않기 때문에 간단한 `sobel`을 많이 사용하는 것이 추세입니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/5.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 위 함수는 `sobel` 필터를 사용하는 방법입니다.
- `dx`, `dy`을 하나는 1, 다른 하나는 0 그리고 `ksize`는 3을 주면 앞에서 살펴본 형태의 3x3 필터를 사용할 수 있습니다. `dx=1, dy=0`를 사용하면 x방향의 편미분이고 `dx=0, dy=1`을 사용하면 y방향의 편미분을 의미합니다.
- `scale`과 `delta`는 최종 결과에 곱하거나 더하는 값을 의미합니다. 이와 같이 값을 변경하는 이유는 시각화하여 볼 때, 보기 어려운 문제가 있기 때문에 사람이 볼 수 있는 영역으로 바꾸기 위함입니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/6.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 가운데 영상은 x 방향의 미분을 적용한 것이고 오른쪽 영상은 y방향의 미분을 적용한 것입니다. 시각화 하기 위하여 그레이스케일 값에서 변화량이 없는 값은 128로 두고 음의 변화량을 가질수록 작은 값 (어두움)을 가지고 양의 변화량을 가질수록 큰 값 (밝음)을 가지도록 시각화 하였습니다.
- 두 결과에서 배경 부분의 기둥을 살펴보면 x 방향에서는 edge가 있는 반면 y 방향에서는 edge가 없는 것을 확인할 수 있습니다. 즉, 기둥은 x 방향에서 변화량이 크기 때문에 x 방향의 그림에서만 edge가 나타난 것입니다. 
- 따라서 실제 edge를 검출할 때에는 x, y 방향 모두 적용한 결과를 이용해야 합니다. 각 x, y 방향의 미분 성분을 이용하여 어떻게 `edge detection`을 하는 지 살펴보도록 하겠습니다.

<br>

## **gradient와 edge detection**

<br>

- 

<br>

## **canny edge detection**

<br>

- 이번 글에서 다루고자 하는 알고리즘은 `canny edge` 알고리즘이며 많은 사람들이 기본적으로 사용하는 `edge detection` 알고리즘입니다.
- 앞에서 다룬 `sobel edge detection`은 두꺼운 형태로 edge를 검출하는 단점을 보입니다. 즉, edge를 정확하게 검출하는 데 한계가 있습니다.
- 좋은 `edge detection`의 조건은 ① `정확한 검출` ② `정확한 위치`, ③ `단일 edge` 조건을 만족해야 합니다. 즉, edge가 아닌 점을 edge로 찾거나 또는 edge인데 edge로 찾지 못하는 확률을 최소화 해야 하며 실제 edge의 중심을 검출하고 하나의 edge는 하나의 점으로 표현해야 합니다.
- ① `정확한 검출` 관련 예시로 조명값이 변화하면 같은 영역이라도 edge 였던 부분이 edge가 아닌 부분이 되기도 합니다. 이런 경우 threshold 값에 민감해집니다. 따라서 이러한 환경에 강건할 수 있도록 알고리즘이 개선되어야 합니다.
- ② `정확한 위치` 와 ③ `단일 edge` 관련 예시로 `sobel filter`를 이용한 edge detection은 edge의 두께가 두껍기 때문에 정확한 edge 위치를 알 기 어렵습니다. 따라서 하나의 edge는 하나의 점으로 표현하는 점이 중요함을 뜻합니다.

<br>

- `canny edge detection`은 총 4가지 단계를 거치게 됩니다.
- ① `가우시안 필터링` 과정을 거칩니다. 목적은 노이즈 제거입니다. 따라서 아래 식의 필터를 이미지 전체에 적용합니다.

<br>

- $$ G_{\sigma}(x, y) = \frac{1}{2\pi \sigma^{2}} \ext{(-\frac{x^{2} + y^{2}}{2\sigma})} $$

<br>

- ② `gradient` 계산 (크기 & 방향)
- `sobel filter`를 이용하면 $$ x, y $$ 방향으로 미분값을 구하고 이 값을 통해 `gradient`의 크기와 방향을 구할 수 있음을 앞에서 확인하였습니다. 크기와 방향 성분은 앞에서 다룬 식과 같습니다.

<br>

- $$ \text{magnitude : } \Vert f \Vert = \sqrt{f_{x}^{2} + f_{y}^{2}}  $$

- $$ \text{phase : } \theta = \tan^{-1}{(\frac{f_{y}}{f_{x}})} $$

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/7.png" alt="Drawing" style="width:400px;"/></center>
<br>

- 이미지의 각 픽셀이 사각형으로 구분되어 졌기 때문에 픽셀의 0도, 45도, 90도, 135도와 대칭되는 영역 까지 그룹으로 묶으면 위 그림과 같이 4가지 색으로 표현하여 구역 구분을 단순화 할 수 있습니다.
- 이 과정을 통해 `gradient`의 크기와 방향 (magnitude와 phase)를 구합니다.

<br>

- ③ `NMS (Non-maximum suppression)`
- 하나의 edge가 여러 개의 픽셀로 표현되는 현상을 없애기 위하여 `gradient` 크기가 `local maximum`인 픽셀만을 edge 픽셀로 설정합니다.
- 따라서 `gradient` 방향에 위치한 두 개의 픽셀을 조사하여 `local maximum`인 지 확인합니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/8.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 위 그림과 같이 `edge`와 `gradient`는 직교입니다. 따라서 `gradient`의 방향을 기준으로 `NMS`를 적용하기 때문에 단일 `edge`를 선택할 수 있습니다. 즉, `sobel`만을 이용하였을 때 나타나는 두꺼운 edge 문제를 개선할 수 있습니다.

<br>

- ④ `Hysteresis edge tracking`
- 마지막으로 사용되는 방법에서 `cany edge` 방식의 파라미터가 추가 됩니다. `threshold low`, `threshold high`인 두가지 임계값입니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/9.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 위 그림과 같이 두 개의 임계값을 사용하며 edge로 판별할 수 있는 기준이 두가지 영역으로 구분됩니다.
- 먼저 $$ T_{High} $$ 이상의 `gradient`를 가지는 픽셀은 edge로 판별 됩니다. $$ T_{High} $$ 이상의 `gradient`를 가지는 edge를 `strong edge`라고 합니다.
- 반면 $$ T_{Low} $$ 이하의 `gradient`를 가지는 픽셀은 edge가 아닌 것으로 판별합니다.
- 마지막으로 $$ T_{Low} $$ 와 $$ T_{High} $$ 사이 (hysterisis 구간)의 `gradient`를 가지는 픽셀들을 계속 이어 갔을 때, $$ T_{High} $$ 이상의 `gradient`의 값을 가지는 픽셀과 연결이 된다면hysterisis 구간의 픽셀을 모두 edge로 판별합니다. 이 값들을 `weak edge`라고 합니다.
- 이와 같은 방식을 사용하는 이유는 앞에서 말한 예시와 같이 조명 등의 환경에 따라서 엣지의 검출 조건이 바뀔 수 있으며 엣지의 특성 상 **갑자기 나타나지 않고 엣지는 연결되어 있다는 것을 이용**하여 기존 문제를 개선하기 위한 아이디어 입니다.
- 따라서 위 그림의 가장 왼쪽 예시는 hysterisis 구간의 픽셀들이 연속적으로 `strong edge`들과 연결되어 있으므로 `weak edge`로 판별합니다. 반면 가운데 예시에서는 hysterisis 구간의 픽셀들이 `strong edge`들과 연결되지 못하므로 edge로 판별하지 않습니다. 가장 오른쪽 예시에서는 hysterisis 구간의 픽셀들이 양쪽 끝에 연결되어 `weak edge`로 판별이 났으나 hysterisis 구간을 벗어나 gradient가 작아진 부분은 edge가 아님을 확인할 수 있습니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/10.png" alt="Drawing" style="width:1200px;"/></center>
<br>

- 이 영상 예시를 보면 입력 값에 ① sobel을 적용하고 ② NMS를 적용 후 ③ Hysterisis Edge Tracking을 적용한 결과입니다. `NMS`를 적용하면서 edge의 굵기가 얇아졌고 `Hysterisis Edge Tracking`을 적용하면서 노이즈 들이 사라진 것을 볼 수 있습니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/11.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 실제 `canny edge detection`을 적용하기 위한 함수 사용법은 위 그림과 같습니다. `threshold1, threshold2` 각각은 비율을 1:2 또는 1:3을 가지도록 설정하면 적당한 수준입니다. 이미지에 따라서 원하는 edge를 얻으려면 어느 수준으로 정해야 하는 지 직접 살펴보는 것이 좋습니다.

<br> 
<center><img src="../assets/img/vision/concept/edge_detection/12.png" alt="Drawing" style="width:800px;"/></center>
<br>

- 위 예시와 같이 간단하게 `canny edge detection`을 적용할 수 있습니다.
- 노이즈를 더 깔끔하게 제거하고 싶으면 입력 영상에 가우시안 필터를 적용하면 효과가 좋습니다.

<br>

## **auto canny edge edtection**

<br>

- `canny edge detection`의 성능은 좋지만 `threshold1, threshold2`를 정해야 한다는 단점이 있습니다. 다양한 환경의 사진을 매번 보면서 매뉴얼한 방식으로 두 값을 정해야 하는 것은 비효율적입니다.
- 따라서 이미지 픽셀 값을 이용하여 효과적으로 두 값을 정하는 방식을 통해 이미지를 매번 확인하지 않고 `edge detection`을 하고자 합니다.
- 방법은 간단히 중앙값을 기준으로 1 sigma 구간의 hysterisis를 적용하는 것입니다. 따라서 중앙값 기준 -1 sigma가 하한값이 되고 +1 sigma가 상한값이 됩니다. 
- 1 sigma의 크기는 중앙값 대비 0.33의 편차 (즉, 0.67, 1.33)이기 때문에 하한값과 상한값은 약 2배 차이가 나며 자동적으로 하한값의 크기도 결정됩니다.

<br>

- 아래는 위 설명을 코드로 구현한 것이며 입력 이미지에 가우시안 블러 또한 적용하였습니다.

<br>

```python
def auto_canny(image, sigma=0.33):
    image = cv2.GaussianBlur(image, (3, 3), 0)
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

image = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
edge = auto_canny(image)
```

<br>

[Vision 관련 글 목차](https://gaussian37.github.io/vision-vision-table/)

<br>