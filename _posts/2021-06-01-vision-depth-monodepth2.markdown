---
layout: post
title: Monodepth 2 (Digging Into Self-Supervised Monocular Depth Estimation )
date: 2021-06-01 00:00:00
img: vision/depth/monodepth2/0.png
categories: [vision-depth] 
tags: [depth estimation, monodepth2, 모노 뎁스 2, 모노뎁스] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 논문 : https://arxiv.org/abs/1806.01260
- 깃헙 : https://github.com/nianticlabs/monodepth2
- 참조 : https://towardsdatascience.com/depth-prediction-autonomous-driving-18d05ff25dd6

<br>

- 이번 글은 `monodepth2`에 대한 내용으로 `monodepth1`에서 사용한 Left Right Consistency를 위한 스테레오 카메라의 학습 데이터 문제를 개선하여 동일한 시점의 left, right 영상이 아닌 **연속된 단안 카메라 이미지 ($$ I_{t-1}, I_{t}, I_{t+1} $$) 3장을 이용하여 딥러닝 모델을 학습**하여 단안 카메라의 깊이를 추정 하는 내용의 논문 리뷰 입니다.
- 현재 단안 카메라를 이용하여 `Self-supervised` 방식으로 depth를 추정하는 많은 방식들이 다루어 지고 있고 그 모델들의 기본이 되는 모델 중 하나가 monodepth2이며 코드 공개와 방법이 자세히 설명되어 있는 몇 안되는 모델 중 하나로 이번 글을 통하여 내용을 깊게 확인해 보고자 합니다.

<br>

## **목차**

<br>

- ### 전체 내용 요약
- ### Abstract
- ### 1. Introduction
- ### 2. Related Work
- ### 3. Method
- ### 4. Experiments
- ### 5. Conclusion
- ### 6. Supplementary Material
- ### Pytorch Code

<br>

## **전체 내용 요약**

<br>

- 이번 글에서 다루는 방법은 단안 영상에서 depth를 추정하기 위해 한 프레임에서 다음 프레임으로 **픽셀의 불일치 또는 차이**를 사용하는 Unsupervised Learning (Self-Supervised) 방식의 딥 러닝 접근 방식입니다.
- `monodepth2`에서는 단일 프레임에서 depth를 예측하기 위해 depth 및 pose를 예측하는 네트워크의 조합을 사용합니다. 이 두 개의 네트워크를 훈련하기 위해 연속적인 프레임과 여러 손실 함수를 이용합니다.
- 이 학습 방법은 학습을 위한 정답 (Ground Truth) 데이터셋이 필요하지 않습니다. 대신 이미지 시퀀스에서 연속적인 프레임 (t-1, t, t+1 프레임)을 사용하여 학습 하고 학습을 제한 (Constrain Learning)하기 위해 pose 추정 네트워크를 사용합니다.
- 모델은 `입력 이미지`와 **pose 네트워크 및 depth 네트워크의 출력**에서 `재구성된 이미지` 간의 **차이에 대해 학습**됩니다. 재구성 과정은 이후에 다시 설명 드리겠습니다.

<br>

- 이와 같은 학습 구조에서 `monodepth2`에서 제안하는 주요 Contribution은 다음 3가지 입니다.

<br>

- ① 중요하지 않은 픽셀에서의 계산을 제거하는 auto-masking 방법
- ② Depth Map을 이용한 photometric reconstruction error를 구하는 변경된 방법
- ③ Multi-scale depth estimation

<br>
<center><img src="../assets/img/vision/depth/monodepth2/0_1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이 논문의 접근 방식은 depth 네트워크와 pose 네트워크를 사용합니다. depth 네트워크는 고전적인 U-Net `인코더`-`디코더` 아키텍처입니다. 인코더는 pre-trained된 ResNet 모델이고 depth 디코더는 시그모이드 출력을 depth로 변환하는 작업입니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/0_2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같은 6 DoF (Degree of Freedom)의 상대 pose 또는 rotation 및 translation을 예측하기 위해 입력으로 두 개의 컬러 이미지를 받는 ResNet18의 pose 네트워크를 사용합니다. pose 네트워크는 일반적인 stereo 이미지 쌍(pair)이 아닌 시간적으로 연속된 이미지 쌍(pair)을 사용합니다. 이것은 연속적인 다른 이미지(t-1, t 번째 프레임 이미지)의 관점에서 대상 이미지 (t번째 이미지)의 모양을 예측합니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/0_3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 전체 학습 과정은 위 그림과 같습니다. Depth Network는 Depth를 추정하고 Pose Network는 6-DoF relative pose를 추정합니다. 최종적으로는 appearance loss에 사용되며 이와 관련된 내용을 차례대로 살펴보도록 하겠습니다.

<br>

##### *Photometric Reconstruction Error**

<br>

- `target 이미지`( $$ I_{t} $$ ) 는 프레임 t에 있고 예측에 사용되는 이미지는 프레임 이전 또는 이후 프레임이 될 수 있으므로 $$ I_{t+1} $$ 또는 $$ I_{t-1} $$ 입니다. Loss는 `target 이미지`와 `reconstruction`된 `target 이미지` (source → target reconstruction) 간의 유사도를 기반으로 계산됩니다.
- `reconstruction` 프로세스는 pose 네트워크를 사용하여 source 프레임인 $$ I_{t+1} $$ 또는 $$ I_{t-1} $$ 에서 target 프레임인 $$ I_{t} $$ 로 변환하는 변환 행렬을 계산하는 것으로 시작됩니다. 이것은 rotation 및 translation에 대한 정보를 사용하여 source 프레임에서 target 프레임으로의 매핑을 계산한다는 것을 의미합니다.
- 그런 다음 `reconstructed target 이미지`를 얻기 위하여 `target 이미지`와 depth 네트워크에서 예측된 depth map과 pose 네트워크를 통해 얻은 변환 행렬을 사용합니다.

- 이 프로세스에서는 깊이 맵을 먼저 3D 포인트 클라우드로 변환한 다음 카메라 내장 기능을 사용하여 3D 위치를 2D 포인트로 변환해야 합니다. 결과 포인트는 대상 이미지에서 쌍선형으로 보간하기 위한 샘플링 그리드로 사용됩니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/0_4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 위 Loss의 목표는 target 이미지와 pose와 depth 정보를 이용하여 생성한 reconstructed target 이미지의 차이를 줄이는 것입니다.

<br>

- 여기 까지 `monodepth2` 논문의 대략적인 방법론에 대하여 살펴보았습니다. 그러면 논문 내용을 차례대로 살펴보도록 하겠습니다.

<br>

## **Abstract**

<br>
<center><img src="../assets/img/vision/depth/monodepth2/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 픽셀 단위로 깊이 정보를 가진 정답 데이터를 얻는 것은 굉장히 어려우므로 이러한 한계를 극복하기 위하여 `self-supervised` 방식 즉, 정답 데이터가 없는 unsupervised 방식의 학습 방식이 많이 연구가 되고 있습니다.
- 이번 글에서 다루는 monodepth2는 self-supervised 방식의 깊이 추정 방식의 모델이며 정성 및 정량 모두에서 기존의 다른 모델에 비해 개선된 성능을 보였다는 점에서 의의가 있습니다.
- 기존의 self-supervised 기반의 깊이 추정을 위해서는 복잡한 모델 구조와 Loss 함수 그리고 disparity를 구하기 위한 이미지 생성 모델이 필요한데 monodepth2에서는 이러한 구조를 단순화 시킨 것에 의의가 있으며 성능 개선을 한 주요 내용은 크게 아래 3가지와 같습니다.
- ① `minimum reprojection loss`: 기존의 average reprojection loss를 단순히 minimum으로 바꿈으로써 연속된 Frame 간 occlusion이 발생하더라도 강건해질 수 있도록 구조를 변경하였습니다.
- ② `full-resolution multi-scale sampling method` : depth estimation 결과에서 [visual artifact](https://www.howtogeek.com/740279/what-are-visual-artifacts/)와 같은 현상으로 depth 결과가 나타나는 데 이러한 문제를 개선하기 위하여 multi-scale에서 depth 정보를 추출할 수 있도록 하는 구조를 도입하였습니다.
- ③ `auto-masking loss` : disparity를 구하기 위해 카메라 pose를 예측할 때, 전제 조건은 카메라는 프레임 간 이동이 발생하지만 그 이외의 물체는 정지되어 있어야 한다는 점입니다. 이러한 전제 조건의 위배되는 경우 중 하나가 카메라의 이동이 발생하지 않아서 물체가 완전히 정지되어 있는 경우입니다. 이와 같은 경우에는 diparity를 구할 수가 없기 때문에 Frame간 변화가 나타나지 않은 경우 필터링하여 학습하지 않도록 하는 방법을 도입하였습니다.

<br>

## **1. Introduction**

<br>

- monodepth2에서는 Monocular video를 입력으로 사용한 경우와 stereo image pair를 입력으로 사용한 경우 2가지에 대한 실험이 있으나 Monocular video를 입력으로 사용한 경우에만 집중해서 이 글을 다룰 예정입니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 일반적으로 깊이는 스테레오 카메라의 두 영상의 `triangulation`을 통하여 추정하게 되지만 본 글에서는 단일 컬러 이미지로 부터 깊이를 추정하는 것을 학습하고자 합니다.
- 라이다 센서 없이 컬러 이미지로 부터 깊이를 추정할 수 있으면 사용처가 다양하며 특히 큰 볼륨의 라벨링이 없는 이미지 데이터셋으로 부터 깊이를 학습하는 것은 데이터 준비 측면에서 굉장히 효율적인 방법입니다.
- 따라서 현실적으로 대량의 데이터 셋을 구하기 어려운 깊이에 대한 정답 데이터를 구하는 대신에 `self-supervised` 방식의 접근 방법이 연구되고 있고 그 방법은 `stereo pairs` 또는 `monocular video`를 이용하는 방법이 있습니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- `stereo pair`, `monocular video` 2가지 방식 중에서 `monocular video` 방식은 단안 카메라 하나를 통해서 학습 데이터를 만들 수 있어서 데이터 준비 측면에서 더 쉬운 방식이며 센서를 구성하기도 쉽습니다. 하지만 Frame 간 자차 이동 발생 (ego motion) 정도를 알 수 있어야 학습 시 고려할 수 있습니다. 따라서 `pose estimation network`를 통하여 ego motion을 예측해야 합니다.
- `pose estimation`은 연속적인 frame을 입력으로 받고 frame 간 이동된 카메라의 Rotation, Translation을 반영한 transformation matrix를 출력합니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 앞에서 다룬 바와 같이 self-supervised 학습 방식을 잘 적용하기 위하여 크게 3가지 loss를 추가하여 사용 하였습니다.
- `novel appearance matching loss` : monocular video에서 Frame 간 물체가 가려지는 문제를 다루기 위해 도입되었습니다.
- `auto masking` : frame간 움직임이 있어야 disparity를 구할 수 있는데 frame 간 움직임이 없는 경우 이를 필터링 하기 위한 masking 기법입니다.
- `multi-scale appearance matching loss` : 입력 이미지의 다양한 스케일 (1배, 1/2배, 1/4배, 1/8배)에서 각각 깊이를 추정하여 학습함으로써 다양한 해상도의 깊이를 학습함을 통해 visual artifact 문제를 개선할 수 있었습니다.
- 이러한 기법을 모두 이용하여 self-supervised 방식의 모델 중에서 좋은 성과를 낼 수 있었습니다.

<br>

## **2. Related Work**

<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/6.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/7.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/8.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/9.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/10.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

## **3. Method**

<br>
<center><img src="../assets/img/vision/depth/monodepth2/11.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 이번 절에서는 monodepth2에서는 깊이를 추정하는 네트워크가 컬러 이미지 $$ I_{t} $$ 를 입력 받아서 depth map $$ D_{t} $$ 를 출력하는 지 방법에 대하여 다루어 보도록 하겠습니다.
- 관련 내용으로는 뎁스 추정을 위한 네트워크와 Self-Supervised로 학습하기 위한 Loss가 이에 해당합니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/12.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- Self-Supervised 방식은 전통적으로 사용하는 Disparity를 구하는 방식을 딥러닝을 이용하는 방식입니다. Disparity를 구하기 위해서는 동일한 Scene에 대하여 2개의 View에 해당하는 이미지가 필요합니다. 
- 따라서 monodepth2에서는 앞에서 설명한 바와 같이 `monocular video`를 이용하여 같은 Scene에 대하여 2개의 View를 구할 것입니다. 기준이 $$ I_{t} $$ 라고 하면 $$ I_{t-1} $$ 또는 $$ I_{t+1} $$ 를 이용하여 $$ I_{t} $$ Scene에 해당하는 영상을 네트워크 학습을 통하여 예측합니다.
- 따라서 네트워크는 학습 파라미터를 이용하여 타겟 이미지인 $$ I_{t} $$ 시점의 Scene을 다른 시점의 이미지로 부터 생성하여 disparity 및 depth를 구하게 됩니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/13.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 따라서 disparity를 구하여 Depth를 추정하기 위하여 $$ I_{t-1} $$, $$ I_{t+1} $$ 이미지를 $$ I_{t} $$ 의 Scene으로 가능한한 유사하게 `reprojection`을 합니다. 
- 즉, $$ t-1 \to t $$, $$ t \to t+1 $$ 로 바뀜에 따라서 카메라의 위치도 바뀌게 되는데, `Pose Network` 라는 별도의 딥러닝 네트워크를 이용하여 서로 다른 시점의 ($$ t $$ 와 $$ t-1 $$ 또는 $$ t $$ 와 $$ t+1 $$) 영상을 **같은 시점( $$ t $$ ) 으로 카메라 위치를 맞추는 Rotation 및 Translation 행렬을 예측하는 방법을 사용**합니다. 
- 따라서 Pose Network를 통해 구한 R,t 행렬을 이용하여 $$ I_{t-1} \to I_{t} $$ 로 변환하거나 $$ I_{t+1} \to I_{t} $$ 로 변환합니다. 이와 같은 작업을 `reprojection` 이라고 합니다.
- 위 식 (1)을 살펴보면 다음과 같습니다.

<br>

- $$ L_{p} = \sum_{t'} \text{pe}(I_{t}, I_{t'\to t}) \tag{1} $$

<br>

- 여기서 $$ L_{p} $$ 는 `photometric reprojection error`라고 하며 `pe`는 `photometric reconstruction error`를 줄인말입니다. `pe`는 즉, 두 이미지가 얼만큼 다른 지 error를 구하는 것이고 $$ t' $$ 즉, $$ t-1, t+1 $$ 2가지 경우에 대하여 `pe`를 구하였을 때 error의 총합을 $$ L_{p} $$ 로 나타냅니다. monodepth2에서 `pe`는 L1 Loss를 사용합니다.

<br>

- $$ I_{t' \to t} = I_{t'}<\text{proj}(D_{t}, T_{t \to t'}, K)> \tag{2} $$

<br>

- 식 (2)는 이미지를 어떻게 $$ t' \to t $$ 로 시점을 변환하는 지 나타냅니다. 실제 코드를 살펴보면 다음과 같은 절차를 따릅니다.
- ① $$ D_{t} $$ 는 $$ t $$ 시점에서 $$ I_{t} $$ 를 입력으로 받은 Depth Network의 출력을 의미합니다. $$ D_{t} $$ 와 intrinsic $$ K $$ 를 이용하여 깊이 추정 결과를 3D 포인트로 변환합니다.
- ② $$ I_{t}, I_{t'} $$ 를 이용하여 Frame 간 카메라의 Rotation, Translation 관계를 나타내는 $$ T_{t \to t'} $$ 를 예측합니다. 즉 $$ t \to t' $$ 로 카메라의 위치를 변환하는 변환 행렬을 구합니다.
- ③ $$ t $$ 시점에서 구한 3D 포인트를 ②에서 구한 변환 행렬을 이용하여 $$ t' $$ 시점의 3D 포인트로 변환합니다.
- ④ 변환된 $$ t' $$ 시점의 3D 포인트 (X, Y, Z)를 intrinsic $$ K $$ 를 이용하여 2D 이미지 좌표인 (u, v)로 변환합니다. 이 **2D 좌표가 의미하는 것은  $$ I_{t} $$ 의 각 픽셀 좌표가 $$ I_{t'} $$ 에서 어떤 픽셀에 대응되는 지 나타냅니다. 따라서 $$ I_{t'} $$ 의 RGB 값을 예측한 2D 이미지 좌표를 이용하여 가져오면 $$ I_{t'} $$ 를 이용하여 $$ I_{t} $$ 를 복원할 수 있습니다.** 이 때, 사용되는 연산은 [grid_sample](https://gaussian37.github.io/dl-pytorch-snippets/#fgrid_sample-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9-%EC%98%88%EC%A0%9C-1)을 클릭하여 확인하시면 됩니다.
- 이 때, 영상의 복원이 잘된다는 의미는 ①의 Depth Network의 출력인 Disparity (Depth)가 의미있게 출력되었다는 뜻이고 ②의 $$ T_{t \to t'} $$ 또한 의미있게 출력되었다는 뜻입니다.
- 이와 같은 방식으로 $$ I_{t' \to t} $$ 를 추정하는 것은 **카메라의 위치만 변경되고 나머지 환경은 변하지 않았다는 가정을 두기 때문**입니다.

<br>

- 따라서 논문의 식 (2)를 통해 구한 $$ I_{t' \to t} $$ 와 $$ I_{t} $$ 를 논문의 식 (1)을 통해 `Loss`를 구하여 학습하면 두 값이 유사해지도록 학습됩니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/23.png" alt="Drawing" style="width: 1000px;"/></center>
<br>

- `photometric reprojection error`을 구하기 위한 전체 과정을 도식화 하면 위 그림과 같습니다. $$ t' $$ 가 $$ t-1, t+1 $$ 2가지 경우가 있으므로 $$ I_{t} $$ 에 대하여 $$ I_{t-1}, I_{t+1} $$ 각각에 대하여 1번씩 총 2번의 경우에 대하여 에러를 계산하여 학습합니다.
- 이 때, $$ I_{t-1 \to t} $$ 를 생성하기 위한 `grid sampling` 연산 시 sampling 방법은 `bilinear` 방식으로 없는 픽셀에 대하여 interpolation 하여 생성하며 이와 같은 grid sampling 방식은 미분 가능하기 때문에 학습에 사용될 수 있습니다. 
- 식 (1)의 $$ L_{p} $$ 를 구하기 위하여 미분 가능한 이미지를 비교하는 대표적인 방식인 `SSIM (Structural Similarity Index)`과 `L1` Loss를 같이 사용하였고 상대적으로 `SSIM`에 좀 더 높은 가중치를 부여하였습니다. `SSIM`의 상세 내용은 아래 링크를 참조하시기 바랍니다.
    - `SSIM (Structural Similarity Index)` : [https://gaussian37.github.io/vision-concept-ssim/](https://gaussian37.github.io/vision-concept-ssim/)

<br>
<center><img src="../assets/img/vision/depth/monodepth2/13_1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 식 (3)에서는 `edge-aware smootheness` Loss를 추가적으로 도입합니다. `edge-aware smootheness`는 monodepth1에서도 사용이 되었으며 사용 목적은 `이미지 변화` ( $$ \partial_{x}I_{t}, \partial_{y}I_{t} $$ )가 낮은 곳에서 `깊이 변화`가 크면 Loss를 크게 반영합니다. 즉, **이미지 변화가 작으면 깊이의 변화도 작도록 학습합니다.** 이와 같은 Loss를 추가함으로써 깊이 추정에서의 문제점 중 하나인 물체의 경계면에서의 깊이 추정 정확도를 개선합니다.
- 식 (3)에서는 $$ d^{*}_{t} = d_{t} / \bar{d_{t}} $$ 와 같은 `normalization` 형태를 사용함으로써 학습이 잘 되도록 하였습니다. 상세 내용은 아래 코드와 같습니다.

<br>

```python
def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    # disparity의 x 방향 변화량 확인
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    # disparity의 y 방향 변화량 확인
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    # image의 x 방향 변화량 확인
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    # image의 y 방향 변화량 확인
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    # 이미지의 변화가 작을 때, exp 값이 크므로 disparity의 변화가 작아지도록 유도함
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
```

<br>
<center><img src="../assets/img/vision/depth/monodepth2/14.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 만약 monocular sequence를 이용하지 않고 stereo pair를 이용한다면 $$ t' $$ 가 서로 다른 시점이 아닌 같은 시점에서의 다른 카메라의 영상이 됩니다. 그 이외의 학습 방식은 Monocular Sequence를 이용하는 방식과 동일합니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/15.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 지금까지 설명한 Loss를 통하여 Self-Supervised 방식으로 

<br>
<center><img src="../assets/img/vision/depth/monodepth2/16.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/17.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/18.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/19.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/20.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/21.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/22.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/fig1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/fig1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/fig2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/fig3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/fig4.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/fig5.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>

## **4. Experiments**

## **5. Conclusion**

## **6. Supplementary Material**

<br>











<br>

## **Pytorch Code**

<br>

- 지금부터 `monodepth2`의 깃헙 코드에 대한 설명을 해보도록 하겠습니다.

<br>

- 링크 : https://github.com/nianticlabs/monodepth2

<br>

- `monodepth2`의 코드는 `trainer.py`에 해당하는 코드 내용을 이해하면 전반적으로 이해할 수 있습니다. 아래 코드의 설명은 몇가지 조건을 정하여 코드를 설명하며 불필요한 부분은 제거하였습니다.
- 아래 코드 조건은 stereo가 아닌 단안 (mono) 카메라를 통해 얻은 비디오 영상을 이용하여 연속하는 3개의 프레임 ( $$ I_{t-1}, I_{t}, I_{t+1} $$ ) 을 이용하는 옵션을 적용하였으며 추가적으로 라이다 데이터가 있다면 학습에 사용할 수 있도록 코드를 추가하였습니다. (라이다 데이터가 없으면 연속된 프레임으로만 학습합니다.)
- 논문에 따라서 Depth Estimation 모델의 Encoder와 Pose Network의 Encoder를 별도로 분리하는 경우 더 좋은 성능이 있기 때문에 Encoder를 분리하였습니다.

<br>

```python


```




<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>