---
layout: post
title: Digging Into Self-Supervised Monocular Depth Estimation (monodepth2)
date: 2021-06-01 00:00:00
img: vision/depth/monodepth2/0.png
categories: [vision-depth] 
tags: [depth estimation, monodepth2] # add tag
---

<br>

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>

- 논문 : https://arxiv.org/abs/1806.01260
- 깃헙 : https://github.com/nianticlabs/monodepth2
- 참조 : https://towardsdatascience.com/depth-prediction-autonomous-driving-18d05ff25dd6

<br>

- 이번 글은 `monodepth2`에 대한 내용으로 `monodepth1`에서 사용한 Left Right Consistency를 위한 스테레오 카메라의 학습 데이터 문제를 개선하여 동일한 시점의 left, right 영상이 아닌 **연속된 단안 카메라 이미지 ($$ I_{t-1}, I_{t}, I_{t+1} $$) 3장을 이용하여 딥러닝 모델을 학습**하여 단안 카메라의 깊이를 추정 하는 내용의 논문 리뷰 입니다.

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

- 여기 까지 `monodepth2` 논문의 대략적인 방법론에 대하여 살펴보았습니다. 지금부터 

<br>

## **. Introduction**
## **. Related Work**
## **. Method**
## **. Experiments**
## **. Conclusion**
## **. Supplementary Material**
## **ytorch Code**




- 내가 설명할 최첨단 방법은 깊이를 측정하기 위해 한 프레임에서 다음 프레임으로 픽셀의 불일치 또는 차이를 사용하는 감독되지 않은 딥 러닝 접근 방식입니다.

<br>
<center><img src="../assets/img/vision/depth/monodepth2/1.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/2.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/3.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/4.png" alt="Drawing" style="width: 600px;"/></center>
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
<center><img src="../assets/img/vision/depth/monodepth2/11.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/12.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/13.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/14.png" alt="Drawing" style="width: 600px;"/></center>
<br>

<br>
<center><img src="../assets/img/vision/depth/monodepth2/15.png" alt="Drawing" style="width: 600px;"/></center>
<br>

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

[Depth Estimation 관련 글 목차](https://gaussian37.github.io/vision-depth-table/)

<br>