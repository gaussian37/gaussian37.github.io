---
layout: post
title: VAE(Variational AutoEncoder)
date: 2019-02-25 00:00:00
img: dl/concept/vae/0.png
categories: [dl-concept] 
tags: [deep learning, autoencoder, vae, variational autoencoder] # add tag
---

<br>

- 이번 글에서는 Variational AutoEncoder의 기본적인 내용에 대하여 다루어 보도록 하겠습니다.

<br>

## **목차**

<br>

- ### AutoEncoder의 의미
- ### Variational AutoEncoder의 의미
- ### Variational AutoEncoder의 상세 내용 
- ### Variational AutoEncoder의 구현

<br>

## **AutoEncoder의 의미**

<br>

- AutoEncoder(이하 AE)는 저차원의 representation $$ z $$를 원본 $$ x $$로부터 구하여 스스로 네트워크를 학습하는 방법을 뜻합니다. 이것은 Unsupervised Learning 방법으로 레이블이 필요 없이 원본 데이터를 레이블로 활용합니다.
- AE의 주 목적은 Encoder를 학습하여 유용한 `Feature Extractor`로 사용하는 것입니다. 현재는 잘 쓰지는 않지만 기본 구조를 응용하여 다양한 딥러닝 네트워크에 적용 중에 있습니다. (ex. VAE, U-Net, Stacked Hourglass 등등)

<br>
<center><img src="../assets/img/dl/concept/vae/1.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 먼저 AE의 전체적인 구조는 위 아키텍쳐와 같습니다.
- 입력 $$ x $$와 출력 $$ y $$가 최대한 동일한 값을 가지도록 하는 것이 AE의 목적이며 입력의 정보를 압축하는 구조를 `Encoder`라 하고 압축된 정보를 복원하는 구조를 `Decoder` 그리고 그 사이에 있는 Variable을 `Latent Variable` $$ Z $$ 라고 합니다.

<br>
<center><img src="../assets/img/dl/concept/vae/2.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- AE에서 저차원 표현 $$ z $$는 위 그림과 같이 원본 데이터의 함축적인 의미를 가지도록 학습이 됩니다. 물론 위 예시처럼 물체의 형상, 카메라 좌표, 광원의 정보와 같이 명시적으로 Latent Variable이 저장되지는 않지만 어떤 특징 정보가 저장된다는 점은 일치합니다.
- 이는 다른 머신 러닝 모델에서 **feature**라 불리는 것과 같은 의미이며 **학습 과정에서 겉으로 드러나지 않는 숨겨진 변수**이므로 `Latent Variable` 이라고 불리게 됩니다.

<br>
<center><img src="../assets/img/dl/concept/vae/3.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 따라서 Encoder는 Latent Variable 자체를 만드는 역할을 하고 Decoder는 Latent Variable로 부터 데이터를 생성하는 데 사용됩니다.

<br>
<center><img src="../assets/img/dl/concept/vae/4.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- AE의 Latent Variable은 입력을 압축하여 Feature를 만들기 때문에, `Feature Extractor`의 역할을 할 수 있습니다.
- 따라서 위 그림과 같이 학습이 완료된 AE에서 `Encoder`와 `Latent Variable` 까지만 가져와 Feature $$ Z $$를 추출하고 그 값을 이용하여 다른 머신 러닝 Classifier와 섞어서 사용하곤 하였습니다.

<br>
<center><img src="../assets/img/dl/concept/vae/5.png" alt="Drawing" style="width: 800px;"/></center>
<br>

<br>
<center><img src="../assets/img/dl/concept/vae/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- MNIST 데이터를 이용하여 AE를 사용한 결과 예시는 위와 같습니다. MNIST 데이터로 학습이 완료된 네트워크를 숫자 7, 2 입력을 넣으로 오른쪽의 약간 흐릿하지만 원본과 유사한 데이터가 생성된 것을 확인할 수 있습니다.

<br>

## **Variational AutoEncoder의 의미**

<br>

- Variational AutoEncoder(이하 VAE)와 AE는 많이 다르면서도 유사한 구조를 가집니다. VAE는 다소 복잡하므로 먼저 의미 관점에서 살펴보고 자세한 수식적인 내용은 VAE의 의미를 살펴본 뒤 자세하게 다루도록 하겠습니다.



## **Variational AutoEncoder의 상세 내용 **



## **Variational AutoEncoder의 구현**




