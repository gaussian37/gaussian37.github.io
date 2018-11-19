---
layout: post
title: PR104 - Vid2Vid (Video-to-Video synthesis)  
date: 2018-11-20 03:40:00
img: gan/concept/vid2vid/vid2vid.PNG
categories: [gan-concept] 
tags: [gan, PR12, vid2vid] # add tag
---

이 블로그 내용은 PR12의 vid2vid 내용을 기반으로 작성하였습니다.
좋은 강의해 주신 김태수님께 감사 드립니다.

레퍼런스 : 

- 논문 : https://arxiv.org/abs/1808.06601 
- 강의 : https://www.youtube.com/watch?v=WxeeqxqnRyE&feature=youtu.be
- 자료 : https://www.slideshare.net/TaesuKim3/pr12-20180916-videotovideosynthesistaesu

![vid2vid_ex](../assets/img/gan/concept/vid2vid/vid2vid_ex.PNG)

<br> 

vid2vid 논문은 비디오를 다른 비디오로 변환하는 역할을 합니다.
따라서 위와 같이 Semantic 영상을 실제 주행영상으로 변환도 가능합니다.

![vid2vid_ex2](../assets/img/gan/concept/vid2vid/vid2vid_ex2.PNG)

<br>

또한 위와 같이 edge 영상을 이용하여 사람 같은 영상을 만들어 내기도 합니다.
다양한 응용이 가능하며 마치 pix2pix의 video 버전이라고 보셔도 됩니다.

`vid2vid` 논문은 `pix2pix` 논문을 기반으로 이루어져 있습니다. pix2pix 관련 논문은 `PR-65` 내용에도 나와있으니 참조하시면 될 것 같습니다.
간단하게 `pix2pix`에 대하여 알아보고 넘어가도록 하겠습니다.

![pix2pix_ex](../assets/img/gan/concept/vid2vid/pix2pix_ex.PNG)

<br>

`pix2pix`: Image-to-image translation with conditional adversarial networks 에서 알 수 있듯이, `cGAN`을 이용하여
하나의 이미지를 다른 이미지로 변형하는 기법을 다룬 논문이었습니다. 

$$ cGAN : {x, z} → y $$ 즉, 입력으로 (x, z)를 출력으로 y를 주게 되고 x, y, z의 뜻은 아래와 같습니다.  

+ x : observed image (condition)
+ z : random noise vector
+ y : generated output

`pix2pix` 에서는 2가지 `Loss`를 사용하였다는 것이 기존의 `GANs`와 차이점이 있었습니다.

+ GAN Loss
    - 정의하면 $$ \mathcal L_{cGAN} (G, D) = \mathbb E_{x, y \sim p_{data}(x,y)}[ logD(x,y) ] +  \mathbb E_{x  \sim p_{data}(x), z \sim p_{z}(z)}[log(1 - D(x, G(x, z)))] $$
    
+ L1 Loss (Enforce correctness at Low Frequencies)
    - 정의하면 $$ \mathcal L_{L1}(G) = \mathbb E_{x, y  \sim P_{data}(x, y), z \sim P_{z}(z)}[||y - G(x, z)||_{1}] $$
    
따라서 `pix2pix`에서의 종합적인 Loss는 다음과 같았습니다.

$$ G^{*} = arg \min_{G} \max_{D} \mathcal L_{cGAN}(G, D) + \lambda \mathcal L_{L1}(G) $$


...작성중...
    
