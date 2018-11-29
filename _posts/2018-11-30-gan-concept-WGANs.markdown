---
layout: post
title: WGANs(Wasserstein GAN)  
date: 2018-11-30 03:40:00
img: gan/concept/wgan/wgan.PNG
categories: [gan-concept] 
tags: [gan, wgan, Wasserstein GAN] # add tag
---

레퍼런스 : 

+ 논문 : https://arxiv.org/abs/1701.07875
+ 사이트
    + https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html#use-wasserstein-distance-as-gan-loss-function
    + https://www.alexirpan.com/2017/02/22/wasserstein-gan.html
    + https://vincentherrmann.github.io/blog/wasserstein/
    + https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
     

이번 글에서는 `Wasserstein GAN`에 대하여 다루어 보려고 합니다.
먼저 핵심 내용을 간략하게 말씀 드린 후 좀 더 자세히 알아보고 마지막으로는 논문 자체를 하나 하나 분석해서 리뷰해 보도록 하겠습니다.

### Earth Mover's Distance
---

이산 확률 분포에서는 `Wasserstein distance`는 **EMD(Earth Mover's Distance)** 라고도 불립니다.
만약 
If we imagine the distributions as different heaps of a certain amount of earth, 




