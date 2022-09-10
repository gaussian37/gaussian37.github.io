---
layout: post
title: Huber Loss와 Berhu (Reverse Huber) Loss (A robust hybrid of lasso and ridge regression)
date: 2021-02-17 00:00:00
img: ml/concept/huber_berhu_loss/0.png
categories: [ml-concept] 
tags: [huber loss, berhu loss, l1 loss, l2 loss] # add tag
---

<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>

- 참조 : https://artowen.su.domains/reports/hhu.pdf
- 참조 : https://arxiv.org/pdf/1606.00373.pdf
- 참조 : https://velog.io/@gjtang/Huber-Loss%EB%9E%80
- 참조 : https://github.com/abduallahmohamed/reversehuberloss/blob/master/rhuloss.py
- 참조 : https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

<br>

- 이번 글에서는 L1 Loss와 L2 Loss의 단점을 개선하고자 두 Loss를 조합하여 사용하는 `Huber Loss`와 `Berhu (Reverse Huber) Loss`를 다루어 보도록 하겠습니다.

<br>



<br>

[deep learning 관련 글 목차](https://gaussian37.github.io/dl-concept-table/)

<br>