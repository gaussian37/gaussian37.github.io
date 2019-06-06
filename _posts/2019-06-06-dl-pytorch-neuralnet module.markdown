---
layout: post
title: 신경망의 모듈화
date: 2019-06-06 01:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, module] # add tag
---

+ 파이토치에서는 직접 신경망 계층을 정의할 수 있습니다. 
+ OOP에서 자체 클래스를 만들 수 있는 것처럼 자체 신경망 계층을 만들어서 재사용하거나 이것을 이용해서 더 복잡한 신경망을 만들 수도 있습니다.

<br>

### 자체 신경망 만들기

+ 파이토치에서 자체 신경망 계층(커스텀 계층)을 만들려면 `nn.Module`을 상속해서 클래스를 정의하면 됩니다.
+ `nn.Module`은 모든 계층의 기반 클래스입니다.
+ 커스텀 계층을 만들 때 forward 메서드를 구현하면 자동 미분까지 가능해집니다.
    + nn.Module의 `__call__` 메서드는 내부에서 forward 메서드를 사용하고 있으므로 model(x)형식이 가능합니다.
+ 아래 코드는 ReLu와 Dropout을 사용한 MLP 입니다.

```python

```
    
