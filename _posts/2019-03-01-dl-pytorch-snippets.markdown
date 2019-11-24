---
layout: post
title: pytorch 코드 snippets
date: 2019-03-01 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, snippets] # add tag
---

<br>

- 이 글은 pytorch 사용 시 참조할 수 있는 코드들을 모아놓았습니다.
- 완전히 기본 문법은 [이 글](https://gaussian37.github.io/dl-pytorch-pytorch-tensor-basic/)에서 참조하시기 바랍니다.

<br>

## **목차**
    - ### GPU/CPU Device 세팅 코드

<br>

## **GPU/CPU Device 세팅 코드**

<br>

```python
device = torch.device("cuda" if torch.cuda.torch.cuda.is_available() else "cpu")
```

<br>

- 위 코드와 같이 device의 유형을 선택하면 GPU가 존재하면 `cuda:0`에 할당되고 GPU가 없으면 `cpu`에 할당 되도록 할 수 있습니다.