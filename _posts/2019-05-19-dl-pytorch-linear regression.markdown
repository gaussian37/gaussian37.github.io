---
layout: post
title: Linear Regression with PyTorch
date: 2019-05-19 02:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, linear regression] # add tag
---

+ 출처 : https://github.com/GunhoChoi/PyTorch-FastCampus
+ 이번 글에서는 Pytorch를 이용하여 Linear regression 하는 방법을 알아보도록 하겠습니다.
+ Linear regression의 의미 보다는 PyTorch에서의 training 과정을 어떻게 구현하면 되는지를 초점으로 살펴보면 되겠습니다.

## 텐서와 자동 미분

+ 텐서에는 `requires_grad` 라는 속성이 있어서 이 값을 True로 설정하면 자동 미분 기능이 활성화 됩니다.
+ Neural Net에서는 파라미터나 데이터가 모두 이 기능을 사용합니다.
+ `requires_grad`가 적용된 텐서에 다양한 계산을 하게 되면 계산 그래프가 만들어 지며, 여기에 `backward` 메소드를 호출하면 그래프가 자동으로 미분을 계산합니다.
  + `requires_grad = True`가 적용된 텐서에 한하여 backprop이 계산되기 때문에 학습이 필요한 `파라미터`에 `requires_grad = True`를 적용하면 됩니다.

+ 다음 식은 $$ L $$ 을 계산해서 $$ a_{k} $$로 미분하는 예제입니다.
  + 　$$ y_{i} = \mathbb a \cdot x_{i}, \ \ L = \sum_{i} y_{i} $$
  + 　$$ \frac{\partial L}{\partial a_{k}} = \sum_{i} x_{ik} $$
+ 이 작업을 자동 미분으로 구해보도록 하겠습니다.

```python
x = torch.randn(100,3)

# 미분할 변수로 사용되는 학습 파라미터는 requires_grad=True 로 설정합니다.
a =  torch.tensor([1,2,3.], requires_grad = True)

# 계산 그래프를 생성합니다.
y = torch.mv(x,a)
o = y.sum()

# 미분을 실행합니다. 이 때 requires_grad=True로 설정되어 있는 파라미터 a는 backprop이 실행 됩니다.
o.backward()

a.grad
: tensor([18.5806, -4.1716, -0.1735])

x.sum(0)
: tensor([18.5806, -4.1716, -0.1735])
```

<br>

+ 위 식을 보면 마지막의 `a.grad`와 `x.sum(0)`의 값이 일치함을 알 수 있습니다.
+ 그 이유는 $$ a $$에 대하여 미분한 결과를 합하면 $$ x $$를 sum한 결과와 같기 때문입니다.
+ `Neural Network`에서 처럼 Chain rule이 적용이 되면 위와 같은 자동 미분의 역할을 매우 중요해집니다.
