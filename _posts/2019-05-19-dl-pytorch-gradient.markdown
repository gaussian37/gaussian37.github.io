---
layout: post
title: PyTorch Gradient 관련 설명
date: 2019-05-19 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, gradient] # add tag
---

<br>

- 이번 글에서는 pytorch에서 수행하는 `미분 방법`에 대하여 알아보겠습니다. 간단한 미분 예제와 그에 대응하는 코드를 통하여 pytorch에서 미분이 어떻게 계산되는지 살펴보겠습니다.

<br>

## **derivative 기본 예제**

<br>

- 먼저 다음 식을 $$ x $$에 대하여 미분한다고 가정하겠습니다.

<br>

$$ f(x) = 9x^{4} + 2x^{3} + 3x^{2} + 6x + 1 $$ 

<br>

- 이 때 도함수 $$ df(x) / dx $$를 구하면 $$ x = a $$ 지점의 경사를 구할 수 있습니다.

<br>

$$ \frac{df(x)}{dx} = \frac{d(9x^{4} + 2x^{3} + 3x^{2} + 6x + 1)}{dx} = 36x^{3} + 6x^{2} + 6x + 6 $$

- 만약 $$ x = 2 $$ 라고 하면 $$ x = 2 $$의 변화율은 $$ 36 \times 2^{3} + 6 \times 2^{2} + 6 \times 2 + 6 = 330 $$ 입니다.
- 이 과정을 코드를 통해서 한번 살펴보겠습니다.

<br>

```python
x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
y.backward()
>> print(x.grad)

tensor(330.)
```

<br>

## **partial derivative 기본 예제**

- 이번에는 편미분 예제를 간단하게 다루어 보겠습니다.
- 식 $$ y = x^{2} + z^{3} $$이 있다고 가정해보겠습니다.
- 먼저 변수 $$ x $$에 대하여 편미분 하겠습니다.
    - 　$$ \frac{\partial}{\partial x}(x^{2} + z^{3}) = 2x $$
- 다음으로 변수 $$ z $$에 대하여 편미분 하면
    - 　$$ \frac{\partial}{\partial z}(x^{2} + z^{3}) = 3z^{2} $$
- 이 때, $$f'(x, z) = f'(1, 2) $$ 연산을 하면
    - 먼저 $$ x $$에 관하여 연산을 하면
        - 　$$ \frac{\partial(f(x) = x^{2})}{\partial x} \ \ where \ x=1 $$
        - 　$$ y'(1) = 2 $$
    - 다음으로 $$ z $$에 관하여 연산을 하면
        - 　$$ \frac{\partial(f(z) = z^{3})}{\partial z} \ \ where \ z=2 $$
        - 　$$ y'(2) = 12 $$
- 이 내용을 코드로 살펴보겠습니다.

```python
x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3
y.backward()
>> print(x.grad, z.grad)

tensor(2.) tensor(12.)

```