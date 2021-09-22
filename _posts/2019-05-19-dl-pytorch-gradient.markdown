---
layout: post
title: PyTorch Gradient 관련 설명 (Autograd)
date: 2019-05-19 00:00:00
img: dl/pytorch/pytorch.jpg
categories: [dl-pytorch] 
tags: [pytorch, gradient] # add tag
---

<br>

- 이번 글에서는 pytorch에서 수행하는 `미분 방법`에 대하여 알아보겠습니다. 간단한 미분 예제와 그에 대응하는 코드를 통하여 pytorch에서 미분이 어떻게 계산되는지 살펴보겠습니다.

<br>

## **목차**

<br>

- ### [Autograd 사용 방법](#autograd-사용-방법-1)
- ### [Autograd 살펴보기](#autograd-살펴보기-1)
- ### [derivative 기본 예제](#derivative-기본-예제-1)
- ### [partial derivative 기본 예제](#partial-derivative-기본-예제-1)
- ### [학습 모드와 평가 모드](#학습-모드와-평가-모드-1)

<br>

## **Autograd 사용 방법**

<br>

- 어떤 tensor가 학습에 필요한 tensor라면 backpropagation을 통하여 gradient를 구해야 합니다. (즉, 미분을 해야 합니다.)
- tensor의 gradient를 구할 때에는 다음 조건들이 만족되어야 gradient를 구할 수 있습니다.
    - 1) tensor의 옵션이 `requires_grad = True` 로 설정되어 있어야 합니다. (tensor의 기본 값은 `requires_grad = False` 입니다.)
    - 2) backparopagation을 시작할 지점의 output은 `scalar` 형태이어야 합니다.
- tensor의 gradient를 구하는 방법은 backpropagation을 시작할 지점의 tensor에서 `.backward()` 함수를 호출하면 됩니다.
- gradient 값을 확인 하려면 `requires_grad = True`로 생성한 Tensor에서 `.grad`를 통해 값을 확인할 수 있습니다.
- 말로 하면 조금 어려우니, 다음 예제를 통해 간단하게 확인해 보겠습니다.

<br>

## **Autograd 살펴보기**

<br>

- 파이토치의 `Autograd`는 `자동 미분 (Auto differentitation)`을 이용하여 `변화도 (Gradient)` 계산을 한다는 것입니다.

<br>

```python
import torch
x1 = torch.ones(2, 2)
print(x1)
# tensor([[1., 1.],
#         [1., 1.]])

x2 = torch.ones(2, 2, requires_grad = True)
print(x2)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
```

<br>

- 위 예제는 단순하게 `torch.ones`를 이용하여 2 x 2 크기의 텐서를 생성하였고 후자는 `requires_grad=True` 옵션을 주고 생성한 것입니다. 후자의 경우 출력 결과에 `requires_grad=True`가 나타난 것을 볼 수 있는데, 이는 이후 역전파 과정을 수행 후, 해당 텐서의 변화도를 구할 수 있도록 합니다.

<br>

- 이제 위에서 사용한 `x1`과 `x2`를 이용하여 추가적인 산술 연산을 하면 어떻게 되는 지 살펴보도록 하겠습니다.

<br>

```python
import torch
x1 = torch.ones(2, 2)
print(x1)
# tensor([[1., 1.],
#         [1., 1.]])

y1 = x1 + 2
print(y1)
# tensor([[3., 3.],
#         [3., 3.]])

x2 = torch.ones(2, 2, requires_grad=True)
print(x2)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)
y2 = x2 + 2
print(y2)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
```

<br>

- 각 `x1`과 `x2`에 덧셈 연산을 수행하여 `y1`, `y2`를 만들었습니다. 코드 결과에 연산 수행 결과와 `grad_fn`이 `<AddBackward0>`인 것을 확인할 수 있습니다. 
- `grad_fn`에는 텐서가 어떤 연산을 하였는 지 연산 정보를 담고 있고, 이 정보는 역전파 과정에 사용될 예정입니다.

<br>

```python
import torch
x = torch.ones(2, 2, requires_grad=True)
y1 = x + 2
print(y1)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)

y2 = x - 2
print(y2)
# tensor([[-1., -1.],
#         [-1., -1.]], grad_fn=<SubBackward0>)

y3 = x * 2
print(y3)
# tensor([[2., 2.],
#         [2., 2.]], grad_fn=<MulBackward0>)

y4 = x / 2
print(y4)
# tensor([[0.5000, 0.5000],
#         [0.5000, 0.5000]], grad_fn=<DivBackward0>)
```

<br>
 
- 행렬의 대부분의 연산은 위 식에서 나타나는 사칙연산 (덧셈, 뺄셈, 곱셈, 나눗셈)을 이용하게 됩니다. 이 때, 각 텐서에서 나중에 이루어지는 역전파를 위해 기록되는 `grad_fn`에는 각각 `AddBackward0`, `SubBackward0`, `MulBackward0`, `DivBackward0`와 같이 저장되어 있는 것을 확인할 수 있습니다.

<br>

```python
x = torch.ones(2, 2)
y = x + 2

print(x)
# tensor([[1., 1.],
#         [1., 1.]])
print(y)
# tensor([[3., 3.],
#         [3., 3.]])

y.requires_grad_(True)

print(x)
# tensor([[1., 1.],
#         [1., 1.]])
print(y)
# tensor([[3., 3.],
#         [3., 3.]], requires_grad=True)
```

<br>

- 만약 `requires_grad`가 없는 텐서에서 속성을 추가할 때에는 `y.requires_grad_(True)`와 같은 방식으로 속성값을 추가해주면 됩니다.
- 다만, 이렇게 별도로 `requires_grad`를 추가한 경우 앞에서 연산한 이력이 `grad_fn`으로 자동으로 저장되지는 않습니다.


<br>

## **derivative 기본 예제**

<br>

- 지금까지 `Autograd`와 관련된 내용을 간략하게 살펴보았습니다. 이제 실제 미분 예제를 통하여 어떻게 연산이 되는 지 한번 살펴보도록 하겠습니다.
- 먼저 다음 식을 $$ x $$에 대하여 미분한다고 가정하겠습니다.

<br>

- $$ f(x) = 9x^{4} + 2x^{3} + 3x^{2} + 6x + 1 $$ 

<br>

- 이 때 도함수 $$ df(x) / dx $$를 구하면 $$ x = a $$ 지점의 경사를 구할 수 있습니다.

<br>

- $$ \frac{df(x)}{dx} = \frac{d(9x^{4} + 2x^{3} + 3x^{2} + 6x + 1)}{dx} = 36x^{3} + 6x^{2} + 6x + 6 $$

- 만약 $$ x = 2 $$ 라고 하면 $$ x = 2 $$의 변화율은 $$ 36 \times 2^{3} + 6 \times 2^{2} + 6 \times 2 + 6 = 330 $$ 입니다.
- 이 과정을 코드를 통해서 한번 살펴보겠습니다.

<br>

```python
x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
y.backward()
print(x.grad)
# tensor(330.)
```

<br>

- 위 예제에서 유심히 볼 점은 output인 `y`가 `scalar` 라는 점입니다. 앞에서 언급한 바와 같이 output이 scalar이어야 `backward` 연산이 됩니다.
- 서두에 설명한 바와 같이 output이 `scalar`가 아니면 `backwrard` 연산이 되지 않습니다. 다음과 같은 예제를 살펴보겠습니다.
- (약간 논점에 어긋나므로 생략하셔도 됩니다.)

<br>

```python
x = torch.randn(2, 2, requires_grad=True)
y = x + 2
z = (y * y)
z.backward()

# RuntimeError: grad can be implicitly created only for scalar outputs
```

<br>

- 위 코드를 실행하면 `RuntimeError: grad can be implicitly created only for scalar outputs`라는 에러가 발생합니다. 왜냐하면 `z`가 scalar 값이 아니라 2 x 2 크기의 매트릭스 이기 때문입니다. 따라서 다음과 같이 고쳐서 사용해야 정상적으로 사용 가능합니다. (아래는 `sum`을 사용하였지만 필요에 따라 `mean` 등을 사용하는 것은 무관합니다.)

<br>

```python
x = torch.randn(2, 2, requires_grad=True)
y = x + 2
z = (y * y).sum()
z.backward()

print(x)
# tensor([[ 2.5455,  1.3913],
#         [-0.4362, -0.5303]], requires_grad=True)

print(y)
# tensor([[4.5455, 3.3913],
#         [1.5638, 1.4697]], grad_fn=<AddBackward0>)

print(z)
# tensor(36.7684, grad_fn=<SumBackward0>)

print(x.grad)
# tensor([[9.0910, 6.7827],
#         [3.1276, 2.9394]])

print(y.grad)
# None

print(z.grad)
# None
```

<br>

- 위 예제를 살펴보면 실제 값이 할당된 텐서인 `x`의 경우 `backward`를 통해 계산된 `grad`가 저장된 것을 볼 수 있습니다. 반면 계산 중간 과정인 `y`, `z`의 경우 실제 `grad`가 저장되진 않고 `backward` 연산에만 참여된 것으로 볼 수 있습니다. 그 결과 `y.grad, z.grad`는 `None`이 됩니다.

<br>

- 만약 `backward`를 `scalar`가 아닌 매트릭스에서 진행하려면 다음과 같이 사용할 수 있습니다.

<br>

```python
x = torch.randn(2, 2, requires_grad=True)
y = x + 2
z = (y * y)
y.backward(z)

print(x.grad)
# tensor([[2.2796, 3.2123],
#         [5.1224, 0.6321]])
```

<br>

- 위 식과 같이 `y.backward(z)`처럼 `z`를 `tensor.backward()`의 인자로 넣어주면 연산이 가능해집니다.

<br>

- 그러면 이번에는 위 예제를 응용하여 input인 x가 scalar가 아니라 (2, 2) shape matrix 형태로 사용해 보겠습니다. 이번 예제에서 사용할 식은 다음과 같습니다.

<br>

- $$ f(X) = \overline{3(X_{ij} + 2)^{2}} = \frac{1}{4}\sum_{i = 0}^{1}\sum_{j=0}^{1} 3(X_{ij} + 2)^{2} $$

- $$ \frac{df(X)}{dX} = \frac{1}{4} \sum_{i=0}^{1}\sum_{j=0}^{1} (6X_{ij} + 12) =  \sum_{i=0}^{1}\sum_{j=0}^{1} (1.5X_{ij} + 3) = 1.5(X_{00} + X_{01} + X_{10} + X_{11} ) + 3 \cdot 4 $$

<br>

- matrix에서 각 원소의 값에 대하여 미분을 하면 다른 원소는 영향을 주지 않습니다. 예를 들어 $$ X_{00} $$에 대하여 미분하면 그 결과는 $$ 1.5(X_{00}) + 3 $$이 됩니다. 따라서 2 x 2 matrix $$ X $$의 미분 결과는 다음과 같습니다.

<br>

- $$ \frac{df(X)}{dX} = \begin{bmatrix} 1.5(X_{00}) + 3 & 1.5(X_{01}) + 3 \\ 1.5(X_{10}) + 3 & 1.5(X_{11}) + 3 \end{bmatrix} $$

<br>

- 그러면 matrix X가 $$ X = [1.0, 2.0; 3.0, 4.0] $$ 일 때, 다음과 같이 gradient 값을 구할 수 있습니다.

<br>

- $$ \frac{df(X)}{dX} = \begin{bmatrix} 1.5(X_{00}) + 3 & 1.5(X_{01}) + 3 \\ 1.5(X_{10}) + 3 & 1.5(X_{11}) + 3 \end{bmatrix} = \begin{bmatrix} 4.5 & 6.0 \\ 7.5 & 9.0 \end{bmatrix} $$

<br>

- 그러면 코드를 통해 한번 살펴보도록 하겠습니다.

<br>

```python
x = torch.tensor([[1.0, 2.0],[3.0, 4.0]], requires_grad = True)
# tensor([[1., 2.],
#         [3., 4.]], requires_grad=True)

y = x + 2
# tensor([[3., 4.],
#         [5., 6.]], grad_fn=<AddBackward0>)

z = y * y * 3
# tensor([[ 27.,  48.],
#         [ 75., 108.]], grad_fn=<MulBackward0>)

out = z.mean()
# tensor(64.5000, grad_fn=<MeanBackward0>)

out.backward()
print(x.grad)
# tensor([[4.5000, 6.0000],
#         [7.5000, 9.0000]])
```

<br>

- 수식으로 푼 것과 pytorch를 이용해서 푼 것이 일치하는 것을 확인할 수 있습니다.
- 만약 tensor에서 `requires_grad = True`를 입력하지 않으면 오류가 발생합니다. 이것도 한번 확인해 보시길 바랍니다.

<br>

## **partial derivative 기본 예제**

<br>

- 이번에는 편미분 예제를 간단하게 다루어 보겠습니다.
- 식 $$ y = x^{2} + z^{3} $$이 있다고 가정해보겠습니다.
- 먼저 변수 $$ x $$에 대하여 편미분 하겠습니다.

<br>

- $$ \frac{\partial}{\partial x}(x^{2} + z^{3}) = 2x $$

<br>

- 다음으로 변수 $$ z $$에 대하여 편미분 하면

<br>

- $$ \frac{\partial}{\partial z}(x^{2} + z^{3}) = 3z^{2} $$

<br>

- 이 때, $$ f'(x, z) = f'(1, 2) $$ 연산을 해보겠습니다.
- 먼저 $$ x $$에 관하여 연산을 하면 다음과 같습니다.

<br>

- $$ \frac{\partial(f(x) = x^{2})}{\partial x} \quad \text{where  }  x=1 $$

- $$ y'(1) = 2 $$

<br>
        
- 다음으로 $$ z $$에 관하여 연산을 하면 다음과 같습니다.

<br>

- $$ \frac{\partial(f(z) = z^{3})}{\partial z} \ \ where \ z=2 $$

- $$ y'(2) = 12 $$

<br>

- 그러면 이 내용을 코드로 살펴보겠습니다.

```python
x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3
y.backward()
print(x.grad, z.grad)
## tensor(2.) tensor(12.)
```

<br>

## **학습 모드와 평가 모드**

<br>

- 앞에서 다룬 모든 예제는 gradient를 구하기 위해서 `tensor`의 속성을 `requires_grad = True`로 설정하였습니다.
- gradient를 구한다는 것은 학습 대상이 되는 weight 입니다.
- 반면 학습이 모두 끝나고 학습한 결과를 실행에 옮기는 inference 단계에서는 굳이 학습 모드로 사용할 필요가 없습니다.
- 이 때, 사용하는 것이 `torch.no_grad()` 입니다. `torch.no_grad()`가 적용된 tensor는 비록 실제 속성은 `requires_grad = True` 이더라도 **gradient를 업데이트 하지 않고, dropout, batchnormalization 등이 적용되지 않습니다.**
- 아래 코드를 살펴보겠습니다.

<br>

```python
x = torch.tensor(1.0, requires_grad = True)
print(x.requires_grad)
# True

with torch.no_grad():
    print(x.requires_grad)
    print((x**2).requires_grad)
# True
# False

print(x.requires_grad)
# True
```

<br>

- 위 예제를 살펴보면 `with torch.no_grad():`에서 `requires_grad = True`인 tensor가 어떤 연산이 적용될 때, `requires_grad = False`로 변경된 것을 확인할 수 있습니다.
- 그리고 with 문을 벗어나면 다시 `requires_grad = True`로 원복된 것을 확인하실 수 있습니다.
- 이러한 방식으로 gradient를 업데이트 하는 모드와 그렇지 않은 모드를 구분하실 수 있습니다.