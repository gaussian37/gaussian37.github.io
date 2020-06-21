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

## **목차**

<br>

- ### Autograd 사용 방법
- ### derivative 기본 예제
- ### partial derivative 기본 예제
- ### 학습 모드와 평가 모드

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
print(x.grad)
# tensor(330.)
```

<br>

- 위 예제에서 유심히 볼 점은 output인 `y`가 scalar 라는 점입니다. 앞에서 언급한 바와 같이 output이 scalar이어야 `backward` 연산이 됩니다.

<br>

- 그러면 이번에는 위 예제를 응용하여 input인 x가 scalar가 아니라 (2, 2) shape matrix 형태로 사용해 보겠습니다. 이번 예제에서 사용할 식은 다음과 같습니다.

<br>

$$ f(X) = \overline{3(X_{ij} + 2)^{2}} = \frac{1}{4}\sum_{i = 0}^{1}\sum_{j=0}^{1} 3(X_{ij} + 2)^{2} $$

$$ \frac{df(X)}{dX} = \frac{1}{4} \sum_{i=0}^{1}\sum_{j=0}^{1} (6X_{ij} + 12) =  \sum_{i=0}^{1}\sum_{j=0}^{1} (1.5X_{ij} + 3) = 1.5(X_{00} + X_{01} + X_{10} + X_{11} ) + 3 $$

<br>

- matrix에서 각 원소의 값에 대하여 미분을 하면 다른 원소는 영향을 주지 않습니다. 예를 들어 $$ X_{00} $$에 대하여 미분하면 그 결과는 $$ 1.5(X_{00}) + 3 $$이 됩니다. 따라서 2 x 2 matrix $$ X $$의 미분 결과는 다음과 같습니다.

<br>

$$ \frac{df(X)}{dX} = \begin{bmatrix} 1.5(X_{00}) + 3 & 1.5(X_{01}) + 3 \\ 1.5(X_{10}) + 3 & 1.5(X_{11}) + 3 \end{bmatrix} $$

<br>

- 그러면 matrix X가 $$ X = [1.0, 2.0; 3.0, 4.0] $$ 일 때, 다음과 같이 gradient 값을 구할 수 있습니다.

<br>

$$ \frac{df(X)}{dX} = \begin{bmatrix} 1.5(X_{00}) + 3 & 1.5(X_{01}) + 3 \\ 1.5(X_{10}) + 3 & 1.5(X_{11}) + 3 \end{bmatrix} = \begin{bmatrix} 4.5 & 6.0 \\ 7.5 & 9.0 \end{bmatrix} $$

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

$$ \frac{\partial}{\partial x}(x^{2} + z^{3}) = 2x $$

<br>

- 다음으로 변수 $$ z $$에 대하여 편미분 하면

<br>

$$ \frac{\partial}{\partial z}(x^{2} + z^{3}) = 3z^{2} $$

<br>

- 이 때, $$ f'(x, z) = f'(1, 2) $$ 연산을 해보겠습니다.
- 먼저 $$ x $$에 관하여 연산을 하면 다음과 같습니다.

<br>

$$ \frac{\partial(f(x) = x^{2})}{\partial x} \quad \text{where  }  x=1 $$

$$ y'(1) = 2 $$

<br>
        
- 다음으로 $$ z $$에 관하여 연산을 하면 다음과 같습니다.

<br>

$$ \frac{\partial(f(z) = z^{3})}{\partial z} \ \ where \ z=2 $$

$$ y'(2) = 12 $$

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

## 학습 모드와 평가 모드

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