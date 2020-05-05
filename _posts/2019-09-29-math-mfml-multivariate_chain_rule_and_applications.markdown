---
layout: post
title: multivariate chain rule과 applications
date: 2019-09-29 00:00:00
img: math/mfml/mfml.jpg
categories: [math-mfml] 
tags: [calculus, multivariate chain rule, application] # add tag
---

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>

- 이 글은 Coursera의 Mathematics for Machine Learning: Multivariate Calculus 을 보고 정리한 
내용입니다.
- 이번 글에서는 앞에서 다룬 charin rule을 multivariate 환경으로 확장을 하고 neural network에 접목을 시켜보겠습니다.
- 

<br>

### **목차**

<br>

- ### multivariate chain rule
- ### neural network
- ### backpropagation

<br>

## **multivariate chain rule**

<br>

- 앞의 글에서 배운 `total derivative`에 대하여 간략한 예제를 살펴보겠습니다.

<br>

$$ f(x, y, z) = \sin{(x)}e^{yz^{2}} $$

$$ x = t - 1, \quad y = t^{2}, \quad z = \frac{1}{t} $$

$$ \frac{\text{d}f}{\text{d}t} = \frac{\text{d}f}{\text{d}x}\frac{\text{d}x}{\text{d}t} + \frac{\text{d}f}{\text{d}y}\frac{\text{d}y}{\text{d}t} + \frac{\text{d}f}{\text{d}z}\frac{\text{d}z}{\text{d}t} $$

$$ \frac{\text{d}f}{\text{d}t} = \cos{(t-1)}e $$

<br>

- `total derivative`의 내용을 요약하면 `chain rule` 형태의 `multivariate derivative`에서는 각각의 변수의 partial derivative를 통해서 분해하여 chain rule을 적용할 수 있고 그 결과를 모두 합하면 된다는 내용입니다.

<br>

- 이것을 좀 더 일반화한 식으로 적어보도록 하겠습니다.

<br>

$$ f(x) = f(x_{1}, x_{2}, \cdots , x_{n}) $$

<br>

$$ \frac{\partial f}{\partial x} = \begin{bmatrix} \partial f / \partial x_{1} \\ \partial f / \partial x_{2} \\ \partial f / \partial x_{3} \\ \vdots \\ \partial f / \partial x_{n} \end{bmatrix} \quad  \frac{\text{d} x}{\text{d} t} = \begin{bmatrix} \text{d} x_{1} / \text{d} t \\ \text{d} x_{2} / \text{d} t \\ \text{d} x_{3} / \text{d} t \\ \vdots \\ \text{d} x_{n} / \text{d} t \end{bmatrix} $$

<br>

$$ \frac{\text{d}f}{\text{d}t} = \frac{\partial f}{\partial x} \cdot \frac{\text{d}x}{\text{d}t} $$

<br>

- 위 식처럼 `inner product` 형태로 나타내면 `total derivative`를 간단하게 표현할 수 있습니다.

<br>

- 그러면 `multivariate chain rule`에 대하여 조금만 더 자세히 살펴보도록 하겠습니다.
- 앞의 예제와 같은 형태로 함수가 있다고 가정해 보겠습니다.
    - 함수 : $$ f(x) $$
    - 변수 $$ x $$ : 벡터이고 변수 $$ t $$에 dependent 함
- 그러면 $$ f(x(t)) $$ 형태로 적을 수 있습니다. 여기서 `independent`한 변수는 $$ t $$이므로 결국 관심이 있는 것은 $$ t $$의 변화에 대한 함수 $$ f $$의 변화를 알고 싶은 것이 derivative의 목적입니다. 그러면 다음과 같이 compact 하게 쓸 수 있습니다.

<br>

$$ \frac{\text{d}f}{\text{d}t} = \frac{\partial f}{\partial x} \cdot \frac{\text{d}x}{\text{d}t} $$

<br>

- 그리고 $$ \partial f / \partial x $$에서 $$ x $$는 `vector`이므로 $$ (x_{1}, x_{2}, \cdots , x_{n}) $$의 형태를 가지게 됩니다.
- 이 글의 윗 부분에서도 `column vector` 형식으로 $$ \partial f / \partial x $$을 표현하였는데, 앞의 글에서 배운 `jacobian`에 대입해서 보면 `transpose`한 형태가 됩니다.

<br>

$$ \frac{\partial f}{\partial x} = \begin{bmatrix} \partial f / \partial x_{1} \\ \partial f / \partial x_{2} \\ \partial f / \partial x_{3} \\ \vdots \\ \partial f / \partial x_{n} \end{bmatrix} = (J_{f})^{T} $$

<br>

- 따라서 위 식의 `column vector`의 `inner product` 또한 다음과 같이 `row vector`와 `column vector`의 곱으로 표현할 수 있습니다.

<br>

$$ \frac{\text{d}f}{\text{d}t} = J_{f} \frac{\text{d}x}{\text{d}t} $$

<br>
    

<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>