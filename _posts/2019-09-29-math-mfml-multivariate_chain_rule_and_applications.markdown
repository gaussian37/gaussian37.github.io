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

$$ f(x, y, z) = \sin{(x)} \cdot exp(yz^{2}) $$

$$ x = t - 1, \quad y = t^{2}, \quad z = \frac{1}{t} $$

$$ \frac{\text{d}f}{\text{d}t} = \frac{\text{d}f}{\text{d}x}\frac{\text{d}x}{\text{d}t} + \frac{\text{d}f}{\text{d}y}\frac{\text{d}y}{\text{d}t} + \frac{\text{d}f}{\text{d}z}\frac{\text{d}z}{\text{d}t} $$

$$ \frac{\text{d}f}{\text{d}t} = \cos{(t-1)}e $$

<br>

- `total derivative`의 내용을 요약하면 `chain rule` 형태의 `multivariate derivative`에서는 각 변수의 partial derivative를 통해 분해한 뒤, chain rule을 적용할 수 있고 그 결과를 모두 합하면 된다는 내용입니다.

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
- 앞의 예제와 같은 형태로 함수가 있다고 가정해 보겠습니다. $$ f(x) $$는 함수이고 $$ x $$는 변수 $$ t $$에 dependent 한 벡터입니다.
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

- `chain rule`은 말 그대로 여러 단계에 걸쳐서 연쇄(chain)적으로 적용 가능합니다. 먼저 아래 예제에서 `univariate` 케이스 & 2단계 chain rule을 적용한 케이스에 대해서 살펴보겠습니다.

<br>

$$ f(x) = 5x \\ x(u) = 1 - u \\ u(t) = t^{2} $$

$$ \color{red}{f(t) = 5(1-t^{2}) = 5 - 5t^{2}} $$

$$ \color{red}{\frac{\text{d}f}{\text{d}t} = -10t} $$

$$ \begin{split} \color{blue}{\frac{\text{d}f}{\text{d}t}} &= \color{blue}{\frac{\text{d}f}{\text{d}x}\frac{\text{d}x}{\text{d}u}\frac{\text{d}u}{\text{d}t}} \\ &= \color{blue}{(5)(-1)(2t) = -10t}  \end{split}$$

<br>

- 대입을 하여 미분을 한 경우와 chain rule을 이용하여 미분을 한 경우 동일한 결과를 얻을 수 있습니다. 따라서 `univariate`에서 2단계, 3단계,  또는 n단계의 `chain rule`을 이용하여 미분을 하는 것과 일일이 대입해서 미분을 하는 것의 결과는 같습니다.
- 물론 이 글의 목적처럼 `multivariate`에서도 `chain rule`을 적용하면 2단계, 3단계, n단계 까지 모두 적용가능합니다. 한번 살펴보겠습니다. 이번에는 `multivariate`입니다.

<br>

$$ f(x(u(t)))$$

$$ \color{orange}{f(x) = f(x_{1}, x_{2})} $$

$$ \color{green}{x(u) = \begin{bmatrix} x_{1}(u_{1}, u_{2}) \\ x_{2}(u_{1}, u_{2}) \end{bmatrix}} $$ 

$$ \color{purple}{u(t) = \begin{bmatrix} u_{1}(t) \\ u_{2}(t) \end{bmatrix}} $$

<br>

- 위 식을 정리해 보면, $$ x = (x_{1}, x_{2}) $$로 벡터이고, $$ u = (u_{1}, u_{2}) $$로 벡터입니다. 용어로 나타내면 $$ x $$와 $$ u $$는 `vector valued function` 이라고 합니다.
- 함수의 `independent`한 변수인 $$ t $$가 `scalar input`이라 하고 함수의 출력 또한 `scalar output`이라고 하면 univariate 케이스와 동일하게 표현할 수 있습니다.

<br>

$$ \frac{\text{d}f}{\text{d}t} = \color{orange}{\frac{\partial f}{\partial x}} \color{green}{\frac{\partial x}{\partial u}}\color{purple}{\frac{\text{d}u}{\text{d}t}} $$

<br>

- 따라서 `chain rule`을 적용하여 계산한 결과를 보면 다음과 같습니다.

<br>

$$ \frac{\text{d}f}{\text{d}t} = \color{orange}{\frac{\partial f}{\partial x}} \color{green}{\frac{\partial x}{\partial u}}\color{purple}{\frac{\text{d}u}{\text{d}t}} = \color{orange}{\begin{bmatrix} \frac{\partial f}{\partial x_{1}} & \frac{\partial f}{\partial x_{2}} \end{bmatrix}} \color{green}{\begin{bmatrix} \frac{\partial x_{1}}{\partial u_{1}} & \frac{\partial x_{1}}{\partial u_{2}} \\ \frac{\partial x_{2}}{\partial u_{1}} & \frac{\partial x_{2}}{\partial u_{2}} \end{bmatrix}} \color{purple}{\begin{bmatrix} \frac{\text{d}u_{1}}{\text{d}t} \\ \frac{\text{d}u_{2}}{\text{d}t} \end{bmatrix}} $$

<br>

- 연산의 각 shape을 살펴보면 $$ (1 \times 1) = \color{orange}{(1 \times 2)}\color{green}{(2 \times 2)}\color{purple}{(2 \times 1)} $$ 가 됩니다.
- 특히, 유심히 봐야할 것은 각 단계를 보면 모두 `jacobian` (벡터/행렬) 이라는 것입니다.
- 앞의 글에서 배운 `jacobian`이 `chain rule`에서 얼마나 잘 활용되는 지 이해하셨으면 여기 까지의 핵심을 잘 따라오신 것입니다.

<br>

## **neural network**

<br>

- 21세기는 가히 머신 러닝의 시대라고 말할 수 있습니다. 특히 neural network의 시대가 밝아진 것이지요.
- 지금부터는 간단한 `neural network`를 살펴보면서 앞에서 배운 `multivariate chain rule`이 어떻게 사용되는 지 알아보도록 하겠습니다.
- 이 글의 초점은 `multivariate chain rule`의 `neural network`에서의 역할입니다. `neural netowork`에 대하여 알고 싶으시면 다음 링크를 참조하시기 바랍니다.
    - 링크 : https://gaussian37.github.io/dl-concept-table/
- 따라서 이번 글에서는 기본적인 neural network의 형태인 dnn(deep neural network)을 다룰 것이고 이는 단순히 fully connected 방식으로 연결한 구조입니다. 이 구조가 가장 간단하면서도 `multivariate chain rule`를 설명하기 용이합니다. 그러면 시작하겠습니다.

<br>

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- neural network가 하는 역할은 무궁무진 하지만, 사실 단순한 함수에 불과합니다. input이 있고 그 input에 대하여 처리를 한 다음에 output을 만들어 내는 말 그대로 함수입니다.
- 위 그림에서 원을 `neuron` 이라고 합니다. 선이 각각의 neuron을 이어주는 역할을 합니다. 

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 예를 들어 $$ a^{(0)} $$ neuron 에서 $$ a^{(1)} $$의 neuron으로 연결이 되었다면, 이 연결을 수식으로 나타내면 다음과 같습니다.

<br>

$$ a^{(1)} = \sigma(wa^{(0)} + b) $$

$$ a \Rightarrow \text{"activity"} $$

$$ w \Rightarrow \text{"weight"} $$

$$ b \Rightarrow \text{"bias"} $$

$$ \sigma \Rightarrow \text{"activation function"} $$

<br>

- 각 기호의 역할을 간단히 적으면 $$ a $$가 `input`, $$ w $$는 위의 neuron을 연결해 주는 선의 역할로 `weight` 값이고 역할은 input에 곱해지는 역할입니다. $$ w $$가 곱해지는 값이라면 $$ b $$는 더해지는 값으로 `bias` 입니다.
- 마지막으로 $$ \sigma $$는 이 neuron이 켜질 지, 꺼질 지 결정하는 함수입니다. 실제 사람의 신경 세포도 전기적 신호에 따라서 반응을 하거나 안하거나 선택이 됩니다. 수많은 neuron들이 동시에 반응을 하고 이 조합들을 통해 어떤 판단을 하게됩니다. 여기서 **반응을 하거나 안하거나 선택**하는 것은 마치 `threshold` 역할을 하는 것처럼 보입니다. $$ \sigma $$ 함수는 threshold 역할을 합니다. 이렇게 threshold를 넘었을 때와 넘지 못했을 때를 **활성화/비활성화** 라는 용어를 사용하여 neuron이 켜지는 지 꺼지는 지를 나타내기 때문에 `activation function`이라고 합니다.
    - 다양한 activation function을 참조 하시기 바랍니다.
    - 링크 : https://gaussian37.github.io/dl-concept-activation_functions/

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/2.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림은 대표적인 activation function 중의 하나인 `tanh`로 0을 기준으로 함수 값이 양수 / 음수로 나뉘게 됨을 볼 수 있습니다.
- 이처럼 $$ \sigma $$는 `non-linear function`의 형태를 가집니다. 왜냐하면 이러한 non-linear function이 중간에 적용되지 않으면 계속 linear transformation만 발생하기 때문입니다. 즉, linear classifier 밖에 할 수 없게 되기 때문에 항상 $$ \sigma $$는 `non-linear function`을 사용합니다.
- 지금까지 neural network에서 사용하는 기호인 $$ a^{(i)}, w, b, \sigma $$에 대하여 알아보았습니다.
- 지금부터는 neuron을 좀 더 사용해 보도록 하겠습니다.

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 그림과 같은 neuron을 구성하면 그 식은 다음과 같이 확장됩니다.

<br>

$$ a^{(1)} = \sigma(w_{0}a_{0}^{(0)} + w_{1}a_{1}^{(0)} + b) $$

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

$$ a^{(1)} = \sigma(w_{0}a_{0}^{(0)} + w_{1}a_{1}^{(0)} + w_{2}a_{2}^{(0)} + b) $$

<br>

- 즉, neuron이 추가될 수록 $$ w_{j}a_{j}^{(i)} $$만 추가되면 됩니다. 일반화 시키면 다음과 같습니다.

<br>

$$ \sigma \Biggl(\biggl(\sum_{j=0}^n w_{j}a_{j}^{(i)} \biggr) + b \Biggr) = \sigma(w \cdot a(i) + b) $$

<br>

- 이번에는 출력 neuron의 갯수를 한 개 더 늘려보겠습니다. 여기서 $$ w_{i} $$는 **벡터**입니다.

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/6.png" alt="Drawing" style="width: 400px;"/></center>
<br>

$$ \color{lime}{a_{0}^{(1)}} = \sigma(\color{lime}{w_{0}} \cdot a^{0} + \color{lime}{b_{0}}) $$

$$ \color{fuchsia}{a_{1}^{(1)}} = \sigma(\color{fuchsia}{w_{1}} \cdot a^{0} + \color{fuchsia}{b_{1}}) $$

$$ a^{(1)} = \sigma(W^{(1)} \cdot a^{(0)} + b^{(1)}) $$

<br>

- 출력의 neuron 갯수가 추가됨에 따라 $$ W $$는 `matrix`로 $$ b $$는 `vector`가 된 것을 확인할 수 있습니다.

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/7.png" alt="Drawing" style="width: 800px;"/></center>
<br>

- 지금 까지 배운 내용을 일반화 시키면 위와 같이 정리할 수 있습니다.
- 위 슬라이드는 조금 헷갈릴 수도 있어서 간략하게 설명해 보겠습니다. 먼저 $$ a_{n}^{(l)} $$는 $$ l $$ 번째 layer의 $$ n $$번째 노드 입니다. 단순하게 $$ a^{(l)} $$로 표시하면 이는 $$ a_{0}^{(l)} $$ 부터 $$ a_{n}^{(l)} $$ 까지의 값을 가지는 **벡터**가 됩니다.
- 이와 유사하게 $$ b_{n}^{(l)} $$는 $$ l - 1 $$번째 layer에서 $$ l $$ 번째 layer를 연결할 때 사용하는 bias로 아랫 첨자와 윗 첨자에 대한 의미는 노드와 같습니다. $$ b^{(l)} $$는 $$ l-1 $$번째 layer 노드를 이용하여 $$ l $$번째 layer를 연산할 때 사용되는 bias **벡터**가 됩니다.
- 조금 헷갈리는 점은 $$ w_{j,i}^{(l)} $$입니다. 이는 $$ a_{j}^{(l-1)} $$ 과 $$ a_{i}^{(l)} $$을 잇는 weight 입니다. 이 모든 weight 들을 한번에 표시한 $$ W^{(l)} $$은 weight **행렬**이 됩니다.
- 예를 들어 $$ a_{0}^{1} $$을 살펴보겠습니다.

<br>

$$ a_{0}^{1} = \sigma(w_{0,0}^{(1)}a_{0}^{0} + w_{0,1}^{(1)}a_{1}^{0} + \cdots + w_{0,n-1}^{(1)}a_{n-1}^{0} + b_{0}^{(1)}) $$

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/9.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 식으로 표현한 node, weight는 그림의 빨간색 선과 이어진 노드입니다. 노드의 인덱스가 $$ j \to i $$ 로 연결되어 있다는 점만 유의해서 보시면 됩니다.

<br>

- 마지막으로 추가해 볼 것은 `layer` 입니다. 

<br>
<center><img src="../assets/img/math/mfml/multivariate_chain_rule_and_applications/8.png" alt="Drawing" style="width: 400px;"/></center>
<br>

$$ \color{fuchsia}{a^{(1)}} = \sigma(W^{(1)} \cdot a^{(0)} + b^{(1)} ) $$

$$ a^{(2)} = \sigma(W^{(2)} \cdot \color{fuchsia}{a^{(1)}} + b^{(2)} ) $$

<br>

- 최종적으로 `input`, `layer`, `output`을 모두 고려한 일반화를 하면 식은 다음과 같습니다.

<br>

$$ a^{(L)} = \sigma(W^{(L)} \cdot a^{(L-1)} + b^{L}) $$

<br>

- 여기서 $$ a^{(L)} $$과 $$ a^{(L-1)} $$의 크기는 풀려고 하는 문제에 따라 다를 수 있습니다. 즉, 처음 모델링 할 때 정해주어야 하는 것입니다.
- 하지만 $$ W $$와 $$ b $$의 크기는 $$ a $$에 따라 정해집니다. 그 이유는 위 그림들을 보면 알 수 있습니다.
    - ① $$ W^{(L)} $$의 크기 : $$ \text{number of neuron in } a^{(L-1)} \times \text{number of neuron in } a^{(L)} $$
    - ② $$ b^{(L)} $$의 크기 : $$ \text{number of neuron in } a^{(L)} $$



<br>

[mathematics for machine learning 글 목록](https://gaussian37.github.io/math-mfml-table/)

<br>