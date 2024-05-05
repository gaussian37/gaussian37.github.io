---
layout: post
title: Jacobian(자코비안) 이란?
date: 2019-02-06 00:00:00
img: math/calculus/jacobian/jacobian.png
categories: [math-calculus] 
tags: [Jacobian, 자코비안] # add tag
---

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>

<br>

- 참조 : http://t-robotics.blogspot.com/2013/12/jacobian.html#.XGlnkegzaUk
- 참조 : https://suhak.tistory.com/944

<br>

## **목차**

<br> 

- ### [자코비안의 정의 및 예시](#자코비안의-정의-및-예시-1)
- ### [Python을 이용한 자코비안 계산](#python을-이용한-자코비안-계산-1)

<br>

## **자코비안의 정의 및 예시**

<br>

- 자코비안은 다양한 문제에서 `approximation` 접근법을 사용할 때 자주 사용 되는 방법입니다.
- 예를 들어 비선형 칼만필터를 사용할 때, 비선형 식을 선형으로 근사시켜서 모델링 할 때 사용하는 **Extended Kalman Filter**가 대표적인 예가 될 수 있습니다.
- 자코비안은 정말 많이 쓰기 때문에 익혀두면 상당히 좋습니다. 이 글에서 자코비안에 대하여 다루어 보도록 하겠습니다.

<br>

<br>
<center><img src="../assets/img/math/calculus/jacobian/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 앞에서 말했듯이 `자코비안의 목적`은 복잡하게 얽혀있는 식을 미분을 통하여 `linear approximation` 시킴으로써 간단한 `근사 선형식`을 만들어 주는 것입니다.
- 위 그래프에서 미분 기울기를 통하여 $$ \Delta x $$ 후의 y값을 `선형 근사`하여 예측하는 것과 비슷한 원리 입니다.
- 그런데 위 그래프에서 가장 알고 싶은 것은 $$ f'(x_{1}) $$ 에서의 함수 입니다.  
- 물론 $$ y = f(x) $$와 같은 `1변수 함수`에서는 미분값도 스칼라 값이 나오기도 합니다. 
- 하지만 $$ x = (x_{1}, x_{2}, ...), y = (y_{1}, y_{2}, ...) $$와 같이 일반화한 경우 미분값이 스칼라 값이 아니라 `행렬`형태로 나오게 됩니다.

<br>
<center><img src="../assets/img/math/calculus/jacobian/2.png" alt="Drawing" style="width: 300px;"/></center>
<br>
    
- 여기서 `J`가 앞의 그래프 예시에 있는 함수 $$ f'(x) $$ 입니다.

<br>
 
- 위키피디아의 Jacobian 행렬 정의를 찾아보면 **"The Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function"** 으로 나옵니다. 
- 즉, 자코비안 행렬은 모든 벡터들의 `1차 편미분값`으로된 행렬로 각 행렬의 값은 **다변수 함수일 때의 미분값**입니다.

<br>
    
- 첫 그래프를 보고 `자코비안을 구한 이유`에 대해서 다시 생각해 보면, $$ q = f^{-1}(x) $$를 구하기 어렵기 때문에 선형화 하여 근사값을 구한 것입니다. 따라서 이 문제를 자코비안의 역행렬을 이용하여 푼다면 근사해를 구할 수 있습니다.

<br>
    
- $$ dx = Jdq $$

- $$ dq = J^{-1}dx $$

- $$ \therefore q_{2} = q_{1} + J^{-1}dx  $$
    
<br>

- 이번에는 다른 방법을 통하여 자코비안의 사용 예시를 설명드리겠습니다.

<br>

- $$ \int_{a}^{b}f(g(x))g^{\prime}(x)dx=\int_{g(a)}^{g(b)}f(u)du \quad\quad( u=g(x)\; \iff \;du=g^{\prime}(x)dx) $$

<br>

- $$ \begin{split}\int_{0}^{1} \sqrt{1-x^2} d x  &=\int_{0}^{\pi/2} \sqrt{1-\sin^2 \theta} \cos \theta d \theta \quad (x=\sin \theta\; \iff \;dx=\cos \theta d \theta) \\ &=\int_{0}^{\pi/2} \cos^2 \theta d \theta =\int_{0}^{\pi/2} \frac{1}{2} (1+\cos 2 \theta) d \theta \\ &= \frac{1}{2} \left[ \theta + \frac{1}{2} \sin 2 \theta \right]_0^{\pi/2}=\frac{\pi}{4} \end{split} $$

- $$ \cos{2\theta} = 2\cos{\alpha}^{2} - 1 $$

<br>

## **Python을 이용한 자코비안 계산**

<br>

- 자코비안은 $$ n \times m $$ 크기의 행렬이며 $$ n $$ 개의 함수식과 $$ m $$ 개의 변수를 이용하여 다음과 같이 구해짐을 확인하였습니다. 편의상 변수는 $$ x_{i} $$ 로 나타내겠습니다.

<br>

- $$ J = \begin{bmatrix} \frac{\partial f_{1}}{\partial x_{1}} & \frac{\partial f_{1}}{\partial x_{2}} & \cdots & \frac{\partial f_{1}}{\partial x_{m-1}} & \frac{\partial f_{1}}{\partial x_{m}} \\ \frac{\partial f_{2}}{\partial x_{1}} & \frac{\partial f_{2}}{\partial x_{2}} & \cdots & \frac{\partial f_{2}}{\partial x_{m-1}} & \frac{\partial f_{2}}{\partial x_{m}} \\ \vdots & \vdots & \ddots & \vdots & \vdots& \\ \frac{\partial f_{n-1}}{\partial x_{1}} & \frac{\partial f_{n-1}}{\partial x_{2}} & \cdots & \frac{\partial f_{n-1}}{\partial x_{m-1}} & \frac{\partial f_{n-1}}{\partial x_{m}} \\ \frac{\partial f_{n}}{\partial x_{1}} & \frac{\partial f_{n}}{\partial x_{2}} & \cdots & \frac{\partial f_{n}}{\partial x_{m-1}} & \frac{\partial f_{n}}{\partial x_{m}} \end{bmatrix} $$

<br>

- 위 식에서 자코비안을 구하기 위한 편미분 계산은 파이썬의 `sympy`를 이용하면 간단하게 처리할 수 있습니다. (울프람 알파 등을 이용하여도 됩니다.)
- 이번 글에서는 자코비안을 `sympy`를 이용하여 구하는 방법에 대하여 알아보도록 하겠습니다.

<br>

- 먼저 $$ f_{i} $$ 를 각각 정의해야 하고 편미분할 변수를 정의해야 합니다. 다루어볼 예제는 다음과 같습니다.

<br>

- $$ F = \begin{bmatrix} x_{1}^{2} + x_{2}^{2} + x_{3} \\ \sin{(x_{1})} + \cos{(x_{2})} \end{bmatrix} $$

- $$ X = \begin{bmatrix} x_{1} & x_{2} & x_{3} \end{bmatrix}^{T} $$

<br>

- 자코비안의 정의에 맞게 각 행렬의 성분을 편미분 하면 다음 결과를 얻을 수 있습니다.

<br>

- $$ J = \begin{bmatrix} \frac{f_{1}}{x_{1}} & \frac{f_{1}}{x_{2}} & \frac{f_{1}}{x_{3}} \\ \frac{f_{2}}{x_{1}} & \frac{f_{2}}{x_{2}} & \frac{f_{2}}{x_{3}} \end{bmatrix} = \begin{bmatrix} 2x_{1} & 2x_{2} & 1 \\ \cos{(x_{1})} & -\sin{(x_{2})} & 0 \end{bmatrix} $$

<br>

- 위 연산을 다음과 같이 계산할 수 있습니다.

<br>

```python
from sympy import symbols, Matrix, sin, cos, lambdify
import numpy as np

# Step 1: Define the symbolic variables
x1, x2, x3 = symbols('x1 x2 x3')

# Step 2: Define the function vector
f = Matrix([x1**2 + x2**2 + x3, sin(x1) + cos(x2)])

# Step 3: Define the variable vector
x = Matrix([x1, x2, x3])

# Step 4: Compute the Jacobian matrix
J = f.jacobian(x)
```

<br>
<center><img src="../assets/img/math/calculus/jacobian/3.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 따라서 코드를 이용해서도 동일한 결과를 얻을 수 있음을 확인하였습니다.
- 다음으로 임의의 입력에 대하여 자코비안 행렬 연산 방법을 확인해 보겠습니다. 연산은 `numpy`를 사용하도록 코드를 작성하였습니다.

<br>

```python
# Step 5: Convert the Jacobian to a NumPy function
jacobian_func = lambdify([x1, x2, x3], J, modules='numpy')

def jacobian_func_np(x1, x2, x3):
    matrix = np.array([
        [2*x1,        2*x2,       1],
        [np.cos(x1), -np.sin(x2), 0]
    ])
    return matrix 

# Step 6: Example usage with NumPy
input_vector = np.array([1.0, 2.0, 3.0])
jacobian_result = jacobian_func(*input_vector)

print("Jacobian matrix with lambdify at", input_vector, "is:")
print(np.array(jacobian_result))
print()
# Jacobian matrix with lambdify at [1. 2. 3.] is:
# [[ 2.          4.          1.        ]
#  [ 0.54030231 -0.90929743  0.        ]]

print("Jacobian matrix with numpy at", input_vector, "is:")
jacobian_result = jacobian_func_np(*input_vector)
print(np.array(jacobian_result))
# Jacobian matrix with numpy at [1. 2. 3.] is:
# [[ 2.          4.          1.        ]
#  [ 0.54030231 -0.90929743  0.        ]]
```

<br>

- 위 코드의 2가지 결과 모두 동일함을 알 수 있습니다. 첫번째 `jacobian_func`는 `lambdify`라는 모듈을 이용하여 자코비안 행렬을 별도 변환 없이 바로 사용한 경우입니다. 반면 `jacobian_func_np`는 직접 `numpy` 배열로 변환하여 사용한 것으로 `jacobian_func`를 확인하기 위한 용도로 별도 정의하였습니다.
- 따라서 실제 사용할 때에는 `jacobian_func`를 사용하여 `jacobian`을 구하고 계산된 결과만 `numpy`로 받아서 사용하는 것을 권장드립니다.

<br>

- 다음은 원의 방정식에 대한 자코비안을 구하는 형태의 예제를 살펴보도록 하겠습니다. 
- 원의 방정식 $$ (x - a)^{2} + (y - b)^{2} = r^{2} $$ 에서 $$ a, b, r $$ 에 대한 편미분을 통해 자코비안을 계산합니다. 
- 첫번째 예제로 복수 개의 식을 나열하기 위하여 직접 $$ (x, y) $$ 를 대입하여 $$ n $$ 개의 식을 만들어 자코비안을 구하는 방법과 $$ (x, y) $$ 또한 변수화 하여 대수적으로 자코비안을 구하는 방식을 차례대로 확인해 보겠습니다.

<br>

- $$ (x - a)^{2} + (y - b)^{2} = r^{2} $$

- $$ (a, b) \text{: center coordinates of the circle} $$

- $$ r \text{: radius} $$

- $$ \text{Jacobian with respect to the circle parameters (a, b, r)} $$

<br>

- $$ \text{residual function: } r_{i} = \sqrt{(x_{i} - a)^{2} + (y_{i} - b)^{2}} - r $$

<br>

```python
from sympy import symbols, Matrix, sqrt, lambdify
import numpy as np

# Define symbolic variables for the circle parameters and points
a, b, r = symbols('a b r')

# Example data points
points = np.array([[1, 7], [5, 8], [9, 8]])

# Example circle parameters (a, b, r)
params = [5.0, 5.0, 3.0]

# Define the residual function for a single point
residuals = Matrix([])
for x, y in points:
    residuals = residuals.row_insert(residuals.shape[0], Matrix([sqrt((x - a)**2 + (y - b)**2) - r]))

# Compute the Jacobian matrix of the residual function
jacobian = residuals.jacobian([a, b, r])

# Convert the Jacobian to a numerical function using lambdify
jacobian_func = lambdify([a, b], jacobian, 'numpy')

# Print the Jacobian matrix
print("Jacobian matrix (3x3):")
jacobian_result = jacobian_func(0, 0)
print(jacobian_result)
```

<br>
<center><img src="../assets/img/math/calculus/jacobian/4.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 식에서 $$ F $$ 로 정의한 `residuals`와 $$ J $$ 인 `jacobian`을 구한 결과는 위 식과 같습니다.

<br>

- 이번에는 약간 $$ (x, y) $$ 또한 변수화 하여 자코비안 행렬을 생성해 보도록 하겠습니다. 코드의 편의상 사용한 것이며 개념적으로 달라진 것은 없습니다.

<br>

```python
from sympy import symbols, Matrix, sqrt, lambdify
import numpy as np

# Define symbolic variables for the circle parameters and points
a, b, r = symbols('a b r')
x, y = symbols('x y')

# Define the residual function for a single point
residuals = Matrix([sqrt((x - a)**2 + (y - b)**2) - r])

# Compute the Jacobian matrix of the residual function
jacobian = residuals.jacobian([a, b, r])

# Convert the Jacobian to a numerical function using lambdify
jacobian_func = lambdify([a, b, r, x, y], jacobian, 'numpy')

# Function to compute the Jacobian matrix for multiple points
def get_jacobian_results(params, points):
    a, b, r = params
    return np.array([jacobian_func(a, b, r, x, y) for x, y in points])

# Example data points
points = np.array([[1.0, 7.0], [4.0, 8.0], [9.0, 8.0]])

# Example circle parameters (a, b, r)
params = [5.0, 5.0, 3.0]

# Compute the 3x3 Jacobian matrix
jacobian_result = get_jacobian_results(params, points)

# Print the Jacobian matrix
jacobian_result_matrix = np.vstack(jacobian_result)
print("Jacobian matrix (3x3):")
print(jacobian_result_matrix)
```

<br>
<center><img src="../assets/img/math/calculus/jacobian/5.png" alt="Drawing" style="width: 400px;"/></center>
<br>

<br>

[Calculus 관련 글 목차](https://gaussian37.github.io/math-calculus-table/)

<br>