---
layout: post
title: 평균필터(Average Filter)
date: 2019-06-22 00:00:00
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [평균 필터, average filter] # add tag
---

<br>

[Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>

- 출처 : 칼만필터는 어렵지 않아
- 평균은 데이터의 총합을 데이터 갯수로 나눈 값을 말합니다. 
- 수식으로 정의하면 다음과 같습니다.

<br>

- $$ \bar{x_{k}} = \frac{x_{1} + x_{2} + ... + x_{k}}{k} $$

<br>

- 위 식 처럼 데이터를 모두 모아서 한꺼번에 계산하는 식을 배치식(batch expression) 이라고 합니다.
    - 이러한 batch expression의 문제점은 데이터가 추가 되면 모든 데이터를 다시 더한 후 k + 1 로 나눠야 한다는 비효율적인 면이 있습니다. 
- 반면 이번 글에서 살펴볼 `재귀식(recursive expression)`을 사용하면 이전 결과를 다시 활용하기 때문에 효율적입니다.
- `재귀식`은 **이전 결과를 사용**하기 때문에 **계산 효율과 메모리 저장공간의 측면에서 유리**합니다.
    - 데이터가 수백만개를 넘어가면 매번 모든 값을 합하는 계산은 상당히 느려지기 떄문에 배치식은 상당히 불리합니다.
    - 또한 배치식에서는 평균을 구하려면 데이터를 모두 저장하고 있어야 하지만, `재귀식`은 **이전 평균값**과 **추가된 데이터** 그리고 **데이터 갯수만 저장**하면 됩니다.
- 위의 배치식을 재귀식으로 바꿔보도록 하겠습니다.

<br>

- $$ \bar{x}_{k} = \frac{x_{1} + x_{2} + ... + x_{k}}{k} $$

<br>

- 위 식의 양변에 k를 곱합니다.

<br>

- $$ k \ \bar{x}_{k} = x_{1} + x_{2} + ... + x_{k} $$

<br>

- 위 식의 양변을 `k-1`로 나눈 뒤, $$ x_{k} $$만 따로 분리해서 우변을 두 개의 항으로 나누어 보겠습니다.

<br>

- $$ \frac{k}{k-1} \bar{x}_{k} = \frac{x_{1} + x_{2} + ... + x_{k}}{k-1} = \frac{x_{1} + x_{2} + ... + x_{k-1}}{k-1} + \frac{x_{k}}{k-1}$$

<br>

- $$ \frac{k}{k-1} \bar{x}_{k} = \bar{x}_{k-1} + \frac{x_{k}}{k-1}$$

<br>


- 이제 양변을 $$ \frac{k}{k-1} $$로 나누어 보겠습니다.

<br>

- $$ \bar{x}_{k} = \frac{k-1}{k} \bar{x}_{k-1} + \frac{1}{k} x_{k} $$

<br>

- 즉, 평균을 계산하려면 직전 평균값인 $$ \bar{x}_{k-1} $$과 데이터의 갯수 $$ k $$ 그리고 새로 추가된 데이터 $$ x_{k} $$만 있으면 됩니다.

<br>

- 식을 좀 더 간결하게 나타내려면 우변의 변수에 곱해지는 상수의 관계를 확인해 보면 됩니다.

<br>

- $$ \alpha = \frac{k-1}{k} $$

<br>

- $$ \alpha = \frac{k-1}{k} = 1 - \frac{1}{k} $$ 

<br>

- $$ \frac{1}{k} = 1 - \alpha $$

<br>

- 따라서 $$ \bar{x}_{k} = \frac{k-1}{k} \bar{x}_{k-1} + \frac{1}{k} x_{k} = \alpha \bar{x}_{k-1} + (1-\alpha)x_{k} $$ 관계가 성립합니다.
- 위 식을 `평균 필터(averaging filter)` 라고 합니다.

<br>

- 평균 필터는 평균 계산 외에 센서 초기화에도 유용하게 사용됩니다. 체중계의 경우 무게 센서는 여러 이유로 영점이 바뀌게 되는데 처음 일정 시간동안의 센서값의 평균을 초기값으로 잡아 영점을 잡으면 초기화 작업을 쉽게 할 수 있습니다.
- 만약 이 작업을 배치식으로 하면 메모리 값도 많이 필요하게 됩니다. 하지만 재귀식을 이용하면 이전 스텝의 값과 샘플의 갯수만 기억하면 됩니다.

<br>

### 평균 필터 C 코드

<br>

```cpp
int num_of_sample;
double prev_average;

double AvgFilter(double x){

    double average, alpha;

    // 샘플 수 +1 (+1 the number of sample)
    num_of_sample += 1;

    // 평균 필터의 alpha 값 (alpha of average filter)
    alpha = (num_of_sample - 1) / (num_of_sample + 0.0);

    // 평균 필터의 재귀식 (recursive expression of average filter)
    average = alpha * prev_average + (1 - alpha) * x;

    // 평균 필터의 이전 상태값 업데이트 (update previous state value of average filter)
    prev_average = average;

    return average;
}
```

<br>

- 정리하면 재귀식인 평균 필터를 사용하면 직전 평균값과 데이터 갯수만 알아도 쉽게 평균을 구할 수 있습니다.
- 특히 데이터가 순차적으로 입력되는 경우 평균 필터를 사용하면 **데이터를 저장할 필요가 없고 계산 효율도 높습니다.**
- 만약 데이터를 실시간으로 처리해야 한다면 재귀식 형태의 필터가 필수 입니다.
- 그리고 평균을 취함으로써 노이즈가 제거되는 효과 또한 있습니다.

<br>

[Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>