---
layout: post
title: 이동 평균 필터(moving average filter)
date: 2019-06-22 00:00:01
img: autodrive/ose/kalman_filter.jpg
categories: [autodrive-ose] 
tags: [이동 평균 필터, moving average filter] # add tag
---

<br>

- [Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>

- 출처 : 칼만필터는 어렵지 않아
- 이동 평균은 모든 측정 데이터가 아니라, 지정된 갯수의 최근 측정값만 가지고 계산한 평균입니다.
- 새로운 데이터가 들어오면 가장 오래된 데이터는 버리는 방식으로 데이터 갯수를 일정하게 유지하면서 평균을 구합니다.
- n개의 데이터에 대한 이동 평균을 수식으로 표현하면 다음과 같습니다.
    - 　$$ \bar{x}_{k} = \frac{x_{k-n+1} + x_{k-n+2} + \cdots + x_{k}}{n} $$
- 이동 평균을 사용하는 이유는 시간의 변화에 따른 변화를 반영하기 위해서 입니다.
    - 단순 평균을 사용하면 전체 데이터에 대한 평균만을 측정하므로 최근 데이터의 변화량을 감지할 수 없습니다.
- 이동 평균을 재귀식으로 바꿔보도록 하겠습니다.
- 　$$ \bar{x}_{k} - \bar{x}_{k-1} = \frac{x_{k-n+1} + x_{k-n+2} + \cdots + x_{k-1} +x_{k}}{n} - \frac{x_{k-n} + x_{k-n+1} + \cdots + x_{k-1}}{n} = \frac{x_{k} - x_{k-n}}{n}$$
- 따라서 $$ \bar{x}_{k} = \bar{x}_{k-1} + \frac{x_{k} - x_{k-n}}{n} $$ 로 정의할 수 있습니다.

<br>

- 위 식이 이동 평균 필터(moving average filter)입니다.
- 평균 필터의 경우에는 계산 속도도 빨라지고 저장해야 할 값 또한 몇개 없어서 메모리도 효율적으로 쓸 수 있었습니다.
- 하지만 이동 평균 필터 같은 경우에는 계산 속도는 빨라지지만 저장해야 할 값은 배치식과 다르지 않습니다.
    - 왜냐하면 $$ k $$번째 식에서 필요한 값 중 $$ x_{k-n} $$이 있습니다. 그러면 $$ k + 1 $$번쨰 식에는 $$ x_{k-n+1} $$ 값이 필요 하게 됩니다.
    - 즉, n개의 최근 데이터 $$ x_{k-1}, x_{k-2}, .. , x_{k-n} $$를 모두 보관하여 있어야 새로운 데이터가 입력될 때 마다 업데이트를 할 수 있습니다.
- 만약 시간에 해당하는 $$ n $$의 값이 작으면 계산 속도에서도 큰 이점이 없고 코드만 가독성이 떨어지게 되는 단점이 생깁니다.
    - 만약 이동 평균을 모르는 사람이 코드를 보면 잘 모르겠죠??
- 또한 이동 평균에서 고려해야 할 점은 평균 계산에 동원되는 데이터의 갯수 $$ n $$ 입니다.
    - 이 때 고려해야할 점은 입력 값이 노이즈 제거와 변화 민감성 중에 어떤 것이 중요한 지 생각해야 합니다.
    - 만약 측정하는 값이 빠르게 변한다면 이동평균의 데이터의 갯수를 줄여 변화를 빨리 쫓는게 좋습니다.
        - 즉 변화 민감성에 초점을 주는 것입니다.
    - 반면 움직임이 느리다면 이동 평균의 데이터의 갯수를 늘려 노이즈 제거를 하는 것이 바람직합니다.

<br>

### 이동 평균 코드

- python 코드

```python
import queue
class MovAvgFilter:
    
    # 이전 스텝의 평균
    prevAvg = 0
    # 가장 최근 n개의 값을 저장하는 큐
    xBuf = queue.Queue()
    # 참조할 데이터의 갯수
    n = 0
    
    def __init__(self, _n):
        # 초기화로 n개의 값을 0으로 둡니다.
        for _ in range(_n):
            self.xBuf.put(0)
        # 참조할 데이터의 갯수를 저장합니다.
        self.n = _n
    
    def movAvgFilter(self, x):
        # 큐의 front 값은 x_(k-n) 에 해당합니다.
        front = self.xBuf.get()
        # 이번 스텝에 입력 받은 값을 큐에 넣습니다.
        self.xBuf.put(x)
        
        avg = self.prevAvg + (x - front) / self.n     
        self.prevAvg = avg
        
        return avg      
```

<br>

- cpp 코드

<br>

- 마지막으로 정리하면 이동평균 필터는 평균 계산에 포함되는 데이터 갯수가 많으면 노이즈 제거 성능은 좋아지지만 측정 신호의 변화가 제때 반영되지 않고 시간 지연이 생깁니다.
- 반대로 데이터의 갯수가 적으면 측정 신호의 변화는 잘 따라가지만 노이즈가 잘 제거되지 않습니다.
- 따라서 신호의 특성을 잘 파악하여 이동평균의 데이터 갯수를 선정해야합니다.

<br>

- [Optimal State Estimation 글 목록](https://gaussian37.github.io/autodrive-ose-table/)

<br>