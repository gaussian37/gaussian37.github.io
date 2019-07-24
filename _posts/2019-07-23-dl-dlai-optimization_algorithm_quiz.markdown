---
layout: post
title: Optimization Algorithm 퀴즈 
date: 2019-07-23 00:00:00
img: dl/dlai/dlai.png
categories: [dl-dlai] 
tags: [python, deep learning, optimization, Local Optima] # add tag
---

- 이전 글 : [Problem of local optima](https://gaussian37.github.io/dl-dlai-problem_of_local_optima/)

- Optimization 알고리즘 관련하여 퀴즈를 한번 풀어보겠습니다.

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/1.PNG" alt="Drawing" style="width: 800px;"/></center>

- 앞의 글을 읽어 보셨다면 layer는 대괄호를 사용하여 표현하였고, 미니 배치는 중괄호를 그리고 노드는 소괄호를 이용하여 표시하였습니다.

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/2.PNG" alt="Drawing" style="width: 800px;"/></center>

- 1번 지문을 보면 training set을 one-pass로 학습 시키는 미니배치 그래디언트 디센트가 사실상 배치 그래디언트 디센트와 같으므로 더 빠르다고 할 수 없습니다.
- 미니 배치 그래디언트 디센트를 구현할 때, 같은 미니 배치에서는 벡터라이제이션을 이용하여 연산을 빠르게 하되, 다른 미니 배치 끼리는 벡터라이제이션을 사용할 수 없이 for문을 사용해서 구현해야 합니다.
- 미니 배치 그래디언트 디센트의 데이터 크기는 1보다 크고 m(데이터 전체)보다 작기 때문에 배치 그래디언트 디센트 보다 연산 속도가 빠릅니다.  

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/3.PNG" alt="Drawing" style="width: 800px;"/></center>

- 미니 배치의 사이즈가 1개 ~ m개 사이인 이유에 대하여 확인하는 문제입니다. 여기서 m은 데이터 전체의 갯수 입니다.
- 1번 설명은 미니 배치 사이즈가 1개가 아니라 m개일 때의 설명입니다. 배치의 크기가 데이터 전체라면 모든 데이터를 다 보고 학습을 해야 하는 비효율성이 있습니다.
- 2번 설명은 배치 사이즈가 1개일 때의 단점 설명입니다. 배치 사이즈가 1개이면 벡터라이제이션을 사용할 수 없어서 계산 효율성이 떨어집니다.
- 3번 설명은 미니 배치 사이즈가 m개일 때가 아니라 1개일 때의 설명입니다. stochastic gradient descent는 통상적으로 배치 사이즈가 1개일 때를 나타냅니다. 
- 4번 설명은 배치 그래디언트 디센트 설명과 배치 그래디언트 디센트의 단점을 잘 설명하였습니다.

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/4.PNG" alt="Drawing" style="width: 800px;"/></center>

- 위 비용 그래프와 같이 진동이 심한 경우는 미니 배치 그래디언트 디센트 알고리즘을 사용한 경우입니다.
- 왜냐하면 미니 배치 마다 데이터의 분포가 다르기 때문에 이전 배치에서 학습한 결과가 이번 배치의 데이터에서 성능이 안좋을 수 있기 때문입니다.
- 하지만 배치 그래디언트 디센트에서는 모든 데이터를 한번에 보기 때문에 위와 같은 진동이 나타난다면 뭔가 이상한 상태입니다.
- 이 내용을 잘 설명한 것이 4번째 지문입니다.

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/5.PNG" alt="Drawing" style="width: 800px;"/></center>

- 위 문제를 읽어보면 답은 간단하게 구할 수 있습니다. 문제에서도 직접 계산할 필요가 없다고도 적어놨습니다.
- 문제는 편향 보정의 역할을 묻는 것이고 편향 보정을 하면 초깃값에서도 실측값과 유사하게 값을 얻을 수 있다는 원리를 묻는 것입니다.
- 3번 지문을 보면 편향 보정을 하면 실측값인 10에 가깝고 편향 보정을 하지 않으면 실측값 보다 작은 값을 가지게 되므로 3번 지문이 정답입니다.
    - 초깃값이 0이었기 때문에 실측값 보다 작은 값을 초기에 가지게 됩니다.
    
<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/6.PNG" alt="Drawing" style="width: 800px;"/></center>

- 러닝 레이트는 점점 감소해야 합니다. 학습 초기에는 넓은 보폭으로 움직이다가 학습이 진행될수록 좁은 보폭으로 움직여야 최솟값에 도달할 수 있습니다.
- 1번 항목은 러닝 레이트가 점점 증가하므로 적당하지 않습니다.

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/7-1.PNG" alt="Drawing" style="width: 800px;"/></center>

<center><img src="../assets/img/dl/dlai/optimization_quiz/7-2.PNG" alt="Drawing" style="width: 800px;"/></center>

- 지수 가중 평균에서 $$ \beta $$ 값을 증가 시키면 평균의 범위가 더 늘어 나게 되어 노이즈에 둔감해 지지만 입력값 또한 둔감해져 들어오는 입력값을 반영하는 데 지연시간이 생깁니다.
    - 따라서 지연 시간 때문에 그래프가 살짝 오른쪽으로 이동하게 됩니다.
- 지수 가중 평균에서 $$ \beta $$ 값이 감소되면 입력 값에 민감해 져서 값의 반영이 좀 더 빨리 됩니다. 하지만 노이즈에 또한 민감해 져서 진동이 더 잦아집니다.
 
<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/8.PNG" alt="Drawing" style="width: 800px;"/></center>

- 지수 가중 평균의 $$ \beta $$가 커질수록 변화율이 큰 수직축에는 +, - 평균값에 가까워져 진동이 줄어들게 됩니다. 수평축은 항상 한 방향으로의 값이기 때문에 지수 가중 평균을 하면 좀 더 폭이 넓어지게 됩니다.

<br>

<center><img src="../assets/img/dl/dlai/optimization_quiz/9.PNG" alt="Drawing" style="width: 800px;"/></center>

- 위 항목 중 선택된 항목은 학습에 도움이 되는 방법들입니다. weight를 0으로 만드는것은 효과가 있다고 할 수 없으며 나머지는 효과가 있는 방법들입니다.

<center><img src="../assets/img/dl/dlai/optimization_quiz/10.PNG" alt="Drawing" style="width: 800px;"/></center> 

- 2,3,4 항목은 Adam을 잘 설명하고 있는 내용이고 1번과 같이 Adam은 배치 그래디언트 디센트에 써야 한다는 내용은 없습니다.