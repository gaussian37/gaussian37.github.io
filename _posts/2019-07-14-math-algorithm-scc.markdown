---
layout: post
title: SCC(Strongly Connected Component)
date: 2019-07-14 00:00:00
img: math/algorithm/scc/scc.png
categories: [math-algorithm] 
tags: [SCC, Strongly Connected Component] # add tag
---

- 이번 글에서는 SCC에 대하여 알아보도록 하겠습니다. SCC는 Strongly Connected Componet의 약자입니다.

<br>

## SCC의 정의 및 용도

- Strongly Connected의 정의 : 모든 정점에서 모든 정점으로 이동할 수 있는 관계
    - 무향 그래프에서는 의미가 없고 방향 그래프에서만 의미가 있습니다. 
- SCC(Strongly Connected Component) : 그래프를 Stronly Connected 하게 서브 그래프로 나눔
    - 각 Component 내부에서는 모든 정점끼리는 서로 이동이 가능하게 됩니다.
    - Component 끼리는 Strongly Connected 하지 않습니다.

<center><img src="image.JPG" alt="Drawing" style="width: 600px;"/></center>

- 위와 같은 방향 그래프에서는 색을 나눈 기준으로 SCC를 만들 수 있습니다.
- 위 그래프에서는 각각 초록색 노드들 끼리, 파란색 노드들 끼리 그리고 회색 노드들 끼리는 모두 연결되어 있어서 이동이 가능합니다. 즉 Strongly Connected 합니다.
- 반면 초록색 Component, 파란색 Component 그리고 회색 Component들 끼리는 Stronly Connected 관계가 아닙니다. 즉, 서로 이동이 가능한 관계는 아닙니다.
    - 왜냐하면 만약 SC 관계라면 Component로 분리되지 않기 때문이지요.
    - 이 때, 한쪽 Component에서 다른쪽 Component로 이동이 가능한 방향 그래프 관계를 가지게 됩니다. 즉, Component끼리는 DAG(Direct Acyclic Graph) 관계를 가지게 됩니다.

<br>

- 위에서 정의한 SCC 내용을 보면 언제 SCC를 사용할 수 있을까요?
- 대표적으로 사용 가능한 케이스는 


<br>

## Kosaraju 알고리즘


<br>


## Tarjan 알고리즘

