---
layout: post
title: 점들의 반시계 방향 정렬
date: 2020-03-01 00:00:00
img: math/algorithm/algorithm.png
categories: [math-algorithm] 
tags: [ccw, counter clockwise, 반시계 방향, 정렬] # add tag
---

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

- 볼록 다각형을 이룰 수 있는 점들이 있을 때, 차례 대로 선분을 이으면 볼록 다각형이 되도록 정렬해야 한다면 어떻게 할 수 있을까요?

<br>
<center><img src="../assets/img/math/algorithm/ccw_sort/0.png" alt="Drawing" style="width: 600px;"/></center>
<br>

- 위 그림을 보면 점들이 있을 때 왼쪽 처럼 순서대로 있으면 선으로 이었을 때, 다각형이 되지만 오른쪽 처럼 순서대로 있으면 선으로 이었을 때 다각형이 되지 않습니다.
- 점들이 주어졌을 때, 왼쪽과 같은 순서대로 점들을 만들어 보도록 하겠습니다. 이 글의 내용을 이해하려면 [ccw](https://gaussian37.github.io/math-algorithm-ccw/)를 이해하셔야 합니다.

<br>
<center><img src="../assets/img/math/algorithm/ccw_sort/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 어떤 다각형의 꼭짓점이 총 n개가 있다면 임의의 점 하나를 기준 점으로 잡으면 n-1개가 생깁니다. 
- 정렬의 기준을 세울 때, (비교 함수에서 i가 첫번째 인자, j가 두번째 인자) 기준점 → i번째 점을 이은 벡터를 기준으로 기준점 → j번째 점을 이은 벡터가 반시계 방향에 있으면 True를 그렇지 않으면 False로 간주합니다.
- 만약 두 벡터가 일직선 상에 있다면 기준점과의 거리로 비교하여 i가 기준점과 더 가까우면 True로 그렇지 않으면 False로 간주합니다.

```cpp
typedef struct Point{
    int x;
    int y;
}Point;

long long dist(const Point* p1, const Point* p2){
    return (long long)(p1->x - p2->x)*(p1->x - p2->x) + (long long)(p1->y - p2->y)*(p1->y - p2->y);
}

// right가 left의 반시계 방향에 있으면 true이다.
// true if right is counterclockwise to left.
Point p;
int comparator(const Point* left, const Point* right){
    int ret;
    int direction = ccw(p, left, right);
    if(direction == 0){
        ret = (dist(&p, left) <= dist(&p, right));
    }
    else if(direction == 1){
        ret = 1;
    }
    else{
        ret = 0;
    }
    return ret;
}

```

<br>

- 위 `comparator`를 정렬할 때 사용하면 됩니다.
- 그러면 다음 코드에서 실제 퀵소트를 이용하여 어떻게 정렬하면 되는 지 다루어 보겠습니다.

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

