---
layout: post
title: KD-Tree 정리
date: 2023-03-02 00:00:00
img: math/algorithm/kdtree/0.png
categories: [math-algorithm] 
tags: [kdtree, normal estimation] # add tag
---

<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

## **목차**

<br>

- ### [KD-Tree의 정의](#kd-tree의-정의-1)
- ### [KD-Tree의 차원 확장](#kd-tree의-차원-확장-1)
- ### [Nearest Search KD-Tree](#nearest-search-kd-tree-1)
- ### [KNN Search KD-Tree](#knn-search-kd-tree-1)
- ### [Radius Search KD-Tree](#radius-search-kd-tree-1)
- ### [Hybrid KNN and Radius Search KD-Tree](#hybrid-knn-and-radius-search-kd-tree)

<br>

## **KD-Tree의 정의**

<br>

- 이번 글에서 다룰 `KD-Tree`는 $$ K $$ 개의 `Dimension`을 이용하여 `Binary Tree`를 만드는 자료 구조를 의미합니다. 즉, $$ K $$ 개 차원의 `Binary Search Tree`의 자료구조를 의미합니다. `KD-Tree`를 이용하면 데이터셋에 대하여 평균적으로 $$ \text{O}(\log{n}) $$ 만에 탐색이 가능하므로 효율적으로 탐색이 가능해 집니다.
- `KD-Tree`는 `Binary Search Tree`를 $$ K $$ 차원으로 확장한 것이므로 2차원 또는 3차원의 데이터를 다룰 때 많이 사용됩니다. 예를 들어 3차원 상의 점들이 좌표값을 가질 때, 특정 점과 가장 가까운 점 또는 가장 가까운 점군들을 찾을 때 많이 사용됩니다. 이번 글에서도 다룰 예제는 **3차원 포인트 클라우드를 이용하여 `KD-Tree`를 만들고 임의의 점과 가장 가까운 점들을 찾는 예제**를 위주로 살펴볼 예정입니다.

<br>

#### **Binary Search Tree 예시**

<br>

- 먼저 `Binary Search Tree`에 대하여 간략히 살펴보도록 하겠습니다. `Binary Search Tree`는 데이터를 효율적으로 검색하기 위하여 특정 노드를 기준으로 그 노드의 값보다 작은 값은 왼쪽으로 구성하고 그 노드의 값보다 큰 값은 오른쪽으로 구성하는 자료 구조를 의미합니다.
- 이와 같이 트리를 구성하였을 때, $$ \text{O}(\log{n}) $$ 으로 자료를 검색할 수 있다는 효율성이 있습니다. 아래 예시와 같습니다.

<br>

- $$ 3, 7, 10, 12, 17, 21 $$

<br>

<br>
<center><img src="../assets/img/math/algorithm/kdtree/1.png" alt="Drawing" style="width: 400px;"/></center>
<br>

- 위 예시와 같이 6개의 데이터가 있을 때, 오름차순 정렬 후 가운데 노드를 기준으로 트리를 만들어 가면 위 그림과 같이 트리를 생성할 수 있습니다.


<br>

#### **2차원 KD-Tree 만들기**

<br>



<br>

## **KD-Tree의 차원 확장**

<br>

<br>

## **Nearest Search KD-Tree**

<br>


<br>

## **KNN Search KD-Tree**

<br>

<br>

## **Radius Search KD-Tree**

<br>

<br>

## **Hybrid KNN and Radius Search KD-Tree**

<br>

<br>



<br>

[알고리즘 관련 글 목차](https://gaussian37.github.io/math-algorithm-table/)

<br>

